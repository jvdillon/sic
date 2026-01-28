"""x179: Control for x179 family. A clean rewrite of x178d."""

from typing import Literal

import math
import warnings

from experiment import (
    HALT_TOKEN_ID,
    ExperimentBase,
    main,
    setup_muon_optimizers,
)
from model import (
    EMA,
    TRM3,
    ModelConfigProtocol,
    ModelProtocol,
)
from torch import Tensor, nn
from util import set_seed

import torch

from data import GPUCachedSudoku, augment_sudoku


warnings.filterwarnings("ignore", message=".*TF32.*")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# To get bit-for-bit you may need to: rm -rf /tmp/torchinductor_jvdillon/


class Experiment:
    model: ModelProtocol
    device: torch.device | None = None
    seed: int = 42

    config: ModelConfigProtocol = TRM3.Config()

    # Dataset
    data_dir: str = "/opt/scratch/datasets/sudoku-extreme-1k-aug-1000"

    # Training
    total_train_steps: int = 28_000
    batch_size: int = 96
    grad_accum_steps: int = 1
    reset_steps: list[int] = []  # noqa: RUF012
    max_train_sec: float = 60 * 60 * 9  # 9 hours

    augment_sudoku: bool = True
    """Apply all 8 dihedral symmetries + digit permutation."""

    # Eval method: "standard", "fast", or "wta"
    eval_method: Literal["standard", "fast", "wta"] = "standard"
    eval_batch_size: int | None = 384  # None = use batch_size
    eval_every_steps: int = 2_000
    max_eval_samples: int = 38_400  # 100 * 384; -1 = full test set
    checkpoint_steps: list[int] = list(range(0, 75_000, 500))  # noqa: RUF012
    # Reset-retry at inference (#22)
    enable_reset_retry: bool = False
    reset_threshold: float = 0.0  # q_halt < this triggers reset

    # Regularization
    label_smoothing: float = 0.2

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9
    ema_warmup_steps: int = 5_000

    # ACT config
    max_reasoning_steps: int = 16
    # {train_step: max_H} or {} for default
    max_steps_schedule: dict[int, int] = {}  # noqa: RUF012
    halt_exploration_prob: float = 0.1
    q_halt_weight: float = 0.05
    q_halt_warmup_steps: int = 0
    z_L_noise: float = 0.0  # Noise added to z_L init (0 = disabled)

    # Setting cast_model_to_dtype=False increases time-to-completion by 18% or about
    # 35min on bs=96, steps=28k (4hrs vs 3:25hrs).
    # Overall: ((644.684+371.293) / (565.900+301.006) - 1) * 100% = 18%
    # Train  : (644.684 / 565.900 - 1) * 100% = 14%
    # Eval   : (371.293 / 301.006 - 1) * 100% = 23%
    cast_model_to_dtype: bool = True

    def __init__(self):
        self.setup_model()

        self.optimizer1: torch.optim.Optimizer | None = None
        self.optimizer2: torch.optim.Optimizer | None = None

        self.current_step = 0
        self.best_acc = 0.0
        self._grad_accum_counter = 0

        # ACT tracking
        self.act_steps_history: list[float] = []
        self.halt_steps_histogram = [0] * self.max_reasoning_steps
        self._data_iter = None
        self._train_loader = None
        self._act_carry: dict[str, Tensor] | None = None

    def setup_model(self) -> None:
        set_seed(self.seed, deterministic=True)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self.config.dtype
        self.model = self.config.setup().to(
            device=self.device,
            # This converts all parameters to bfloat16 meaning that gradients will be
            # computed as bfloat16.
            dtype=self.dtype if self.cast_model_to_dtype else None,
        )
        self.ema = EMA(self.model, decay=self.ema_decay) if self.use_ema else None

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.02,
        )

    def make_train_loader(self):
        assert self.device is not None
        return GPUCachedSudoku(
            data_dir=self.data_dir,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
            train=True,
        )

    def make_test_loader(self):
        assert self.device is not None
        bs = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.batch_size
        )
        return GPUCachedSudoku(
            data_dir=self.data_dir,
            device=self.device,
            dtype=self.dtype,
            batch_size=bs,
            train=False,
            shuffle=False,
        )

    def _init_act_carry(self) -> dict[str, Tensor]:
        B = self.batch_size
        cfg = self.config
        return {
            "model_carry": self.model.init_carry(B),
            "steps": torch.zeros(B, device=self.device, dtype=torch.long),
            "halted": torch.ones(B, device=self.device, dtype=torch.bool),
            "inputs": torch.zeros(
                B,
                cfg.seq_len - 1,
                device=self.device,
                dtype=torch.long,
            ),
            "labels": torch.zeros(
                B,
                cfg.seq_len - 1,
                device=self.device,
                dtype=torch.long,
            ),
        }

    def step(self, inputs: Tensor, labels: Tensor) -> dict | None:
        if self.current_step >= self.total_train_steps:
            return None
        if self.augment_sudoku:
            inputs, labels = augment_sudoku(inputs, labels)
        return self._step_act(inputs, labels)

    def _step_act(self, inputs: Tensor, labels: Tensor) -> dict:
        """Generalized K_H/K head WTA with configurable pairing and carry policies."""
        B = self.batch_size

        if self._act_carry is None:
            self._act_carry = self._init_act_carry()
        carry = self._act_carry

        # Reset halted puzzles with fresh state
        self._reset_halted_puzzles(carry, inputs, labels)

        # WTA forward pass (samples heads, computes losses, updates carry)
        assert self.device is not None
        inputs_with_halt = self._prepend_halt_token(carry["inputs"])
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            out = self.model.wta_forward(
                inputs_with_halt,
                carry["model_carry"],
                carry["labels"],
                label_smoothing=self.label_smoothing,
                z_L_noise=self.z_L_noise,
            )

        losses = out["losses"]
        winner_idx = out["winner_idx"]
        logits_all = out["logits"]
        q_halt_all = out["q_halt"]

        # WTA: backprop through winner only
        lm_loss = losses[torch.arange(B, device=self.device), winner_idx].mean()

        # q_halt loss on winner
        winner_q_halt = q_halt_all[torch.arange(B, device=self.device), winner_idx]
        winner_logits = logits_all[torch.arange(B, device=self.device), winner_idx]
        labels_flat = carry["labels"].reshape(B, -1)

        with torch.no_grad():
            preds = winner_logits.argmax(dim=-1)
            correct = (preds == labels_flat).all(dim=-1).float()

        running = ~carry["halted"]
        if running.any():
            q_halt_loss = nn.functional.binary_cross_entropy_with_logits(
                winner_q_halt[running],
                correct[running],
                reduction="mean",
            )
        else:
            q_halt_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.current_step < self.q_halt_warmup_steps:
            q_halt_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        total_loss = lm_loss + self.q_halt_weight * q_halt_loss
        total_loss.backward()
        self._update_weights()

        # Update carry state
        carry["model_carry"] = out["carry"]
        carry["steps"] = carry["steps"] + 1
        carry["halted"] = self._halted(winner_q_halt, carry["steps"])

        head0_wins = (winner_idx == 0).float().mean()

        return {
            "lm": lm_loss.detach(),
            "q_halt": q_halt_loss.detach(),
            "head0_wins": head0_wins.detach(),
        }

    def _reset_halted_puzzles(
        self,
        carry: dict[str, Tensor],
        inputs: Tensor,
        labels: Tensor,
    ) -> None:
        """Reset halted puzzles with fresh state and new data."""
        halted = carry["halted"]
        if not halted.any():
            return

        halted_indices = halted.nonzero(as_tuple=True)[0]

        for idx in halted_indices:
            steps_taken = int(carry["steps"][idx].item())
            if steps_taken > 0:
                self.act_steps_history.append(steps_taken)
                if 1 <= steps_taken <= self.max_reasoning_steps:
                    self.halt_steps_histogram[steps_taken - 1] += 1

        n_reset = min(len(halted_indices), len(inputs))
        carry["model_carry"] = self.model.reset_carry_at_indices(
            carry["model_carry"], halted_indices, n_reset
        )
        for i in range(n_reset):
            idx = halted_indices[i]
            carry["steps"][idx] = 0
            carry["inputs"][idx] = inputs[i]
            carry["labels"][idx] = labels[i]
            carry["halted"][idx] = False

    def _get_effective_max_steps(self) -> int:
        """Get max reasoning steps for current training step from schedule."""
        if not self.max_steps_schedule:
            return self.max_reasoning_steps
        valid_keys = [k for k in self.max_steps_schedule if k <= self.current_step]
        if not valid_keys:
            return self.max_reasoning_steps
        return self.max_steps_schedule[max(valid_keys)]

    def _halted(self, winner_q_halt: Tensor, steps: Tensor) -> Tensor:
        B = self.batch_size
        effective_max = self._get_effective_max_steps()
        with torch.no_grad():
            is_last_step = steps >= effective_max
            halted = is_last_step | (winner_q_halt > 0)

            force_continue = (
                torch.rand(B, device=self.device) < self.halt_exploration_prob
            )
            min_steps = torch.randint(
                2,
                max(3, effective_max + 1),
                (B,),
                device=self.device,
            )
            exploration_halt = steps >= min_steps
            halted = halted & (~force_continue | exploration_halt)

            return halted

    def _update_weights(self) -> None:
        """Common weight update logic with gradient accumulation."""
        self._grad_accum_counter += 1

        if self._grad_accum_counter < self.grad_accum_steps:
            return

        self._grad_accum_counter = 0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        assert self.optimizer1 is not None
        assert self.optimizer2 is not None
        r = self.current_step / self.total_train_steps
        lr_scale = 0.5 * (1 + math.cos(math.pi * r))
        for opt in [self.optimizer1, self.optimizer2]:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_scale

        self.optimizer1.step()
        self.optimizer2.step()
        self.model.zero_grad(set_to_none=True)
        self.current_step += 1

        if self.ema is not None and self.current_step >= self.ema_warmup_steps:
            if self.current_step == self.ema_warmup_steps:
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        self.ema.shadow[n].copy_(p.data)
            self.ema.update(self.model)

    def reset_transient_state(self) -> None:
        """Reset optimizer, EMA, and ACT carry. Keeps model weights."""
        print(f"  Resetting transient state at step {self.current_step}")
        self.setup_optimizers()
        if self.ema is not None:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.ema.shadow:
                    self.ema.shadow[n].copy_(p.data)
        self._act_carry = None

    def _make_checkpoint(self) -> dict:
        """Create checkpoint dict with all state."""
        assert self.optimizer1 is not None
        assert self.optimizer2 is not None
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "step": self.current_step,
            "best_acc": self.best_acc,
        }
        if self.ema is not None:
            ckpt["ema"] = dict(self.ema.shadow)
        return ckpt

    def _prepend_halt_token(self, inputs: Tensor) -> Tensor:
        """Prepend HALT token for ACT mode."""
        B = inputs.shape[0]
        halt_tokens = torch.full(
            (B, 1),
            HALT_TOKEN_ID,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return torch.cat([halt_tokens, inputs], dim=1)

    def _get_next_batch(self) -> tuple[Tensor, Tensor]:
        """Get next batch for ACT mode (streaming samples)."""
        if self._train_loader is None:
            self._train_loader = self.make_train_loader()
        if self._data_iter is None:
            self._data_iter = iter(self._train_loader)
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._train_loader)
            batch = next(self._data_iter)
        return batch[0].to(self.device), batch[1].to(self.device)

    # --- Evaluation Functions ---

    @property
    def K(self) -> int:
        """For ExperimentBase: number of WTA heads (= K_L)."""
        return self.config.K_L

    def _init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """For ExperimentBase: Create initial z_H and z_L states for eval (single head)."""
        seq_len = self.config.seq_len
        z_H = self.model.H_init[0].expand(batch_size, seq_len, -1).contiguous()
        z_L = self.model.L_init[0].expand(batch_size, seq_len, -1).contiguous()
        return z_H, z_L

    def _make_z_L_single(self, puzzle_idx: int) -> Tensor:
        """For ExperimentBase: Create z_L for all K heads for a single puzzle (WTA eval).

        Returns: [K, seq_len, hidden]
        """
        del puzzle_idx
        # L_init is [K, hidden], expand to [K, seq_len, hidden]
        return self.model.L_init.unsqueeze(1).expand(-1, self.config.seq_len, -1)

    # Monkey patch in eval stuff from ExperimentBase.
    evaluate = ExperimentBase.evaluate
    _evaluate_act = ExperimentBase._evaluate_act  # noqa: SLF001
    _evaluate_act_full = ExperimentBase._evaluate_act_full  # noqa: SLF001
    _evaluate_act_haltfast = ExperimentBase._evaluate_act_haltfast  # noqa: SLF001
    _evaluate_act_haltfast_wta = ExperimentBase._evaluate_act_haltfast_wta  # noqa: SLF001


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
