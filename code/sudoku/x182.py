"""x182: Sparse chain pooling to expunge converged chains.

Evidence
--------
Analysis of x179p (K_H=1, K_L=4, carry_H=copy_top1, carry_L=all) revealed:

1. z_L chains converge rapidly during inference:
   - H step 0: z_L_cos ≈ -0.02 (diverse)
   - H step 3: z_L_cos ≈ 0.90
   - H step 5+: z_L_cos ≈ 0.99 (nearly identical)

2. Training accelerates this convergence:
   - Untrained (step 0): z_L maintains diversity (cos ≈ 0.47 at H=16)
   - Trained (step 36k): z_L converges within 3-5 H steps

3. Winner selection is effectively random after convergence:
   - Streak lengths match uniform distribution (p=0.25)
   - All 4 chains win ~25% of the time

This means we're paying 4x compute for 1x effective diversity after H step 5.

Rationale
---------
If chains converge to nearly identical states, maintaining all 4 is wasteful.
We can expunge redundant chains and use freed slots for new samples.

Proposed Design
---------------
- Fixed pool of P = B * K chains (e.g., 192 * 4 = 768)
- Track `sample_id[p]` for which sample each slot belongs to (-1 = free)
- Each sample starts with K chains
- After each H step, check pairwise z_L cosine similarity within each sample
- If pair exceeds threshold (e.g., 0.95), nuke the higher-loss chain
- When ≥K slots are free, assign them to a new sample

Efficiency: worst case 3 idle slots = 3/768 = 0.4% wasted compute.

Expected benefit: ~3x throughput on samples that converge early, while
maintaining full K-chain diversity on early H steps where it matters.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import field
from typing import Literal, TypedDict

import dataclasses
import math

from torch import Tensor, nn

import torch

from data import GPUCachedSudoku, augment_sudoku
from experiment import (
    HALT_TOKEN_ID,
    ExperimentBase,
    main,
    setup_muon_optimizers,
)
from model import EMA, TRM3
from util import set_seed


class ForwardResult(TypedDict):
    losses: Tensor  # [P] per-chain losses (inf for inactive)
    logits: Tensor  # [P, S-1, V]
    q_halt: Tensor  # [P]
    z_H: Tensor  # [P, S, C]
    z_L: Tensor  # [P, S, C]


class Experiment:
    """Sparse chain pooling experiment."""

    seed: int = 42
    device: torch.device | None = None

    config: TRM3.Config = TRM3.Config(
        K_H=1,
        K_L=4,
        carry_H="copy_top1",
        carry_L="all",
        z_L_init_svd=True,
    )

    # Dataset
    data_dir: str = "/opt/scratch/datasets/sudoku-extreme-1k-aug-1000"

    # Training
    total_train_steps: int = 36_000
    batch_size: int = 192  # Samples per batch (for data loading)
    K: int = 4  # Chains per sample (must match config.num_effective_heads = K_H * K_L)
    grad_accum_steps: int = 1
    reset_steps: list[int] = []  # noqa: RUF012
    max_train_sec: float = 60 * 60 * 9  # 9 hours

    augment_sudoku: bool = True

    # Eval
    eval_method: Literal["standard", "fast", "wta"] = "standard"
    eval_batch_size: int | None = 768  # Must match train pool size for compile
    eval_every_steps: int = 2_000
    max_eval_samples: int = 38_400
    checkpoint_steps: list[int] = list(range(0, 75_000, 500))  # noqa: RUF012
    enable_reset_retry: bool = False
    reset_threshold: float = 0.0

    # Regularization
    label_smoothing: float = 0.2

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9
    ema_warmup_steps: int = 5_000

    # ACT config
    # max_steps_schedule[train_step] = (max_h_steps, train_q_halt)
    max_steps_schedule: dict[int, tuple[int, bool]] = {0: (16, True)}  # noqa: RUF012
    halt_exploration_prob: float = 0.1
    q_halt_weight: float = 0.05

    # Sparse pooling config
    expunge_threshold: float = 0.95  # z_L cosine sim threshold for expunge
    min_expunge_step: int = 0  # Don't expunge before this H step

    cast_model_to_dtype: bool = True
    max_pending_samples: int = 1000

    def __init__(self) -> None:
        self.setup_model()  # Sets self.device and self.dtype

        self.optimizer1: torch.optim.Optimizer | None = None
        self.optimizer2: torch.optim.Optimizer | None = None

        self.current_step = 0
        self.best_acc = 0.0
        self._grad_accum_counter = 0

        self.act_steps_history: list[float] = []
        self.halt_steps_histogram = [0] * max(
            v[0] for v in self.max_steps_schedule.values()
        )
        self._train_loader: GPUCachedSudoku | None = None
        self._data_iter: Iterator[tuple[Tensor, Tensor]] | None = None

        self._state = TrainingState(
            num_chains=self.num_chains,
            seq_len=self.config.seq_len,
            hidden_size=self.config.hidden_size,
            K=self.K,
            device=self.device,
            dtype=self.dtype,
        )
        self._pending_inputs = torch.empty(
            0,
            self.config.seq_len - 1,
            device=self.device,
            dtype=torch.long,
        )
        self._pending_labels = torch.empty(
            0,
            self.config.seq_len - 1,
            device=self.device,
            dtype=torch.long,
        )

    def setup_model(self) -> None:
        set_seed(self.seed, deterministic=True)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = self.config.dtype or torch.bfloat16
        self.model: TRM3 = self.config.setup().to(
            device=self.device,
            dtype=self.dtype if self.cast_model_to_dtype else None,
        )
        self.ema = EMA(self.model, decay=self.ema_decay) if self.use_ema else None

    def setup_optimizers(self) -> None:
        """Must be called before step(). Called by main() before train()."""
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.02,
        )

    def make_train_loader(self) -> GPUCachedSudoku:
        assert self.device is not None
        return GPUCachedSudoku(
            data_dir=self.data_dir,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
            train=True,
        )

    def make_test_loader(self) -> GPUCachedSudoku:
        assert self.device is not None
        bs = self.batch_size if self.eval_batch_size is None else self.eval_batch_size
        return GPUCachedSudoku(
            data_dir=self.data_dir,
            device=self.device,
            dtype=self.dtype,
            batch_size=bs,
            train=False,
            shuffle=False,
        )

    @property
    def num_chains(self) -> int:
        return self.batch_size * self.K

    @property
    def max_reasoning_steps(self) -> int:
        return max(v[0] for v in self.max_steps_schedule.values())

    def step(self, inputs: Tensor, labels: Tensor) -> dict[str, Tensor] | None:
        if self.current_step >= self.total_train_steps:
            return None
        if self.augment_sudoku:
            inputs, labels = augment_sudoku(inputs, labels)

        valid_keys = [k for k in self.max_steps_schedule if k <= self.current_step]
        max_h_steps, train_q_halt = self.max_steps_schedule[max(valid_keys)]
        state = self._state

        self._enqueue(inputs, labels)
        self._fill_pending()

        forward_result = self._forward(state)
        active_at_forward = state.batch_index >= 0
        active_samples, winner_chains = state.select_winners(forward_result["losses"])

        if len(active_samples) > 0:
            loss = forward_result["losses"][winner_chains].mean()
            if train_q_halt:
                with torch.no_grad():
                    predictions = forward_result["logits"][winner_chains].argmax(dim=-1)
                    correct = (
                        (predictions == state.labels[active_samples])
                        .all(dim=-1)
                        .float()
                    )
                loss = (
                    loss
                    + self.q_halt_weight
                    * nn.functional.binary_cross_entropy_with_logits(
                        forward_result["q_halt"][winner_chains],
                        correct,
                        reduction="mean",
                    )
                )
            loss.backward()

        self._update_weights()

        if len(active_samples) > 0:
            chains = state.chain_indices[active_samples]
            valid = chains >= 0
            valid_chains = chains[valid]
            state.z_L[valid_chains] = forward_result["z_L"][valid_chains].detach()
            winner_z_H = (
                forward_result["z_H"][winner_chains]
                .detach()
                .unsqueeze(1)
                .expand(-1, self.K, -1, -1)
            )
            state.z_H[valid_chains] = winner_z_H[valid]
            state.h_step[active_samples] += 1

        num_halted = 0
        if len(active_samples) > 0:
            h_steps = state.h_step[active_samples]
            at_max = h_steps >= max_h_steps
            if train_q_halt:
                q_halt_positive = forward_result["q_halt"][winner_chains] > 0
                force_continue = (
                    torch.rand(len(h_steps), device=state.device)
                    < self.halt_exploration_prob
                )
                min_random_steps = torch.randint(
                    low=2,
                    high=max(3, max_h_steps + 1),
                    size=(len(h_steps),),
                    device=state.device,
                )
                halt = (at_max | q_halt_positive) & (
                    ~force_continue | (h_steps >= min_random_steps)
                )
            else:
                halt = at_max
            if halt.any():
                halt_indices = active_samples[halt]
                num_halted = int(halt.sum().item())
                for h_step in state.h_step[halt_indices].cpu().tolist():
                    if h_step > 0:
                        self.act_steps_history.append(h_step)
                        if 1 <= h_step <= len(self.halt_steps_histogram):
                            self.halt_steps_histogram[h_step - 1] += 1
                state.release(halt_indices)

        num_expunged = state.expunge(
            forward_result["losses"].detach(),
            self.expunge_threshold,
            self.min_expunge_step,
        )
        self._fill_pending()

        num_active = int(state.active.sum().item())
        return {
            "lm": (
                forward_result["losses"][active_at_forward].mean().detach()
                if active_at_forward.any()
                else torch.tensor(0.0, device=state.device)
            ),
            "n_active": torch.tensor(
                float(num_active * self.K),
                device=state.device,
            ),
            "n_expunged": torch.tensor(
                float(num_expunged),
                device=state.device,
            ),
            "n_halted": torch.tensor(
                float(num_halted),
                device=state.device,
            ),
            "pool_util": torch.tensor(
                num_active * self.K / state.num_chains,
                device=state.device,
            ),
        }

    def reset_transient_state(self) -> None:
        print(f"  Resetting transient state at step {self.current_step}")
        self.setup_optimizers()
        if self.ema:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.ema.shadow:
                    self.ema.shadow[n].copy_(p.data)
        self._state = TrainingState(
            num_chains=self.num_chains,
            seq_len=self.config.seq_len,
            hidden_size=self.config.hidden_size,
            K=self.K,
            device=self.device,
            dtype=self.dtype,
        )
        self._pending_inputs = torch.empty(
            0,
            self.config.seq_len - 1,
            device=self.device,
            dtype=torch.long,
        )
        self._pending_labels = torch.empty_like(self._pending_inputs)

    # --- Private methods (training) ---

    def _enqueue(self, inputs: Tensor, labels: Tensor) -> None:
        space = self.max_pending_samples - len(self._pending_inputs)
        if space <= 0:
            return
        self._pending_inputs = torch.cat([self._pending_inputs, inputs[:space]])
        self._pending_labels = torch.cat([self._pending_labels, labels[:space]])

    def _fill_pending(self) -> int:
        if len(self._pending_inputs) == 0:
            return 0

        num_to_fill = min(len(self._pending_inputs), self._state.max_fillable)
        if num_to_fill == 0:
            return 0

        carry = self.model.init_carry(num_to_fill)
        z_H = carry["z_H"].detach()
        z_L = carry["z_L"].detach()

        num_filled = self._state.fill(
            self._pending_inputs[:num_to_fill],
            self._pending_labels[:num_to_fill],
            z_H,
            z_L,
        )
        if num_filled > 0:
            self._pending_inputs = self._pending_inputs[num_filled:]
            self._pending_labels = self._pending_labels[num_filled:]
        return num_filled

    def _forward(self, state: TrainingState) -> ForwardResult:
        num_chains = state.num_chains
        active = state.batch_index >= 0

        if not active.any():
            return ForwardResult(
                losses=torch.full(
                    (num_chains,),
                    fill_value=math.inf,
                    device=state.device,
                ),
                logits=torch.zeros(
                    num_chains,
                    self.config.seq_len - 1,
                    self.config.vocab_size,
                    device=state.device,
                    dtype=self.dtype,
                ),
                q_halt=torch.zeros(
                    num_chains,
                    device=state.device,
                    dtype=self.dtype,
                ),
                z_H=state.z_H.clone(),
                z_L=state.z_L.clone(),
            )

        batch_indices = state.batch_index.clamp(min=0)
        chain_inputs = state.inputs[batch_indices]
        chain_labels = state.labels[batch_indices]

        tokens = torch.cat(
            [
                torch.full(
                    (num_chains, 1),
                    fill_value=HALT_TOKEN_ID,
                    device=state.device,
                    dtype=torch.long,
                ),
                chain_inputs,
            ],
            dim=1,
        )

        # Forward all chains (active and inactive - padding for compile)
        assert self.device is not None
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):  # pyright: ignore[reportPrivateImportUsage]
            embeddings = self.model.embed_scale * self.model.embed_tokens(
                tokens,
                self.config.dtype,
            )
            core = (
                self.model._core_compiled  # noqa: SLF001
                if self.model.config.compile_core
                else self.model._core  # noqa: SLF001
            )
            z_H = state.z_H
            z_L = state.z_L
            for _ in range(self.model.config.H_cycles - 1):
                with torch.no_grad():
                    logits, q_halt, z_H, z_L = core(
                        embeddings.detach(),
                        z_H.detach(),
                        z_L.detach(),
                        None,
                    )
            logits, q_halt, z_H, z_L = core(embeddings, z_H, z_L, None)

        logits = logits[:, 1:, :]
        loss = (
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chain_labels.reshape(-1),
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            .reshape(num_chains, -1)
            .mean(dim=-1)
        )
        loss = torch.where(
            active,
            loss,
            torch.full_like(loss, fill_value=math.inf),
        )

        return ForwardResult(
            losses=loss,
            logits=logits,
            q_halt=q_halt,
            z_H=z_H,
            z_L=z_L,
        )

    def _update_weights(self) -> None:
        self._grad_accum_counter += 1
        if self._grad_accum_counter < self.grad_accum_steps:
            return

        self._grad_accum_counter = 0
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        lr_scale = 0.5 * (
            1 + math.cos(math.pi * self.current_step / self.total_train_steps)
        )
        for optimizer in [self.optimizer1, self.optimizer2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lr_scale

        self.optimizer1.step()
        self.optimizer2.step()
        self.model.zero_grad(set_to_none=True)
        self.current_step += 1

        if self.ema and self.current_step >= self.ema_warmup_steps:
            if self.current_step == self.ema_warmup_steps:
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        self.ema.shadow[n].copy_(p.data)
            self.ema.update(self.model)

    def _get_next_batch(self) -> tuple[Tensor, Tensor]:
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

    # --- Evaluation ---
    # Monkey-patch eval methods from ExperimentBase (intentionally not inheriting
    # to avoid pulling in ExperimentBase's full __init__ and training logic).

    evaluate = ExperimentBase.evaluate
    _evaluate_act = ExperimentBase._evaluate_act  # noqa: SLF001
    _evaluate_act_full = ExperimentBase._evaluate_act_full  # noqa: SLF001
    _evaluate_act_haltfast = ExperimentBase._evaluate_act_haltfast  # noqa: SLF001
    _evaluate_act_haltfast_wta = ExperimentBase._evaluate_act_haltfast_wta  # noqa: SLF001

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
        return self.model.L_init.unsqueeze(1).expand(-1, self.config.seq_len, -1)

    def _prepend_halt_token(self, inputs: Tensor) -> Tensor:
        """Prepend HALT token for ACT mode (required by ExperimentBase eval methods)."""
        B = inputs.shape[0]
        halt_tokens = torch.full(
            (B, 1),
            fill_value=HALT_TOKEN_ID,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return torch.cat([halt_tokens, inputs], dim=1)

    def _make_checkpoint(self) -> dict[str, object]:
        checkpoint: dict[str, object] = {
            "model": self.model.state_dict(),
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "step": self.current_step,
            "best_acc": self.best_acc,
        }
        if self.ema:
            checkpoint["ema"] = dict(self.ema.shadow)
        return checkpoint


@dataclasses.dataclass(kw_only=True, slots=True)
class TrainingState:
    """State for sparse chain pooling. Each batch element runs K chains; winner's z_H propagates to all."""

    num_chains: int
    seq_len: int
    hidden_size: int
    K: int
    device: torch.device
    dtype: torch.dtype

    # Per-chain [num_chains]
    z_H: Tensor = field(init=False)
    z_L: Tensor = field(init=False)
    # which batch element owns this chain (-1 if free)
    batch_index: Tensor = field(init=False)

    # Per-batch-element [N = num_chains // K]
    h_step: Tensor = field(init=False)
    inputs: Tensor = field(init=False)
    labels: Tensor = field(init=False)
    # [N, K] which chains belong to this batch element
    chain_indices: Tensor = field(init=False)
    active: Tensor = field(init=False)

    def __post_init__(self) -> None:
        num_batch_elements = self.num_chains // self.K

        # Per-chain tensors [num_chains, seq_len, hidden_size]
        self.z_H = torch.zeros(
            self.num_chains,
            self.seq_len,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.z_L = torch.zeros_like(self.z_H)
        self.batch_index = torch.full(
            [self.num_chains],
            fill_value=-1,
            device=self.device,
            dtype=torch.long,
        )

        # Per-batch-element tensors [num_batch_elements, ...]
        self.h_step = torch.zeros(
            num_batch_elements,
            device=self.device,
            dtype=torch.long,
        )
        self.inputs = torch.zeros(
            num_batch_elements,
            self.seq_len - 1,
            device=self.device,
            dtype=torch.long,
        )
        self.labels = torch.zeros_like(self.inputs)
        self.chain_indices = torch.full(
            [num_batch_elements, self.K],
            fill_value=-1,
            device=self.device,
            dtype=torch.long,
        )
        self.active = torch.zeros(
            num_batch_elements,
            device=self.device,
            dtype=torch.bool,
        )

    @property
    def max_fillable(self) -> int:
        """Max samples that can be filled (limited by free chains AND inactive slots)."""
        # Single GPU sync for both counts
        counts = torch.stack([
            (self.batch_index == -1).sum(),
            (~self.active).sum(),
        ]).tolist()
        num_free, num_inactive = int(counts[0]), int(counts[1])
        return min(num_free // self.K, num_inactive)

    def fill(
        self,
        inputs: Tensor,
        labels: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> int:
        """Fill free chains with new batch elements. Returns number filled.

        Caller should pre-compute max_fillable and pass exactly that many inputs
        for efficiency. This method still validates but assumes caller sized correctly.
        """
        num_to_fill = len(inputs)
        if num_to_fill == 0:
            return 0

        free_chains = (self.batch_index == -1).nonzero(as_tuple=True)[0]
        inactive = (~self.active).nonzero(as_tuple=True)[0]

        # Validate caller's pre-computation (should match if caller used max_fillable)
        max_possible = min(len(free_chains) // self.K, len(inactive))
        if num_to_fill > max_possible:
            num_to_fill = max_possible
            if num_to_fill == 0:
                return 0

        free_chains = free_chains[: num_to_fill * self.K].reshape(num_to_fill, self.K)
        inactive = inactive[:num_to_fill]

        self.z_H[free_chains.reshape(-1)] = z_H[:num_to_fill].reshape(
            -1,
            *z_H.shape[2:],
        )
        self.z_L[free_chains.reshape(-1)] = z_L[:num_to_fill].reshape(
            -1,
            *z_L.shape[2:],
        )
        self.batch_index[free_chains.reshape(-1)] = (
            inactive.unsqueeze(1).expand(-1, self.K).reshape(-1)
        )
        self.chain_indices[inactive] = free_chains
        self.active[inactive] = True
        self.h_step[inactive] = 0
        self.inputs[inactive] = inputs[:num_to_fill].long()
        self.labels[inactive] = labels[:num_to_fill].long()
        return num_to_fill

    def select_winners(
        self,
        losses: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns (active_batch_indices, winner_chain_indices)."""
        if not self.active.any():
            empty: Tensor = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty

        active = self.active.nonzero(as_tuple=True)[0]
        chains = self.chain_indices[active]
        valid = chains >= 0
        loss_mat = losses[chains.clamp(min=0)]
        loss_mat = torch.where(
            valid,
            loss_mat,
            torch.full_like(loss_mat, fill_value=math.inf),
        )
        winner_k = loss_mat.argmin(dim=1)
        winner_chain = chains[torch.arange(len(active), device=self.device), winner_k]
        return active, winner_chain

    def expunge(
        self,
        losses: Tensor,
        threshold: float,
        min_h_step: int,
    ) -> int:
        """Remove converged chains (cosine sim > threshold). Returns count removed."""
        eligible = self.active & (self.h_step >= min_h_step)
        if not eligible.any():
            return 0

        eligible_indices = eligible.nonzero(as_tuple=True)[0]
        chains = self.chain_indices[eligible_indices]
        valid = chains >= 0
        if (valid.sum(dim=1) < 2).all():
            return 0

        z_flat = (
            self.z_L[chains.clamp(min=0).reshape(-1)]
            .reshape(len(eligible_indices), self.K, -1)
            .float()
        )
        z_normalized = z_flat / (z_flat.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = torch.bmm(z_normalized, z_normalized.transpose(1, 2))
        similarity.masked_fill_(
            torch.eye(self.K, device=self.device, dtype=torch.bool).unsqueeze(0),
            value=-2,
        )
        similarity.masked_fill_(~valid.unsqueeze(2), value=-2)
        similarity.masked_fill_(~valid.unsqueeze(1), value=-2)

        above_threshold = similarity > threshold
        upper_triangular = torch.triu(
            torch.ones(self.K, self.K, device=self.device, dtype=torch.bool),
            diagonal=1,
        )
        has_pair = (above_threshold & upper_triangular.unsqueeze(0)).any(dim=(1, 2))
        if not has_pair.any():
            return 0

        loss_matrix = losses[chains.clamp(min=0).reshape(-1)].reshape(
            len(eligible_indices),
            self.K,
        )
        loss_matrix = torch.where(
            valid,
            loss_matrix,
            torch.full_like(loss_matrix, fill_value=-math.inf),
        )
        in_pair = above_threshold.any(dim=2) | above_threshold.any(dim=1)
        loss_matrix = torch.where(
            in_pair,
            loss_matrix,
            torch.full_like(loss_matrix, fill_value=-math.inf),
        )
        expunge_k = loss_matrix.argmax(dim=1)
        expunge_chain = chains[
            torch.arange(len(eligible_indices), device=self.device),
            expunge_k,
        ]
        mask = has_pair & (expunge_chain >= 0)

        if not mask.any():
            return 0

        chains_to_free = expunge_chain[mask]
        self.batch_index[chains_to_free] = -1
        self.chain_indices[eligible_indices[mask], expunge_k[mask]] = -1
        return int(mask.sum().item())

    def release(
        self,
        batch_indices: Tensor,
    ) -> None:
        """Release chains for given batch indices."""
        chains = self.chain_indices[batch_indices]
        self.batch_index[chains[chains >= 0]] = -1
        self.active[batch_indices] = False
        self.chain_indices[batch_indices] = -1


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
