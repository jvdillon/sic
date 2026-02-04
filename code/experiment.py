"""TRM experiment framework."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

import contextlib
import math
import pathlib
import random
import sys
import time
import warnings

from evaluation import print_diagnostics
from model import EMA, TRM, ModuleProtocol, TRM1ConfigProtocol, TRM1Protocol
from optimizer import DummyOptimizer, Muon
from torch import Tensor, nn
from torch.optim import AdamW
from util import Tee, numpy_rng, set_seed

import numpy as np
import torch

from data import PuzzleDatasetIterator, augment_sudoku


warnings.filterwarnings("ignore", message=".*TF32.*")


HALT_TOKEN_ID = 11


@dataclass
class _PuzzleIterState:
    """Mutable state for streaming puzzle iteration."""

    puzzle_iter: Iterator[Any]
    max_samples: int
    pending_inputs: Tensor | None = None
    pending_labels: Tensor | None = None
    pending_idx: int = 0
    pending_valid: int = 0
    samples_seen: int = 0


def _get_next_puzzle(state: _PuzzleIterState) -> tuple[Tensor | None, Tensor | None]:
    """Get next puzzle from iterator, respecting max_samples limit."""
    if state.max_samples >= 0 and state.samples_seen >= state.max_samples:
        return None, None

    if state.pending_inputs is not None and state.pending_idx < state.pending_valid:
        inp = state.pending_inputs[state.pending_idx]
        lab = state.pending_labels[state.pending_idx]  # pyright: ignore[reportOptionalSubscript]
        state.pending_idx += 1
        state.samples_seen += 1
        return inp, lab
    try:
        batch = next(state.puzzle_iter)
        state.pending_inputs = batch[0]
        state.pending_labels = batch[1]
        # batch[3] is valid_count: number of non-padded samples in the batch.
        # Last batch may be zero-padded to batch_size; valid_count indicates
        # how many samples are real data vs padding.
        state.pending_valid = batch[3]
        state.pending_idx = 0
        if state.pending_valid > 0:
            assert state.pending_inputs is not None
            assert state.pending_labels is not None
            inp = state.pending_inputs[state.pending_idx]
            lab = state.pending_labels[state.pending_idx]
            state.pending_idx += 1
            state.samples_seen += 1
            return inp, lab
    except StopIteration:
        pass
    return None, None


def _reset_eval_slot(
    exp: Any,
    idx: int,
    inp: Tensor,
    lab: Tensor,
    puzzle_idx: int,
    inputs: Tensor,
    labels: Tensor,
    z_H: Tensor,
    z_L: Tensor,
    steps: Tensor,
    has_winner: Tensor,
    valid: Tensor,
    H_init_single: Tensor,
    L_inits: list[Tensor],
    seq_len: int,
) -> None:
    """Reset a slot for a new puzzle in WTA eval."""
    inputs[idx] = inp
    labels[idx] = lab
    z_H[idx] = H_init_single.expand(exp.K, seq_len, -1)
    if hasattr(exp, "make_z_L_single"):
        z_L[idx] = exp.make_z_L_single(puzzle_idx)
    else:
        for k in range(exp.K):
            z_L[idx, k] = L_inits[k]
    steps[idx] = 0
    has_winner[idx] = False
    valid[idx] = True


class ExperimentBase:
    """TRM experiment with ACT training."""

    model: TRM1Protocol
    device: torch.device  # Set in __init__ if not provided by subclass
    seed: int = 42

    # Dataset
    data_dir: str = "/opt/scratch/datasets/sudoku-extreme-1k-aug-1000"
    eval_every_steps: int = 500
    max_eval_samples: int = 38_400  # 100 * 384; -1 = full test set

    # Training
    total_train_steps: int = 7_000
    batch_size: int = 384
    eval_batch_size: int | None = None  # None = use batch_size
    max_train_sec: float = 60 * 60 * 6
    checkpoint_steps: list[int] = list(range(500, 75_000, 500))  # noqa: RUF012
    reset_steps: list[int] = []  # noqa: RUF012
    augment_sudoku: bool = True

    # Regularization
    label_smoothing: float = 0.2
    diversity_weight: float = 0.0  # Cosine sim penalty between H iterations

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9
    ema_warmup_steps: int = 5_000

    # ACT config
    max_reasoning_steps: int = 16
    halt_exploration_prob: float = 0.1
    q_halt_weight: float = 0.05
    q_halt_warmup_steps: int = 0
    z_L_noise: float = 0.0  # Noise added to z_L init (0 = disabled)
    early_entropy_weight: float = 0.0  # 0 = disabled
    early_entropy_steps: int = 4  # Apply entropy bonus to H1-H4

    # Gradient accumulation
    grad_accum_steps: int = 1

    # Reset-retry at inference (#22)
    enable_reset_retry: bool = False
    reset_threshold: float = 0.0  # q_halt < this triggers reset

    # Eval method: "standard", "fast", or "wta"
    eval_method: Literal["standard", "fast", "wta"] = "standard"

    # K-head WTA config
    K: int = 1  # Number of z_L heads (1 = standard, >1 = WTA)
    auto_register_k_heads: bool = True  # Set False to register L_init_k manually

    # JSD config (for _step_act_wta_jsd)
    jsd_weight_init: float = 10.0
    jsd_weight_final: float = 0.1
    jsd_warmup_steps: int = 2000

    config: TRM1ConfigProtocol = TRM.Config()

    def __init__(self):
        """Set self.model before calling super().__init__()."""
        set_seed(self.seed)
        if not hasattr(self, "device"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self.config.dtype
        self.model = cast(
            TRM1Protocol,
            cast(object, self.config.setup().to(device=self.device, dtype=self.dtype)),
        )

        self.ema = EMA(self.model, decay=self.ema_decay) if self.use_ema else None

        # Register K-1 additional L_init parameters for WTA
        if self.K > 1 and self.auto_register_k_heads:
            self._register_k_heads()

        self.optimizer1: AdamW | DummyOptimizer | None = None
        self.optimizer2: Muon | DummyOptimizer | None = None

        self.current_step = 0
        self.best_acc = 0.0
        if not hasattr(self, "reset_steps"):
            self.reset_steps = []
        self._grad_accum_counter = 0

        # ACT tracking (only used when max_reasoning_steps > 1)
        self.act_steps_history: list[float] = []
        self.halt_steps_histogram = [0] * self.max_reasoning_steps
        self.act_carry: dict[str, Tensor] | None = None

    def _register_k_heads(self) -> None:
        """Register K-1 additional L_init parameters for WTA."""
        for k in range(1, self.K):
            L_init_k = nn.Parameter(
                torch.randn(
                    self.model.L_init.shape, device=self.device, dtype=self.dtype
                )
                * 0.1
            )
            self.model.register_parameter(f"L_init_{k}", L_init_k)
            if self.ema is not None:
                self.ema.shadow[f"L_init_{k}"] = L_init_k.data.clone()

    def setup_optimizers(self) -> None:
        """Set up optimizers. Override in subclass for custom optimizer config."""
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(self.model)

    def make_checkpoint(self) -> dict[str, Any]:
        """Create checkpoint dict with all state including RNG."""
        assert self.optimizer1 is not None
        assert self.optimizer2 is not None
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "step": self.current_step,
            "best_acc": self.best_acc,
            "rng_python": random.getstate(),
            "rng_numpy": numpy_rng.bit_generator.state,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
        }
        if self.ema is not None:
            ckpt["ema"] = dict(self.ema.shadow)
        return ckpt

    def reset_transient_state(self) -> None:
        """Reset optimizer, EMA, and ACT carry. Keeps model weights."""
        print(f"  Resetting transient state at step {self.current_step}")
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(self.model)
        if self.ema is not None:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.ema.shadow:
                    self.ema.shadow[n].copy_(p.data)
        self.act_carry = None

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

    def _init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Create initial z_H and z_L states."""
        seq_len = self.model.config.seq_len
        # These contiguous calls prevent recompiles.
        z_H = self.model.H_init.expand(batch_size, seq_len, -1).contiguous()
        z_L = self.model.L_init.expand(batch_size, seq_len, -1).contiguous()
        return z_H, z_L

    def make_train_loader(self):
        return PuzzleDatasetIterator(
            data_dir=self.data_dir,
            device=self.device,
            batch_size=self.batch_size,
            train=True,
        )

    def make_test_loader(self):
        bs = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.batch_size
        )
        return PuzzleDatasetIterator(
            data_dir=self.data_dir,
            device=self.device,
            batch_size=bs,
            train=False,
            shuffle=False,
        )

    def step(
        self, inputs: Tensor, labels: Tensor, puzzle_ids: Tensor | None = None
    ) -> dict[str, Tensor] | None:
        del puzzle_ids  # Not used in base class (no puzzle_identifier conditioning)
        if self.current_step >= self.total_train_steps:
            return None

        if self.augment_sudoku:
            inputs, labels = augment_sudoku(inputs, labels)
        return self._step_act(inputs, labels)

    def _init_act_carry(self) -> dict[str, Tensor]:
        """Initialize persistent ACT carry state."""
        B = self.batch_size
        z_H, z_L = self._init_z(B)
        return {
            "z_H": z_H.clone(),
            "z_L": z_L.clone(),
            "steps": torch.zeros(B, device=self.device, dtype=torch.long),
            "halted": torch.ones(B, device=self.device, dtype=torch.bool),
            "inputs": torch.zeros(B, 81, device=self.device, dtype=torch.long),
            "labels": torch.zeros(B, 81, device=self.device, dtype=torch.long),
        }

    def _step_act(self, inputs: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """ACT step: one forward pass per call, persist carry across calls."""
        B = self.batch_size
        seq_len = self.model.config.seq_len

        if self.act_carry is None:
            self.act_carry = self._init_act_carry()

        carry = self.act_carry
        halted = carry["halted"]

        if halted.any():
            halted_indices = halted.nonzero(as_tuple=True)[0]

            for idx in halted_indices:
                steps_taken = int(carry["steps"][idx].item())
                if steps_taken > 0:
                    self.act_steps_history.append(steps_taken)
                    if 1 <= steps_taken <= 16:
                        self.halt_steps_histogram[steps_taken - 1] += 1

            n_reset = min(len(halted_indices), len(inputs))
            for i in range(n_reset):
                idx = halted_indices[i]
                carry["z_H"][idx] = self.model.H_init.squeeze(0)
                noise = (
                    torch.randn(
                        seq_len,
                        self.model.config.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * self.z_L_noise
                )
                carry["z_L"][idx] = self.model.L_init.squeeze(0) + noise
                carry["steps"][idx] = 0
                carry["inputs"][idx] = inputs[i]
                carry["labels"][idx] = labels[i]
                carry["halted"][idx] = False

        # Forward pass - single H-cycle for ACT training
        inputs_with_halt = self._prepend_halt_token(carry["inputs"])
        z_H = carry["z_H"]
        z_L = carry["z_L"]

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            # Call forward which runs H_cycles internally
            out = self.model(inputs_with_halt, z_H, z_L)
            logits = out["logits"][:, 1:]  # Strip halt position
            all_logits = [lg[:, 1:] for lg in out["all_logits"]]
            q_halt = out["q_halt"]
            new_z_H = out["z_H"]
            new_z_L = out["z_L"]

        # Compute losses
        labels_flat = carry["labels"].reshape(-1).long()
        lm_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels_flat,
            label_smoothing=self.label_smoothing,
            reduction="sum",
        )

        # Diversity loss on H iterations (all_logits always available now)
        diversity_loss = (
            compute_diversity_loss(all_logits)
            if self.diversity_weight > 0
            else torch.tensor(0.0, device=self.device)
        )

        # Entropy bonus on early steps
        entropy_loss = torch.tensor(0.0, device=self.device)
        if self.early_entropy_weight > 0:
            step_num = carry["steps"][0].item()
            if step_num <= self.early_entropy_steps:
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                entropy_loss = -self.early_entropy_weight * entropy

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = preds == carry["labels"]
            q_halt_target = correct.all(dim=-1).float()

        running = ~carry["halted"]
        if running.any():
            q_halt_loss = nn.functional.binary_cross_entropy_with_logits(
                q_halt[running],
                q_halt_target[running],
                reduction="sum",
            )
        else:
            q_halt_loss = torch.tensor(0.0, device=self.device)

        if self.current_step < self.q_halt_warmup_steps:
            q_halt_loss = torch.tensor(0.0, device=self.device)

        total_loss = (
            lm_loss
            + self.q_halt_weight * q_halt_loss
            + self.diversity_weight * diversity_loss * B * 81
            + entropy_loss
        )
        total_loss.backward()
        self._update_weights()

        carry["z_H"] = new_z_H.detach()
        carry["z_L"] = new_z_L.detach()
        carry["steps"] = carry["steps"] + 1

        with torch.no_grad():
            is_last_step = carry["steps"] >= self.max_reasoning_steps
            halted = is_last_step | (q_halt > 0)

            force_continue = (
                torch.rand(B, device=self.device) < self.halt_exploration_prob
            )
            min_steps = torch.randint(
                low=2,
                high=self.max_reasoning_steps + 1,
                size=(B,),
                device=self.device,
            )
            exploration_halt = carry["steps"] >= min_steps
            halted = halted & (~force_continue | exploration_halt)

            carry["halted"] = halted

        return {
            "lm": lm_loss.detach() / (B * 81),
            "q_halt": q_halt_loss.detach() / B,
        }

    def _step_act_wta(self, inputs: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """K-head WTA ACT step. Batches K heads into single [B*K] forward pass.

        Note: BatchNorm would break with this batching (stats computed across heads).
        Current model uses RMSNorm which is per-sample, so this is safe.
        """
        B = self.batch_size
        seq_len = self.model.config.seq_len

        if self.act_carry is None:
            self.act_carry = self._init_act_carry()

        carry = self.act_carry
        halted = carry["halted"]

        if halted.any():
            halted_indices = halted.nonzero(as_tuple=True)[0]

            for idx in halted_indices:
                steps_taken = int(carry["steps"][idx].item())
                if steps_taken > 0:
                    self.act_steps_history.append(steps_taken)
                    if 1 <= steps_taken <= 16:
                        self.halt_steps_histogram[steps_taken - 1] += 1

            n_reset = min(len(halted_indices), len(inputs))
            for i in range(n_reset):
                idx = halted_indices[i]
                carry["z_H"][idx] = self.model.H_init.squeeze(0)
                noise = (
                    torch.randn(
                        seq_len,
                        self.model.config.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * self.z_L_noise
                )
                carry["z_L"][idx] = self.model.L_init.squeeze(0) + noise
                carry["steps"][idx] = 0
                carry["inputs"][idx] = inputs[i]
                carry["labels"][idx] = labels[i]
                carry["halted"][idx] = False

        inputs_with_halt = self._prepend_halt_token(carry["inputs"])
        labels_flat = carry["labels"].reshape(B, -1)
        z_H = carry["z_H"]

        # Batch all K heads into single forward pass: [B*K, ...]
        # Build z_L for all heads with same noise order as sequential loop
        z_L_all = []
        for k in range(self.K):
            L_init_k = (
                self.model.L_init if k == 0 else getattr(self.model, f"L_init_{k}")
            )
            noise = (
                torch.randn(
                    B,
                    seq_len,
                    self.model.config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                * self.z_L_noise
            )
            z_L_all.append(L_init_k.expand(B, seq_len, -1) + noise)

        # Stack and reshape: [K, B, ...] -> [B*K, ...]
        z_L_batched = torch.stack(z_L_all, dim=1).reshape(B * self.K, seq_len, -1)
        z_H_batched = (
            z_H.unsqueeze(1)
            .expand(B, self.K, seq_len, -1)
            .reshape(B * self.K, seq_len, -1)
        )
        inputs_batched = (
            inputs_with_halt.unsqueeze(1).expand(B, self.K, -1).reshape(B * self.K, -1)
        )

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            out = self.model(inputs_batched, z_H=z_H_batched, z_L=z_L_batched)

        # Reshape outputs back: [B*K, ...] -> [B, K, ...]
        logits_all = out["logits"][:, 1:].reshape(B, self.K, 81, -1)
        q_halt_all = out["q_halt"].reshape(B, self.K)
        z_H_out = out["z_H"].reshape(B, self.K, seq_len, -1)
        z_L_out = out["z_L"].reshape(B, self.K, seq_len, -1)

        # Convert to lists for compatibility with rest of function
        all_logits = [logits_all[:, k] for k in range(self.K)]
        all_q_halt = [q_halt_all[:, k] for k in range(self.K)]
        all_z_H = [z_H_out[:, k] for k in range(self.K)]
        all_z_L = [z_L_out[:, k] for k in range(self.K)]

        # Per-puzzle loss for each head
        losses = []
        for k in range(self.K):
            loss_k = (
                nn.functional.cross_entropy(
                    all_logits[k].reshape(-1, all_logits[k].shape[-1]),
                    labels_flat.reshape(-1).long(),
                    label_smoothing=self.label_smoothing,
                    reduction="none",
                )
                .reshape(B, 81)
                .mean(dim=-1)
            )
            losses.append(loss_k)

        # WTA: backprop through winner only
        stacked_losses = torch.stack(losses, dim=1)  # [B, K]
        winner_idx = stacked_losses.argmin(dim=1)  # [B]
        winner_loss = stacked_losses[torch.arange(B, device=self.device), winner_idx]
        lm_loss = winner_loss.mean()

        # q_halt loss on winner
        winner_q_halt = torch.stack(all_q_halt, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]
        winner_logits = torch.stack(all_logits, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]

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
            q_halt_loss = torch.tensor(0.0, device=self.device)

        if self.current_step < self.q_halt_warmup_steps:
            q_halt_loss = torch.tensor(0.0, device=self.device)

        total_loss = lm_loss + self.q_halt_weight * q_halt_loss
        total_loss.backward()
        self._update_weights()

        # Update carry state using winner's z_H/z_L
        new_z_H = torch.stack(all_z_H, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]
        new_z_L = torch.stack(all_z_L, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]

        carry["z_H"] = new_z_H.detach()
        carry["z_L"] = new_z_L.detach()
        carry["steps"] = carry["steps"] + 1

        with torch.no_grad():
            is_last_step = carry["steps"] >= self.max_reasoning_steps
            halted = is_last_step | (winner_q_halt > 0)

            force_continue = (
                torch.rand(B, device=self.device) < self.halt_exploration_prob
            )
            min_steps = torch.randint(
                low=2,
                high=self.max_reasoning_steps + 1,
                size=(B,),
                device=self.device,
            )
            exploration_halt = carry["steps"] >= min_steps
            halted = halted & (~force_continue | exploration_halt)

            carry["halted"] = halted

        head0_wins = (winner_idx == 0).float().mean()

        return {
            "lm": lm_loss.detach(),
            "q_halt": q_halt_loss.detach(),
            "head0_wins": head0_wins.detach(),
        }

    def _jsd_loss(self, logits_list: list[Tensor]) -> Tensor:
        """JSD = H(avg) - avg(H). Maximizing this encourages head diversity."""
        probs = [nn.functional.softmax(logits, dim=-1) for logits in logits_list]
        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        H_avg = -(avg_prob * (avg_prob + 1e-8).log()).sum(dim=-1).mean()
        avg_H = torch.stack(
            [-(p * (p + 1e-8).log()).sum(dim=-1).mean() for p in probs]
        ).mean()
        return H_avg - avg_H

    def _step_act_wta_jsd(self, inputs: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """K-head WTA + JSD ACT step. Batches K heads into single [B*K] forward pass.

        Note: BatchNorm would break with this batching (stats computed across heads).
        Current model uses RMSNorm which is per-sample, so this is safe.
        """
        B = self.batch_size
        seq_len = self.model.config.seq_len

        if self.act_carry is None:
            self.act_carry = self._init_act_carry()

        carry = self.act_carry
        halted = carry["halted"]

        if halted.any():
            halted_indices = halted.nonzero(as_tuple=True)[0]

            for idx in halted_indices:
                steps_taken = int(carry["steps"][idx].item())
                if steps_taken > 0:
                    self.act_steps_history.append(steps_taken)
                    if 1 <= steps_taken <= 16:
                        self.halt_steps_histogram[steps_taken - 1] += 1

            n_reset = min(len(halted_indices), len(inputs))
            for i in range(n_reset):
                idx = halted_indices[i]
                carry["z_H"][idx] = self.model.H_init.squeeze(0)
                noise = (
                    torch.randn(
                        seq_len,
                        self.model.config.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * self.z_L_noise
                )
                carry["z_L"][idx] = self.model.L_init.squeeze(0) + noise
                carry["steps"][idx] = 0
                carry["inputs"][idx] = inputs[i]
                carry["labels"][idx] = labels[i]
                carry["halted"][idx] = False

        inputs_with_halt = self._prepend_halt_token(carry["inputs"])
        z_H = carry["z_H"]

        # Batch all K heads into single forward pass: [B*K, ...]
        z_L_all = []
        for k in range(self.K):
            L_init_k = (
                self.model.L_init if k == 0 else getattr(self.model, f"L_init_{k}")
            )
            noise = (
                torch.randn(
                    B,
                    seq_len,
                    self.model.config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                * self.z_L_noise
            )
            z_L_all.append(L_init_k.expand(B, seq_len, -1) + noise)

        z_L_batched = torch.stack(z_L_all, dim=1).reshape(B * self.K, seq_len, -1)
        z_H_batched = (
            z_H.unsqueeze(1)
            .expand(B, self.K, seq_len, -1)
            .reshape(B * self.K, seq_len, -1)
        )
        inputs_batched = (
            inputs_with_halt.unsqueeze(1).expand(B, self.K, -1).reshape(B * self.K, -1)
        )

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            out = self.model(inputs_batched, z_H=z_H_batched, z_L=z_L_batched)

        logits_all = out["logits"][:, 1:].reshape(B, self.K, 81, -1)
        q_halt_all = out["q_halt"].reshape(B, self.K)
        z_H_out = out["z_H"].reshape(B, self.K, seq_len, -1)
        z_L_out = out["z_L"].reshape(B, self.K, seq_len, -1)

        all_logits = [logits_all[:, k] for k in range(self.K)]
        all_q_halt = [q_halt_all[:, k] for k in range(self.K)]
        all_z_H = [z_H_out[:, k] for k in range(self.K)]
        all_z_L = [z_L_out[:, k] for k in range(self.K)]

        labels_flat = carry["labels"].reshape(B, -1)
        losses = []
        for k in range(self.K):
            loss_k = (
                nn.functional.cross_entropy(
                    all_logits[k].reshape(-1, all_logits[k].shape[-1]),
                    labels_flat.reshape(-1).long(),
                    label_smoothing=self.label_smoothing,
                    reduction="none",
                )
                .reshape(B, 81)
                .mean(dim=-1)
            )
            losses.append(loss_k)

        stacked_losses = torch.stack(losses, dim=1)
        winner_idx = stacked_losses.argmin(dim=1)
        winner_loss = stacked_losses[torch.arange(B, device=self.device), winner_idx]
        lm_loss = winner_loss.mean()

        # JSD loss
        jsd = self._jsd_loss(all_logits)
        if self.current_step < self.jsd_warmup_steps:
            t = self.current_step / self.jsd_warmup_steps
            jsd_weight = self.jsd_weight_init * (1 - t) + self.jsd_weight_final * t
        else:
            jsd_weight = self.jsd_weight_final
        jsd_loss = -jsd * jsd_weight

        winner_q_halt = torch.stack(all_q_halt, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]
        winner_logits = torch.stack(all_logits, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]

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
            q_halt_loss = torch.tensor(0.0, device=self.device)

        if self.current_step < self.q_halt_warmup_steps:
            q_halt_loss = torch.tensor(0.0, device=self.device)

        total_loss = lm_loss + jsd_loss + self.q_halt_weight * q_halt_loss
        total_loss.backward()
        self._update_weights()

        new_z_H = torch.stack(all_z_H, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]
        new_z_L = torch.stack(all_z_L, dim=1)[
            torch.arange(B, device=self.device), winner_idx
        ]

        carry["z_H"] = new_z_H.detach()
        carry["z_L"] = new_z_L.detach()
        carry["steps"] = carry["steps"] + 1

        with torch.no_grad():
            is_last_step = carry["steps"] >= self.max_reasoning_steps
            halted = is_last_step | (winner_q_halt > 0)

            force_continue = (
                torch.rand(B, device=self.device) < self.halt_exploration_prob
            )
            min_steps = torch.randint(
                low=2,
                high=self.max_reasoning_steps + 1,
                size=(B,),
                device=self.device,
            )
            exploration_halt = carry["steps"] >= min_steps
            halted = halted & (~force_continue | exploration_halt)

            carry["halted"] = halted

        head0_wins = (winner_idx == 0).float().mean()

        return {
            "lm": lm_loss.detach(),
            "jsd": jsd.detach(),
            "jsd_weight": torch.tensor(jsd_weight),
            "q_halt": q_halt_loss.detach(),
            "head0_wins": head0_wins.detach(),
        }

    def _update_weights(self) -> None:
        """Common weight update logic with gradient accumulation."""
        assert self.optimizer1 is not None
        assert self.optimizer2 is not None
        self._grad_accum_counter += 1

        # Only step optimizer every grad_accum_steps
        if self._grad_accum_counter < self.grad_accum_steps:
            return

        self._grad_accum_counter = 0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

    def evaluate(self, loader: Any) -> tuple[float, float]:
        ema = self.ema  # Local var for type narrowing
        use_ema = ema is not None and self.current_step >= self.ema_warmup_steps
        if use_ema:
            assert ema is not None  # Type guard
            ema.apply(self.model)

        try:
            self.model.eval()
            cell_acc, puzzle_acc = self.evaluate_act(loader)

            ckpt_start = time.perf_counter()
            script_path = pathlib.Path(sys.argv[0]).resolve()
            exp_name = script_path.stem
            ckpt_dir = script_path.parent / "ckpts" / exp_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if (
                hasattr(self, "checkpoint_steps")
                and self.current_step in self.checkpoint_steps
            ):
                path = ckpt_dir / f"step{self.current_step:05d}.pt"
                torch.save(self.make_checkpoint(), path)
                ckpt_elapsed = time.perf_counter() - ckpt_start
                print(f"  saved: {path} acc={cell_acc:.2f}% ({ckpt_elapsed:5.3f}s)")

            if cell_acc > self.best_acc:
                self.best_acc = cell_acc
                path = ckpt_dir / "best.pt"
                torch.save(self.make_checkpoint(), path)
                ckpt_elapsed = time.perf_counter() - ckpt_start
                print(f"  saved: {path} acc={cell_acc:.2f}% ({ckpt_elapsed:5.3f}s)")

            return cell_acc, puzzle_acc
        finally:
            self.model.train()
            if use_ema:
                assert ema is not None  # Type guard
                ema.restore(self.model)

    def evaluate_act(self, loader: Any) -> tuple[float, float]:
        if self.eval_method == "wta":
            return self.evaluate_act_haltfast_wta(loader)
        if self.eval_method == "fast":
            return self.evaluate_act_haltfast(loader)
        return self.evaluate_act_full(loader)

    def evaluate_act_haltfast(self, loader: Any) -> tuple[float, float]:
        """Fast eval: streaming replacement when q_halt > 0, fixed batch size.

        Mirrors qhalt_eval.py: uses model() not step(), halted samples replaced with new puzzles.
        """
        seq_len = self.model.config.seq_len
        bs = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.batch_size
        )
        B = bs

        # Results accumulators
        total_puzzles = 0
        total_correct = 0
        total_cell_correct = 0
        total_cells = 0
        total_steps = 0
        halt_histogram = [0] * self.max_reasoning_steps

        # Batch state
        inputs = torch.zeros(B, 81, device=self.device, dtype=torch.int32)
        labels = torch.zeros(B, 81, device=self.device, dtype=torch.int32)
        z_H = self.model.H_init.expand(B, seq_len, -1).clone()
        L_init = getattr(self.model, "L_init_0", self.model.L_init)
        z_L = L_init.expand(B, seq_len, -1).clone()
        steps = torch.zeros(B, device=self.device, dtype=torch.int32)
        valid = torch.zeros(B, device=self.device, dtype=torch.bool)

        # Cache init states for resets
        H_init_single = self.model.H_init.squeeze(0)
        L_init_single = L_init.squeeze(0)

        # Puzzle iterator
        puzzle_iter = iter(loader)
        pending_inputs = None
        pending_labels = None
        pending_idx = 0
        pending_valid = 0
        samples_seen = 0

        def get_next_puzzle():
            """Get next puzzle from iterator. Returns (input, label) or (None, None) if exhausted."""
            nonlocal \
                pending_inputs, \
                pending_labels, \
                pending_idx, \
                pending_valid, \
                samples_seen

            # Respect max_eval_samples limit
            if self.max_eval_samples >= 0 and samples_seen >= self.max_eval_samples:
                return None, None

            if pending_inputs is not None and pending_idx < pending_valid:
                assert pending_labels is not None
                inp = pending_inputs[pending_idx]
                lab = pending_labels[pending_idx]
                pending_idx += 1
                samples_seen += 1
                return inp, lab

            # Need new batch
            try:
                batch = next(puzzle_iter)
                pending_inputs = batch[0]
                pending_labels = batch[1]
                # batch[3] is valid_count: number of non-padded samples (last batch
                # may be zero-padded to batch_size).
                pending_valid = batch[3]
                pending_idx = 0
                if pending_valid > 0:
                    inp = pending_inputs[pending_idx]
                    lab = pending_labels[pending_idx]
                    pending_idx += 1
                    samples_seen += 1
                    return inp, lab
            except StopIteration:
                pass

            return None, None

        # Initialize batch
        for i in range(B):
            inp, lab = get_next_puzzle()
            if inp is not None:
                assert lab is not None
                inputs[i] = inp
                labels[i] = lab
                valid[i] = True

        if not valid.any():
            print("  cell_acc=0.00%, puzzle_acc=0.00%")
            return 0.0, 0.0

        halt_token = torch.full(
            (B, 1), HALT_TOKEN_ID, device=self.device, dtype=torch.int32
        )

        with torch.no_grad():
            while valid.any():
                # Forward pass
                inputs_with_halt = torch.cat([halt_token, inputs], dim=1)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self.model(inputs_with_halt, z_H, z_L)

                z_H = out["z_H"]
                z_L = out["z_L"]
                logits = out["logits"][:, 1:]  # [B, 81, vocab]
                q_halt = out["q_halt"]  # [B] - raw logit, halt when > 0

                preds = logits.argmax(dim=-1)  # [B, 81]
                steps += valid.int()

                # Determine which samples halt (q_halt > 0 means sigmoid(q_halt) > 0.5)
                halted = valid & ((q_halt > 0) | (steps >= self.max_reasoning_steps))

                if halted.any():
                    # Vectorized metrics for halted samples
                    correct_mask = preds == labels  # [B, 81]
                    cell_correct_per_sample = correct_mask.sum(dim=1)  # [B]
                    puzzle_correct_per_sample = cell_correct_per_sample == 81  # [B]

                    # Accumulate only for halted samples
                    n_halted = halted.sum().item()
                    total_puzzles += n_halted
                    total_correct += puzzle_correct_per_sample[halted].sum().item()
                    total_cell_correct += cell_correct_per_sample[halted].sum().item()
                    total_cells += 81 * n_halted
                    total_steps += steps[halted].sum().item()

                    # Update histogram
                    halted_steps = steps[halted].cpu().tolist()
                    for s in halted_steps:
                        if 1 <= s <= self.max_reasoning_steps:
                            halt_histogram[s - 1] += 1

                    # Replace halted samples with new puzzles (must be sequential for iterator)
                    halted_idx = halted.nonzero(as_tuple=True)[0]
                    for idx_tensor in halted_idx:
                        idx = idx_tensor.item()
                        inp, lab = get_next_puzzle()
                        if inp is not None:
                            assert lab is not None
                            i = int(idx)
                            inputs[i] = inp
                            labels[i] = lab
                            z_H[i] = H_init_single
                            z_L[i] = L_init_single
                            steps[i] = 0
                            valid[i] = True
                        else:
                            valid[int(idx)] = False

        cell_acc = 100 * total_cell_correct / total_cells if total_cells > 0 else 0.0
        puzzle_acc = 100 * total_correct / total_puzzles if total_puzzles > 0 else 0.0
        print(f"  cell_acc={cell_acc:.2f}%, puzzle_acc={puzzle_acc:.2f}%")

        if total_puzzles > 0:
            avg_halt = total_steps / total_puzzles
            print(f"  eval_avg_halt_step: {avg_halt:.2f}")

        return cell_acc, puzzle_acc

    def evaluate_act_haltfast_wta(
        self,
        loader: Any,
        progress: bool = False,
    ) -> tuple[float, float]:
        """WTA eval: run K heads, first to halt wins. Streaming replacement."""
        start_time = time.perf_counter()
        seq_len = self.model.config.seq_len
        hidden = self.model.config.hidden_size
        bs = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.batch_size
        )
        B = bs

        total_correct = 0
        total_cells_correct = 0
        total_cells = 0
        total_puzzles = 0

        # Batch state: [B, 81] for inputs/labels, [B, K, seq_len, hidden] for z
        inputs = torch.zeros(B, 81, device=self.device, dtype=torch.int32)
        labels = torch.zeros(B, 81, device=self.device, dtype=torch.int32)
        z_H = torch.zeros(
            B, self.K, seq_len, hidden, device=self.device, dtype=self.dtype
        )
        z_L = torch.zeros(
            B, self.K, seq_len, hidden, device=self.device, dtype=self.dtype
        )
        steps = torch.zeros(B, device=self.device, dtype=torch.int32)
        valid = torch.zeros(B, device=self.device, dtype=torch.bool)
        has_winner = torch.zeros(B, device=self.device, dtype=torch.bool)
        winner_logits = torch.zeros(
            B, 81, self.model.config.vocab_size, device=self.device, dtype=self.dtype
        )

        # Cache init states
        H_init_single = self.model.H_init.squeeze(0)
        L_inits = []
        for k in range(self.K):
            if hasattr(self.model, f"L_init_{k}"):
                L_inits.append(getattr(self.model, f"L_init_{k}").squeeze(0))
            else:
                L_inits.append(self.model.L_init.squeeze(0))

        # Puzzle iterator state
        puz_state = _PuzzleIterState(iter(loader), self.max_eval_samples)

        # Initialize batch
        for i in range(B):
            puzzle_idx = puz_state.samples_seen
            inp, lab = _get_next_puzzle(puz_state)
            if inp is not None:
                assert lab is not None
                _reset_eval_slot(
                    self,
                    i,
                    inp,
                    lab,
                    puzzle_idx,
                    inputs,
                    labels,
                    z_H,
                    z_L,
                    steps,
                    has_winner,
                    valid,
                    H_init_single,
                    L_inits,
                    seq_len,
                )

        if not valid.any():
            print("  cell_acc=0.00%, puzzle_acc=0.00%")
            return 0.0, 0.0

        halt_token = torch.full(
            (B, 1), HALT_TOKEN_ID, device=self.device, dtype=torch.int32
        )

        with torch.no_grad():
            while valid.any():
                inputs_with_halt = torch.cat([halt_token, inputs], dim=1)

                # Batch all K heads into single forward pass: [B*K, ...]
                z_H_batched = z_H.reshape(B * self.K, seq_len, -1)
                z_L_batched = z_L.reshape(B * self.K, seq_len, -1)
                inputs_batched = (
                    inputs_with_halt.unsqueeze(1)
                    .expand(B, self.K, -1)
                    .reshape(B * self.K, -1)
                )

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self.model(inputs_batched, z_H_batched, z_L_batched)

                # Reshape outputs back: [B*K, ...] -> [B, K, ...]
                z_H = out["z_H"].reshape(B, self.K, seq_len, -1)
                z_L = out["z_L"].reshape(B, self.K, seq_len, -1)
                q_halt_stack = out["q_halt"].reshape(B, self.K)  # [B, K]
                logits_stack = out["logits"][:, 1:].reshape(
                    B, self.K, 81, -1
                )  # [B, K, 81, vocab]

                steps += valid.int()

                # For puzzles without winner: check if any head halted
                can_win = valid & ~has_winner
                halted_mask = q_halt_stack > 0  # [B, K]
                any_halted = can_win & halted_mask.any(dim=1)  # [B]

                if any_halted.any():
                    # Pick head with highest q_halt among those that halted
                    q_for_selection = q_halt_stack.clone()
                    q_for_selection[~halted_mask] = float("-inf")
                    winner_head = q_for_selection.argmax(dim=1)  # [B]

                    # Gather winner logits
                    batch_idx = torch.arange(B, device=self.device)
                    selected_logits = logits_stack[
                        batch_idx, winner_head
                    ]  # [B, 81, vocab]

                    winner_logits[any_halted] = selected_logits[any_halted]
                    has_winner[any_halted] = True

                # Check for puzzles that are done (have winner or hit max steps)
                done = valid & (has_winner | (steps >= self.max_reasoning_steps))

                if done.any():
                    # Fallback: no winner by max steps -> use head 0's logits
                    fallback = done & ~has_winner
                    if fallback.any():
                        winner_logits[fallback] = logits_stack[fallback, 0]

                    # Score done puzzles
                    done_idx = done.nonzero(as_tuple=True)[0]
                    for idx_tensor in done_idx:
                        idx = int(idx_tensor.item())
                        preds = winner_logits[idx].argmax(dim=-1)
                        correct = (preds == labels[idx]).sum().item()
                        total_cells_correct += correct
                        total_cells += 81
                        total_correct += 1 if correct == 81 else 0
                        total_puzzles += 1

                        if progress and total_puzzles % 1000 == 0:
                            elapsed = time.perf_counter() - start_time
                            rate = total_puzzles / elapsed
                            print(
                                f"  {total_puzzles} puzzles, {elapsed:.1f}s, {rate:.1f} puz/s",
                                flush=True,
                            )

                        # Replace with new puzzle
                        puzzle_idx = puz_state.samples_seen
                        inp, lab = _get_next_puzzle(puz_state)
                        if inp is not None:
                            assert lab is not None
                            _reset_eval_slot(
                                self,
                                idx,
                                inp,
                                lab,
                                puzzle_idx,
                                inputs,
                                labels,
                                z_H,
                                z_L,
                                steps,
                                has_winner,
                                valid,
                                H_init_single,
                                L_inits,
                                seq_len,
                            )
                        else:
                            valid[idx] = False

        cell_acc = 100 * total_cells_correct / total_cells if total_cells > 0 else 0.0
        puzzle_acc = 100 * total_correct / total_puzzles if total_puzzles > 0 else 0.0
        print(f"  cell_acc={cell_acc:.2f}%, puzzle_acc={puzzle_acc:.2f}%")
        return cell_acc, puzzle_acc

    def evaluate_act_full(self, loader: Any) -> tuple[float, float]:
        """Full eval: run all 16 steps with diagnostics."""
        correct = torch.tensor(0, device=self.device)
        puzzles_correct = torch.tensor(0, device=self.device)
        total = 0
        total_halt_steps = torch.tensor(0.0, device=self.device)
        total_samples = 0
        last_batch = None

        # Reset-retry stats (#22)
        total_stuck = 0
        total_helped = 0

        samples_seen = 0
        with torch.no_grad():
            for batch in loader:
                if self.max_eval_samples >= 0 and samples_seen >= self.max_eval_samples:
                    break
                last_batch = batch

                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                # batch[3] is valid_count: number of non-padded samples (last batch
                # may be zero-padded to batch_size).
                valid_count = batch[3]
                B = inputs.shape[0]

                z_H, z_L = self._init_z(B)
                inputs_with_halt = self._prepend_halt_token(inputs)

                # Track first halt and its logits per puzzle
                has_halted = torch.zeros(B, device=self.device, dtype=torch.bool)
                halt_logits = torch.zeros(
                    B,
                    self.config.seq_len - 1,
                    self.model.config.vocab_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                halt_step = torch.full(
                    (B,),
                    self.max_reasoning_steps,
                    device=self.device,
                    dtype=torch.int32,
                )

                out: dict[str, Tensor] = {}
                for step in range(self.max_reasoning_steps):
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                        out = self.model(inputs_with_halt, z_H, z_L)
                    z_H = out["z_H"]
                    z_L = out["z_L"]
                    q_halt = out["q_halt"]
                    logits = out["logits"][:, 1:]

                    # Record logits for puzzles that halt this step (first halt wins)
                    newly_halted = ~has_halted & (q_halt > 0)
                    if newly_halted.any():
                        halt_logits[newly_halted] = logits[newly_halted]
                        halt_step[newly_halted] = step + 1
                        has_halted[newly_halted] = True

                # Fallback: puzzles that never halted use final logits
                if not has_halted.all():
                    halt_logits[~has_halted] = out["logits"][~has_halted, 1:]

                total_halt_steps += halt_step[:valid_count].float().sum()
                total_samples += valid_count
                samples_seen += valid_count

                # If we were to use:
                #   preds = halt_logits.argmax(dim=-1)
                # then we would be basing the prediction on the first "halt" signal.
                # However, if we use the following then we base the prediction on
                # on the final last step.
                preds = out["logits"][:, 1:].argmax(
                    dim=-1
                )  # Use final logits, not halt

                # Reset-retry for stuck puzzles (#22)
                if self.enable_reset_retry:
                    stuck_mask = ~has_halted  # Puzzles that never halted

                    if stuck_mask.any():
                        n_stuck = int(stuck_mask.sum().item())
                        total_stuck += n_stuck

                        # Track which were wrong before retry
                        stuck_idx = stuck_mask.nonzero(as_tuple=True)[0]
                        orig_preds_stuck = preds[stuck_idx]
                        labels_stuck = labels[stuck_idx]
                        was_wrong = ~(orig_preds_stuck == labels_stuck).all(dim=-1)

                        # Reset z_H and z_L for stuck puzzles
                        z_H_retry, z_L_retry = self._init_z(n_stuck)
                        inputs_stuck = inputs_with_halt[stuck_idx]

                        # Re-run H-cycles for stuck puzzles
                        out_retry: dict[str, Tensor] = {}
                        for _ in range(self.max_reasoning_steps):
                            with torch.autocast(
                                device_type=self.device.type, dtype=self.dtype
                            ):
                                out_retry = self.model(
                                    inputs_stuck, z_H_retry, z_L_retry
                                )
                            z_H_retry = out_retry["z_H"]
                            z_L_retry = out_retry["z_L"]

                        # Get retry predictions
                        retry_preds = out_retry["logits"][:, 1:].argmax(dim=-1)

                        # Count how many were helped (wrong->correct)
                        now_correct = (retry_preds == labels_stuck).all(dim=-1)
                        helped = (was_wrong & now_correct).sum().item()
                        total_helped += helped

                        # Replace predictions for stuck puzzles
                        preds[stuck_idx] = retry_preds

                # Only count valid samples (exclude padding)
                preds_valid = preds[:valid_count]
                labels_valid = labels[:valid_count]
                correct += (preds_valid == labels_valid).sum()
                puzzles_correct += (preds_valid == labels_valid).all(dim=-1).sum()
                total += labels_valid.numel()

        cell_acc = 100 * correct.item() / total
        puzzle_acc = 100 * puzzles_correct.item() / total_samples
        print(f"  cell_acc={cell_acc:.2f}%, puzzle_acc={puzzle_acc:.2f}%")

        if total_samples > 0:
            avg_halt = total_halt_steps.item() / total_samples
            print(f"  eval_avg_halt_step: {avg_halt:.2f}")

        # Print reset-retry stats (#22)
        if self.enable_reset_retry and total_stuck > 0:
            retry_rate = 100 * total_stuck / total_samples
            help_rate = 100 * total_helped / total_stuck
            print(
                f"  reset_retry: n_stuck={total_stuck} ({retry_rate:.1f}%), "
                f"n_helped={total_helped} ({help_rate:.1f}% of stuck)"
            )

        if last_batch is not None:
            with torch.no_grad():
                # Use full batch to avoid torch.compile recompilation
                sample_inputs = last_batch[0].to(self.device)
                sample_labels = last_batch[1].to(self.device)
                # batch[3] is valid_count: number of non-padded samples.
                diag_valid = last_batch[3]
                inputs_with_halt = self._prepend_halt_token(sample_inputs)
                diag_B = sample_inputs.shape[0]

                # Get per-H logits from single model call
                z_H, z_L = self._init_z(diag_B)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self.model(inputs_with_halt, z_H, z_L)
                all_logits = [lg[:, 1:] for lg in out["all_logits"]]
                all_z_H = list(out["all_z_H"])
                print_diagnostics(
                    all_logits,
                    sample_labels,
                    {"z_H": all_z_H},
                    valid_count=diag_valid,
                )

                # Per-ACT-step diagnostics (accumulate all, compute post-hoc)
                z_H, z_L = self._init_z(diag_B)
                step_logits = []
                step_q_halts = []
                for _ in range(self.max_reasoning_steps):
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                        out = self.model.step(inputs_with_halt, z_H, z_L)
                    step_logits.append(out["logits"][:, 1:])
                    step_q_halts.append(out["q_halt"])
                    z_H = out["z_H"]
                    z_L = out["z_L"]

                # Use only valid samples for diagnostics (exclude padding)
                cell_accs = []
                puzzle_accs = []
                for lg in step_logits:
                    preds = lg[:diag_valid].argmax(dim=-1)
                    labels_v = sample_labels[:diag_valid]
                    cell_accs.append((preds == labels_v).float().mean().item() * 100)
                    puzzle_accs.append(
                        (preds == labels_v).all(dim=-1).float().mean().item() * 100
                    )
                q_stack = torch.stack(step_q_halts)[:, :diag_valid]
                mean_sigmoids = torch.sigmoid(q_stack).mean(dim=1).tolist()
                halt_rates = (q_stack > 0).float().mean(dim=1).tolist()

                print(
                    f"  per_ACT_cell_acc: {[f'H{i + 1}={a:.1f}%' for i, a in enumerate(cell_accs)]}"
                )
                print(
                    f"  per_ACT_puzzle_acc: {[f'H{i + 1}={a:.1f}%' for i, a in enumerate(puzzle_accs)]}"
                )
                print(f"  mean_sigmoid_q_halt: {[f'{s:.3f}' for s in mean_sigmoids]}")
                print(f"  halt_rates: {[f'{h:.2f}' for h in halt_rates]}")

                total_halts = sum(self.halt_steps_histogram)
                if total_halts > 0:
                    hist_pct = [
                        f"{100 * c / total_halts:.1f}"
                        for c in self.halt_steps_histogram
                    ]
                    print(f"  halt_histogram %: [{', '.join(hist_pct)}]")

                if self.act_steps_history:
                    recent = self.act_steps_history[-100:]
                    print(
                        f"  Avg ACT steps (last 100): {sum(recent) / len(recent):.2f}"
                    )

        return cell_acc, puzzle_acc


def train(experiment: ExperimentBase):
    """TRM training loop."""
    testloader = experiment.make_test_loader()
    trainloader = experiment.make_train_loader()
    experiment.model.train()

    max_train_sec = experiment.max_train_sec
    eval_every_steps = experiment.eval_every_steps

    loader = iter(trainloader)

    # Save step 0 checkpoint with RNG state before any training
    script_path = pathlib.Path(sys.argv[0]).resolve()
    exp_name = script_path.stem
    ckpt_dir = script_path.parent / "ckpts" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step0_path = ckpt_dir / "step00000_rng.pt"
    if not step0_path.exists():
        torch.save(experiment.make_checkpoint(), step0_path)
        print(f"Saved step 0 checkpoint with RNG: {step0_path}")

    train_start = epoch_start = time.perf_counter()
    epoch_losses: list[dict[str, Tensor] | Tensor] = []
    step_count = 0
    done = False

    def do_eval():
        nonlocal epoch_losses
        eval_start = time.perf_counter()

        cell_acc, puzzle_acc = experiment.evaluate(iter(testloader))

        eval_time = time.perf_counter() - eval_start
        epoch_elapsed = eval_start - epoch_start
        train_loss = ""
        if epoch_losses:
            first = epoch_losses[0]
            if isinstance(first, dict):
                keys = list(first.keys())
                parts = []
                dicts = [d for d in epoch_losses if isinstance(d, dict)]
                for k in keys:
                    vals = torch.stack([d[k] for d in dicts])
                    parts.append(
                        f"{k}={vals.mean():.4f}±{vals.std():.4f}ϵ[{vals.min():.4f},{vals.max():.4f}]"
                    )
                train_loss = "  Train " + " ".join(parts)
            else:
                tensors = [t for t in epoch_losses if isinstance(t, Tensor)]
                losses = torch.stack(tensors)
                train_loss = f"  Train {losses.mean():.4f}±{losses.std():.4f} [{losses.min():.4f}, {losses.max():.4f}]"
            epoch_losses = []

        step = f"Step{step_count:6d}"
        acc_str = f"  Test {cell_acc:5.2f}% / {puzzle_acc:5.2f}%{train_loss}"
        timing = f"  ({eval_time:5.3f}s / {epoch_elapsed:6.3f}s)"

        print(f"{step}{acc_str}{timing}", flush=True)
        return eval_start

    while True:
        done = done or (time.perf_counter() - train_start >= max_train_sec)

        if done:
            break

        try:
            samples, targets, puzzle_ids, valid_count = next(loader)
            del valid_count  # TODO(josh): We need to be normalizing using this.
        except StopIteration:
            loader = iter(trainloader)
            continue

        loss = experiment.step(samples, targets, puzzle_ids)
        step_count += 1
        if loss is None:
            done = True
        else:
            epoch_losses.append({k: v.detach() for k, v in loss.items()})

        if eval_every_steps and step_count % eval_every_steps == 0:
            eval_start = do_eval()
            epoch_start = time.perf_counter()
            train_start += epoch_start - eval_start
            experiment.model.train()

    return experiment.model


def main(
    experiment: ExperimentBase | object,
):  # object: kludge for duck-typed experiments
    """TRM main entry point."""
    script_path = pathlib.Path(sys.argv[0])
    with (
        Tee(script_path.with_suffix(".log"))
        if script_path.name
        else contextlib.nullcontext()
    ):
        experiment.setup_optimizers()  # pyright: ignore[reportAttributeAccessIssue]
        return train(experiment)  # pyright: ignore[reportArgumentType]


def resume_from_checkpoint(
    exp: "ExperimentBase",
    ckpt_path: str,
    resume_optimizer: bool = True,
    resume_ema: bool = True,
) -> None:
    """Resume experiment from checkpoint.

    Args:
        exp: Experiment instance
        ckpt_path: Path to checkpoint file
        resume_optimizer: If True, load optimizer state. If False, fresh optimizers.
        resume_ema: If True, load EMA state. If False, copy model weights to EMA.

    """
    ckpt = torch.load(ckpt_path, map_location=exp.device, weights_only=False)

    # Handle both old format (just state_dict) and new format (dict with keys)
    if "model" in ckpt:
        exp.model.load_state_dict(ckpt["model"])
        exp.current_step = ckpt["step"]
        exp.best_acc = ckpt.get("best_acc", 0.0)

        if resume_optimizer and "optimizer1" in ckpt:
            assert exp.optimizer1 is not None
            assert exp.optimizer2 is not None
            exp.optimizer1.load_state_dict(ckpt["optimizer1"])
            exp.optimizer2.load_state_dict(ckpt["optimizer2"])
        else:
            exp.optimizer1, exp.optimizer2 = setup_muon_optimizers(exp.model)

        if exp.ema is not None:
            if resume_ema and "ema" in ckpt:
                for n, v in ckpt["ema"].items():
                    if n in exp.ema.shadow:
                        exp.ema.shadow[n].copy_(v)
            else:
                for n, p in exp.model.named_parameters():
                    if p.requires_grad and n in exp.ema.shadow:
                        exp.ema.shadow[n].copy_(p.data)
        # Restore RNG state if available
        if "rng_numpy" in ckpt:
            rng_state = ckpt["rng_numpy"]
            # Handle both old (tuple from np.random.get_state) and new (dict from Generator)
            if isinstance(rng_state, dict):
                numpy_rng.bit_generator.state = rng_state
            else:
                # Old tuple format from np.random.get_state() - restore to global RNG
                np.random.set_state(rng_state)  # noqa: NPY002
    else:
        # Old format: just model state_dict
        exp.model.load_state_dict(ckpt)
        # No step info, optimizer, or EMA in old format
        exp.optimizer1, exp.optimizer2 = setup_muon_optimizers(exp.model)
        if exp.ema is not None:
            for n, p in exp.model.named_parameters():
                if p.requires_grad and n in exp.ema.shadow:
                    exp.ema.shadow[n].copy_(p.data)

    exp.act_carry = None
    print(f"Resumed from {ckpt_path} at step {exp.current_step}")


def compute_diversity_loss(all_logits: list[Tensor]) -> Tensor:
    """Diversity loss: penalize high cosine similarity between consecutive H iterations."""
    if len(all_logits) < 2:
        return torch.tensor(0.0, device=all_logits[0].device)
    total_cos = torch.tensor(0.0, device=all_logits[0].device)
    for i in range(1, len(all_logits)):
        z_prev = all_logits[i - 1].reshape(all_logits[i - 1].shape[0], -1)
        z_curr = all_logits[i].reshape(all_logits[i].shape[0], -1)
        cos_sim = nn.functional.cosine_similarity(z_prev, z_curr, dim=1).mean()
        total_cos = total_cos + cos_sim
    return total_cos / (len(all_logits) - 1)


def setup_muon_optimizers(
    model: ModuleProtocol,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-4,
    muon_wd: float | None = None,
    adamw_wd: float | None = None,
    qk_wd: float | None = None,
) -> tuple[AdamW | DummyOptimizer, Muon | DummyOptimizer]:
    """Muon + AdamW optimizer setup for TRM.

    Args:
        qk_wd: If set, applies this weight decay to qk_proj params (for AttentionSplitQKV).

    """
    if muon_wd is None:
        muon_wd = 1e-4 / muon_lr
    if adamw_wd is None:
        adamw_wd = 1e-4 / adamw_lr

    muon_2d_params, muon_3d_params, muon_3d_qk_params, non_muon_params = {}, {}, {}, {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in n and "head" not in n:
            if p.ndim == 3:
                if qk_wd is not None and "qk_proj" in n:
                    muon_3d_qk_params[n] = p
                else:
                    muon_3d_params[n] = p
            else:
                muon_2d_params[n] = p
        else:
            non_muon_params[n] = p

    print(f"Adam params: {tuple(non_muon_params.keys())}")
    print(f"Muon 2D params: {tuple(muon_2d_params.keys())}")
    print(f"Muon 3D params (ensemble_dims=1): {tuple(muon_3d_params.keys())}")
    if muon_3d_qk_params:
        print(f"Muon 3D QK params (wd={qk_wd}): {tuple(muon_3d_qk_params.keys())}")

    if non_muon_params:
        opt1 = torch.optim.AdamW(
            non_muon_params.items(),
            lr=adamw_lr,
            betas=(0.9, 0.95),
            eps=1e-08,
            weight_decay=adamw_wd,
        )
    else:
        opt1 = DummyOptimizer()

    # Combine 2D and 3D Muon params into one optimizer with param groups
    muon_param_groups = []
    if muon_2d_params:
        muon_param_groups.append(
            {
                "params": list(muon_2d_params.values()),
                "ensemble_dims": 0,
            }
        )
    if muon_3d_params:
        muon_param_groups.append(
            {
                "params": list(muon_3d_params.values()),
                "ensemble_dims": 1,  # Per-head orthogonalization
            }
        )
    if muon_3d_qk_params:
        muon_param_groups.append(
            {
                "params": list(muon_3d_qk_params.values()),
                "ensemble_dims": 1,
                "weight_decay": qk_wd,
            }
        )

    if muon_param_groups:
        opt2 = Muon(
            muon_param_groups,
            lr=muon_lr,
            momentum=0.6,
            nesterov=True,
            ns_steps=3,
            weight_decay=muon_wd,
        )
    else:
        opt2 = DummyOptimizer()

    for opt in [opt1, opt2]:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return opt1, opt2
