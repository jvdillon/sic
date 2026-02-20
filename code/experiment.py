"""TRM experiment framework."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)

import contextlib
import dataclasses
import math
import pathlib
import sys
import time


if False:
    from util import enable_determinism

    enable_determinism()

from evaluation import print_diagnostics
from model import (
    EMA,
    TRM3,
    HCycleResult,
    ModuleProtocol,
    TRM3ConfigProtocol,
)
from optimizer import DummyOptimizer, Muon, lr_scale
from torch import Tensor, nn
from util import Tee, numpy_rng, set_seed

import numpy as np
import torch

from data import PuzzleDataset, augment_sudoku


if TYPE_CHECKING:
    from torch.optim import AdamW


def check_maze_paths(
    preds: Tensor,
    labels: Tensor,
    H: int,
    W: int,
) -> tuple[int, int]:
    """BFS check: does predicted solution connect start→goal?

    Returns (num_valid, num_optimal).
    Tokens: 1=wall, 2=empty, 3=start, 4=goal, 5=solution.
    """
    B = preds.shape[0]
    preds_np = preds.cpu().numpy().reshape(B, H, W)
    labels_np = labels.cpu().numpy().reshape(B, H, W)
    n_valid = 0
    n_optimal = 0
    for b in range(B):
        pred = preds_np[b]
        label = labels_np[b]
        starts = list(zip(*np.where(label == 3), strict=False))
        goals = list(zip(*np.where(label == 4), strict=False))
        if not starts or not goals:
            continue
        path = (pred == 5) | (pred == 3) | (pred == 4)
        visited = np.zeros((H, W), dtype=bool)
        q: deque[tuple[int, int]] = deque()
        for r, c in starts:
            if path[r, c]:
                visited[r, c] = True
                q.append((r, c))
        reached_goal = False
        while q:
            r, c = q.popleft()
            if label[r, c] == 4:
                reached_goal = True
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and path[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        if reached_goal:
            n_valid += 1
            if int((pred == 5).sum()) <= int((label == 5).sum()):
                n_optimal += 1
    return n_valid, n_optimal


@dataclass
class _PuzzleIterState:
    """Mutable state for streaming puzzle iteration."""

    puzzle_iter: Iterator[Any]
    max_samples: int
    pending_inputs: Tensor | None = None
    pending_labels: Tensor | None = None
    pending_puzzle_ids: Tensor | None = None
    pending_idx: int = 0
    pending_valid: int = 0
    samples_seen: int = 0


def _get_next_puzzle(
    state: _PuzzleIterState,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Get next puzzle from iterator, respecting max_samples limit.

    Returns (input, label, puzzle_id) or (None, None, None) if exhausted.
    """
    if state.max_samples >= 0 and state.samples_seen >= state.max_samples:
        return None, None, None

    if (
        state.pending_inputs is not None
        and state.pending_labels is not None
        and state.pending_idx < state.pending_valid
    ):
        inp = state.pending_inputs[state.pending_idx]
        lab = state.pending_labels[state.pending_idx]
        pid = (
            state.pending_puzzle_ids[state.pending_idx]
            if state.pending_puzzle_ids is not None
            else None
        )
        state.pending_idx += 1
        state.samples_seen += 1
        return inp, lab, pid
    try:
        batch = next(state.puzzle_iter)
        state.pending_inputs = batch[0]
        state.pending_labels = batch[1]
        state.pending_puzzle_ids = batch[2]
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
            pid = (
                state.pending_puzzle_ids[state.pending_idx]
                if state.pending_puzzle_ids is not None
                else None
            )
            state.pending_idx += 1
            state.samples_seen += 1
            return inp, lab, pid
    except StopIteration:
        pass
    return None, None, None


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


class EvalResult(NamedTuple):
    """Result from Experiment.evaluate()."""

    cell_acc: float
    puzzle_acc: float
    halt_cell_acc: float
    halt_puzzle_acc: float
    train_cell_acc: float
    train_puzzle_acc: float
    train_halt_cell_acc: float
    train_halt_puzzle_acc: float


class ForwardResult(TypedDict):
    losses: Tensor  # [P] per-chain losses (inf for inactive)
    logits: Tensor  # [P, num_puzzle_grid_tokens, V]
    q_halt: Tensor  # [P]
    z_H: Tensor  # [P, total_seq_len, C]
    z_L: Tensor  # [P, total_seq_len, C]
    z_H_pre: Tensor  # [P, total_seq_len, C] z_H input to last core()
    z_L_pre: Tensor  # [P, total_seq_len, C] z_L input to last core()


class Experiment:
    """Sparse chain pooling experiment (TRM3)."""

    seed: int = 42
    device: torch.device | None = None

    config: TRM3ConfigProtocol = TRM3.Config(
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
    num_instances: int | None = (
        None  # Limit training to first N instances (for ablations)
    )
    augmentation_random_bundle_max_size: int = (
        1  # Max augs of same instance kept contiguous in batch
    )
    use_additive_puzzle_emb: bool = False  # If True, embed puzzle_id and add to z_H
    # Size of puzzle_id embedding table (set to 256 for ARC)
    max_puzzle_ids_per_batch: int = 1

    # Eval
    eval_method: Literal["full", "fast", "wta"] = "full"
    eval_batch_size: int | None = 768  # Must match train pool size for compile
    eval_every_steps: int = 2_000
    max_eval_samples: int = 38_400
    max_eval_samples_train: int = 1_000
    checkpoint_steps: list[int] = list(range(0, 75_000, 500))  # noqa: RUF012
    enable_reset_retry: bool = False
    reset_threshold: float = 0.0
    eval_path_valid: bool = False  # Maze: BFS check predicted path connects start→goal
    eval_grid_shape: tuple[int, int] = (30, 30)  # Grid dims for path validity check

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

    # Learning rate schedule
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 0.0  # 1.0 = no decay, 0.0 = decay to zero

    # Gradient clipping
    grad_clip_max_norm: float | None = 1.0  # None = no clipping

    # If True (default), the PAD logit (class 0) participates in label smoothing
    # at non-PAD positions. If False, head outputs vocab_size-1 logits (no PAD class)
    # and loss uses ce(logits, labels-1, ignore_index=-1). Requires model config
    # to also set label_smoothing_includes_pad_token=False so the head is built accordingly.
    label_smoothing_includes_pad_token: bool = True

    # Loss normalization: if True, use sum + divide-by-batch_size (reference style)
    # instead of mean. Equivalent when all slots active, but matches reference pattern.
    loss_sum_normalize: bool = False

    cast_model_to_dtype: bool = True
    max_pending_samples: int = 1000
    log_lip: bool = False  # Log reasoning block Lipschitz estimate per step
    collect_h_cycle_intermediates: bool = False  # Collect per-H-cycle logits/z_H

    def __init__(self) -> None:
        self.setup_model()  # Sets self.device and self.dtype

        self.optimizer1: torch.optim.Optimizer | None = None
        self.optimizer2: torch.optim.Optimizer | None = None

        self.current_step = 0
        self.best_acc = 0.0
        self._grad_accum_counter = 0
        self._last_hc_result: HCycleResult | None = None

        self.act_steps_history: list[float] = []
        self.halt_steps_histogram = [0] * self.max_reasoning_steps

        assert self.device is not None
        self._state = TrainingState(
            num_chains=self.num_chains,
            num_puzzle_grid_tokens=self.config.num_puzzle_grid_tokens,
            total_seq_len=self.config.total_seq_len,
            hidden_size=self.config.hidden_size,
            K=self.K,
            device=self.device,
            dtype=self.dtype,
        )
        self._pending_inputs = torch.empty(
            0,
            self.config.num_puzzle_grid_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self._pending_labels = torch.empty(
            0,
            self.config.num_puzzle_grid_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self._pending_puzzle_ids = torch.empty(
            0,
            device=self.device,
            dtype=torch.long,
        )

    def setup_model(self) -> None:
        set_seed(self.seed)
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = self.config.dtype or torch.bfloat16
        self.model: TRM3 = cast(
            "TRM3",
            self.config.make().to(
                device=self.device,
                dtype=self.dtype if self.cast_model_to_dtype else None,
            ),
        )
        # Validate K matches model's num_effective_heads
        if self.config.num_effective_heads != self.K:
            raise ValueError(
                f"K ({self.K}) must match config.num_effective_heads "
                f"({self.config.num_effective_heads} = K_H * K_L)",
            )

        if (
            self.label_smoothing_includes_pad_token
            != self.config.label_smoothing_includes_pad_token
        ):
            raise ValueError(
                f"label_smoothing_includes_pad_token mismatch: "
                f"experiment={self.label_smoothing_includes_pad_token}, "
                f"model config={self.config.label_smoothing_includes_pad_token}",
            )

        # Validate that both puzzle embedding modes aren't enabled simultaneously
        if self.use_additive_puzzle_emb and self.config.num_puzzle_id_tokens > 0:
            raise ValueError(
                "Cannot enable both use_additive_puzzle_emb and num_puzzle_id_tokens. "
                "These are mutually exclusive puzzle conditioning approaches.",
            )

        self.ema = EMA(self.model, decay=self.ema_decay) if self.use_ema else None

        # Puzzle identifier embedding (for ARC multi-example conditioning)
        # NOTE: This is the OLD additive approach. Prefer TRM-style prefix tokens
        # (num_puzzle_id_tokens > 0) for new experiments.
        self.puzzle_id_embed: nn.Embedding | None = None
        if self.use_additive_puzzle_emb:
            self.puzzle_id_embed = nn.Embedding(
                self.max_puzzle_ids_per_batch,
                self.config.hidden_size,
            ).to(device=self.device, dtype=self.dtype)

    def setup_optimizers(self) -> None:
        """Must be called before step(). Called by main() before train()."""
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.02,
        )
        # Add experiment-level puzzle_id_embed to optimizer if enabled
        if self.puzzle_id_embed is not None and not isinstance(
            self.optimizer1, DummyOptimizer
        ):
            self.optimizer1.add_param_group(
                {"params": self.puzzle_id_embed.parameters()}
            )

    def make_train_loader(self) -> PuzzleDataset:
        assert self.device is not None
        return PuzzleDataset(
            data_dir=self.data_dir,
            device=self.device,
            batch_size=self.batch_size,
            train=True,
            augmentation_random_bundle_max_size=self.augmentation_random_bundle_max_size,
            num_instances=self.num_instances,
        )

    def make_test_loader(self) -> PuzzleDataset:
        assert self.device is not None
        bs = self.batch_size if self.eval_batch_size is None else self.eval_batch_size
        return PuzzleDataset(
            data_dir=self.data_dir,
            device=self.device,
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

    @property
    def current_max_steps(self) -> int:
        """Max H-steps allowed at the current training step."""
        valid_keys = [k for k in self.max_steps_schedule if k <= self.current_step]
        if not valid_keys:
            return min(v[0] for v in self.max_steps_schedule.values())
        return self.max_steps_schedule[max(valid_keys)][0]

    def step(
        self,
        inputs: Tensor,
        labels: Tensor,
        puzzle_ids: Tensor | None = None,
        valid_count: int | None = None,
    ) -> dict[str, Tensor] | None:
        if self.current_step >= self.total_train_steps:
            return None
        if valid_count is None:
            valid_count = inputs.shape[0]
        if self.augment_sudoku:
            inputs, labels = augment_sudoku(inputs, labels)

        valid_keys = [k for k in self.max_steps_schedule if k <= self.current_step]
        if not valid_keys:
            raise ValueError(
                f"max_steps_schedule has no entry for step {self.current_step}."
                " Keys specify start boundaries; include 0 as the first key."
                f" Got keys: {sorted(self.max_steps_schedule)}"
            )
        max_h_steps, train_q_halt = self.max_steps_schedule[max(valid_keys)]
        max_h_steps = min(max_h_steps, self.max_reasoning_steps)
        state = self._state

        self._enqueue(
            inputs[:valid_count],
            labels[:valid_count],
            puzzle_ids[:valid_count] if puzzle_ids is not None else None,
        )
        self._fill_pending()

        forward_result = self._forward(state)
        active_at_forward = state.batch_index >= 0
        active_samples, winner_chains = state.select_winners(forward_result["losses"])

        if len(active_samples) > 0:
            self._backward(
                forward_result, state, active_samples, winner_chains, train_q_halt
            )

        self._update_weights()

        if len(active_samples) > 0:
            chains = state.chain_indices[active_samples]
            valid = chains >= 0
            valid_chains = chains[valid]
            carry_H = self.config.carry_H
            carry_L = self.config.carry_L
            if carry_H == "copy_top1":
                winner_z_H = (
                    forward_result["z_H"][winner_chains]
                    .detach()
                    .unsqueeze(1)
                    .expand(-1, self.K, -1, -1)
                )
                state.z_H[valid_chains] = winner_z_H[valid]
            elif carry_H == "all":
                state.z_H[valid_chains] = forward_result["z_H"][valid_chains].detach()
            else:
                raise ValueError(f"Unsupported carry_H: {carry_H!r}")
            if carry_L == "copy_top1":
                winner_z_L = (
                    forward_result["z_L"][winner_chains]
                    .detach()
                    .unsqueeze(1)
                    .expand(-1, self.K, -1, -1)
                )
                state.z_L[valid_chains] = winner_z_L[valid]
            elif carry_L == "all":
                state.z_L[valid_chains] = forward_result["z_L"][valid_chains].detach()
            else:
                raise ValueError(f"Unsupported carry_L: {carry_L!r}")
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
                halt = at_max | (
                    (at_max | q_halt_positive)
                    & (~force_continue | (h_steps >= min_random_steps))
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
        result = {
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
        if self.log_lip:
            result["lip"] = torch.tensor(
                self._probe_lip(forward_result, winner_chains),
                device=state.device,
            )
        return result

    def reset_transient_state(self) -> None:
        print(f"  Resetting transient state at step {self.current_step}")
        self.setup_optimizers()
        if self.ema:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.ema.shadow:
                    self.ema.shadow[n].copy_(p.data)
        assert self.device is not None
        self._state = TrainingState(
            num_chains=self.num_chains,
            num_puzzle_grid_tokens=self.config.num_puzzle_grid_tokens,
            total_seq_len=self.config.total_seq_len,
            hidden_size=self.config.hidden_size,
            K=self.K,
            device=self.device,
            dtype=self.dtype,
        )
        self._pending_inputs = torch.empty(
            0,
            self.config.num_puzzle_grid_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self._pending_labels = torch.empty_like(self._pending_inputs)
        self._pending_puzzle_ids = torch.empty(
            0,
            device=self.device,
            dtype=torch.long,
        )

    # --- Private methods (training) ---

    def _enqueue(
        self,
        inputs: Tensor,
        labels: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> None:
        space = self.max_pending_samples - len(self._pending_inputs)
        if space <= 0:
            return
        self._pending_inputs = torch.cat([self._pending_inputs, inputs[:space]])
        self._pending_labels = torch.cat([self._pending_labels, labels[:space]])
        if puzzle_ids is not None:
            self._pending_puzzle_ids = torch.cat(
                [self._pending_puzzle_ids, puzzle_ids[:space]],
            )
        else:
            # Fill with zeros if no puzzle_ids provided
            self._pending_puzzle_ids = torch.cat(
                [
                    self._pending_puzzle_ids,
                    torch.zeros(
                        min(space, len(inputs)),
                        device=self.device,
                        dtype=torch.long,
                    ),
                ],
            )

    def _fill_pending(self) -> int:
        if len(self._pending_inputs) == 0:
            return 0

        num_to_fill = min(len(self._pending_inputs), self._state.max_fillable)
        if num_to_fill == 0:
            return 0

        carry = self.model.init_carry(num_to_fill)
        z_H = carry["z_H"].detach()
        z_L = carry["z_L"].detach()

        puzzle_ids = (
            self._pending_puzzle_ids[:num_to_fill]
            if len(self._pending_puzzle_ids) > 0
            else None
        )
        num_filled = self._state.fill(
            self._pending_inputs[:num_to_fill],
            self._pending_labels[:num_to_fill],
            z_H,
            z_L,
            puzzle_ids,
        )
        if num_filled > 0:
            self._pending_inputs = self._pending_inputs[num_filled:]
            self._pending_labels = self._pending_labels[num_filled:]
            if len(self._pending_puzzle_ids) > 0:
                self._pending_puzzle_ids = self._pending_puzzle_ids[num_filled:]
        return num_filled

    def _run_h_cycles(
        self,
        core: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]],
        embeddings: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> HCycleResult:
        return self.model.run_h_cycles(
            core,
            embeddings,
            z_H,
            z_L,
            cos_sin,
            collect_intermediates=self.collect_h_cycle_intermediates,
        )

    def _forward(self, state: TrainingState) -> ForwardResult:
        num_chains = state.num_chains
        active = state.batch_index >= 0
        cfg = self.config

        if not active.any():
            return ForwardResult(
                losses=torch.full(
                    (num_chains,),
                    fill_value=math.inf,
                    device=state.device,
                ),
                logits=torch.zeros(
                    num_chains,
                    cfg.num_puzzle_grid_tokens,
                    self.model.head.c_out,
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
                z_H_pre=state.z_H.clone(),
                z_L_pre=state.z_L.clone(),
            )

        batch_indices = state.batch_index.clamp(min=0)
        chain_inputs = state.inputs[batch_indices]
        chain_labels = state.labels[batch_indices]

        tokens = chain_inputs

        # Forward all chains (active and inactive - padding for compile)
        assert self.device is not None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            embeddings = self.model.embed_scale * self.model.embed_tokens(
                tokens,
                cfg.dtype,
            )

            # TRM-style prefix tokens: [puzzle_id_tokens..., register_tokens..., puzzle_grid...]
            chain_puzzle_ids = (
                state.puzzle_ids[batch_indices]
                if cfg.num_puzzle_id_tokens > 0
                else None
            )
            embeddings = self.model._prepend_prefix(embeddings, chain_puzzle_ids)  # noqa: SLF001

            # Add puzzle_id embedding to z_H if enabled (old additive approach).
            # NOTE: This is a separate feature from TRM-style prefix tokens above.
            # They use different embedding tables and shouldn't both be enabled.
            z_H = state.z_H
            if self.use_additive_puzzle_emb and self.puzzle_id_embed is not None:
                additive_puzzle_ids = state.puzzle_ids[batch_indices]
                # Remap to batch-local indices (modulo embedding table size)
                additive_local_ids = additive_puzzle_ids % self.max_puzzle_ids_per_batch
                puzzle_emb = self.puzzle_id_embed(
                    additive_local_ids
                )  # [num_chains, hidden]
                # Add to z_H (broadcast across sequence dimension)
                z_H = z_H + puzzle_emb.unsqueeze(1)

            core = (
                self.model.core_compiled
                if self.model.config.compile_core
                else self.model.core
            )
            z_L = state.z_L
            cos_sin = self.model._get_cos_sin(embeddings.device)  # noqa: SLF001
            hc = self._run_h_cycles(
                core,
                embeddings,
                z_H,
                z_L,
                cos_sin,
            )
            self._last_hc_result = hc
            logits, q_halt, z_H, z_L = hc.logits, hc.q_halt, hc.z_H, hc.z_L
            z_H_pre, z_L_pre = hc.z_H_pre, hc.z_L_pre

        # Slice output to exclude prefix positions
        # Sequence was: [puzzle_id_tokens..., register_tokens..., input...]
        # So we skip (puzzle_id + register) positions
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        logits = logits[:, n_prefix:, :]

        if self.label_smoothing_includes_pad_token:
            loss_labels = chain_labels.reshape(-1)
            loss_ignore = 0
        else:
            loss_labels = (chain_labels - 1).reshape(-1)
            loss_ignore = -1
        loss_per_token = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            loss_labels,
            label_smoothing=self.label_smoothing,
            ignore_index=loss_ignore,
            reduction="none",
        ).reshape(num_chains, -1)
        loss_count = (chain_labels != 0).sum(dim=-1).clamp(min=1)
        loss = loss_per_token.sum(dim=-1) / loss_count
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
            z_H_pre=z_H_pre,
            z_L_pre=z_L_pre,
        )

    def _compute_loss(
        self,
        forward_result: ForwardResult,
        state: TrainingState,
        active_samples: Tensor,
        winner_chains: Tensor,
        train_q_halt: bool,
    ) -> Tensor:
        """Compute the total training loss for one step.

        Override this in subclasses to add custom loss terms.
        """
        reduction = "sum" if self.loss_sum_normalize else "mean"
        loss = getattr(forward_result["losses"][winner_chains], reduction)()
        if train_q_halt:
            with torch.no_grad():
                predictions = self._preds(forward_result["logits"][winner_chains])
                correct = (
                    (predictions == state.labels[active_samples]).all(dim=-1).float()
                )
            loss = (
                loss
                + self.q_halt_weight
                * nn.functional.binary_cross_entropy_with_logits(
                    forward_result["q_halt"][winner_chains],
                    correct,
                    reduction=reduction,
                )
            )
        return loss

    def _probe_lip(
        self,
        forward_result: ForwardResult,
        winner_chains: Tensor,
        eps: float = 1e-3,
    ) -> float:
        """Estimate local Lipschitz constant of the reasoning block.

        Probes at z_H_pre + z_L_pre (the input to the last H-cycle's core
        call) for each active sample's winner chain. Uses a single random
        perturbation direction, averages the per-sample ratios.
        """
        if len(winner_chains) == 0:
            return 0.0
        x = (
            forward_result["z_H_pre"][winner_chains].detach()
            + forward_result["z_L_pre"][winner_chains].detach()
        )
        eps_vec = torch.randn_like(x)
        # Normalize per sample: each sample gets a unit-norm perturbation.
        norms = eps_vec.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
        eps_vec = eps_vec / norms.clamp(min=1e-12) * eps
        cos_sin = self.model._get_cos_sin(x.device)  # noqa: SLF001
        with torch.no_grad():
            f_x = self.model.reasoning(x, cos_sin)
            f_x_eps = self.model.reasoning(x + eps_vec, cos_sin)
        per_sample = (f_x_eps - f_x).flatten(1).norm(dim=1) / eps
        return per_sample.mean().item()

    def _backward(
        self,
        forward_result: ForwardResult,
        state: TrainingState,
        active_samples: Tensor,
        winner_chains: Tensor,
        train_q_halt: bool,
    ) -> None:
        """Compute loss and run backward pass.

        Override this in subclasses that need custom backward behavior
        (e.g. per-group gradient surgery).
        """
        assert self.device is not None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            loss = self._compute_loss(
                forward_result, state, active_samples, winner_chains, train_q_halt
            )
        if self.loss_sum_normalize:
            (loss / self.batch_size).backward()
        else:
            loss.backward()

    def _lr_scale(self) -> float:
        return lr_scale(
            self.current_step,
            self.total_train_steps,
            self.lr_warmup_steps,
            self.lr_min_ratio,
        )

    def _update_weights(self) -> None:
        self._grad_accum_counter += 1
        if self._grad_accum_counter < self.grad_accum_steps:
            return

        self._grad_accum_counter = 0
        if self.grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_max_norm,
            )

        lr_scale_ = self._lr_scale()
        assert self.optimizer1 is not None
        assert self.optimizer2 is not None
        for optimizer in [self.optimizer1, self.optimizer2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lr_scale_

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

    # --- Evaluation ---
    # NOTE: These eval methods don't pass puzzle_ids, so they're incompatible with
    # num_puzzle_id_tokens > 0. The model will raise a clear error in that case.

    def evaluate(
        self,
        loader: Any,
        train_loader: Any | None = None,
    ) -> EvalResult:
        ema = self.ema  # Local var for type narrowing
        use_ema = ema is not None and self.current_step >= self.ema_warmup_steps
        if use_ema:
            assert ema is not None  # Type guard
            ema.apply(self.model)

        try:
            self.model.eval()
            cell_acc, puzzle_acc, halt_cell, halt_puz = self.evaluate_act(loader)

            train_cell_acc, train_puzzle_acc = 0.0, 0.0
            train_halt_cell, train_halt_puz = 0.0, 0.0
            if train_loader is not None:
                saved_samples = self.max_eval_samples
                saved_path = self.eval_path_valid
                self.max_eval_samples = self.max_eval_samples_train
                self.eval_path_valid = False
                try:
                    (
                        train_cell_acc,
                        train_puzzle_acc,
                        train_halt_cell,
                        train_halt_puz,
                    ) = self.evaluate_act(train_loader)
                finally:
                    self.max_eval_samples = saved_samples
                    self.eval_path_valid = saved_path

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

            return EvalResult(
                cell_acc=cell_acc,
                puzzle_acc=puzzle_acc,
                halt_cell_acc=halt_cell,
                halt_puzzle_acc=halt_puz,
                train_cell_acc=train_cell_acc,
                train_puzzle_acc=train_puzzle_acc,
                train_halt_cell_acc=train_halt_cell,
                train_halt_puzzle_acc=train_halt_puz,
            )
        finally:
            self.model.train()
            if use_ema:
                assert ema is not None  # Type guard
                ema.restore(self.model)

    def evaluate_act(self, loader: Any) -> tuple[float, float, float, float]:
        """Returns (cell_acc, puzzle_acc, halt_cell_acc, halt_puzzle_acc)."""
        if self.eval_method == "wta":
            return self.evaluate_act_haltfast_wta(loader)
        if self.eval_method == "fast":
            return self.evaluate_act_haltfast(loader)
        return self.evaluate_act_full(loader)

    def _preds(self, logits: Tensor) -> Tensor:
        """Convert logits to token predictions, accounting for head offset."""
        p = logits.argmax(dim=-1)
        return p if self.label_smoothing_includes_pad_token else p + 1

    def _eval_forward(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        """Single ACT-step forward for eval. Override for prediction feedback."""
        if puzzle_ids is not None:
            return self.model(input_ids, z_H, z_L, puzzle_ids)
        return self.model(input_ids, z_H, z_L)

    def evaluate_act_haltfast(self, loader: Any) -> tuple[float, float, float, float]:
        """Fast eval: streaming replacement when q_halt > 0, fixed batch size."""
        assert self.device is not None
        bs = (
            self.eval_batch_size
            if self.eval_batch_size is not None
            else self.batch_size
        )
        B = bs

        total_puzzles = 0
        total_correct = 0
        total_cell_correct = 0
        total_cells = 0
        total_steps = 0
        total_path_valid = 0
        total_path_optimal = 0
        halt_histogram = [0] * self.current_max_steps

        grid_len = self.config.num_puzzle_grid_tokens
        inputs = torch.zeros(B, grid_len, device=self.device, dtype=torch.int32)
        labels = torch.zeros(B, grid_len, device=self.device, dtype=torch.int32)
        z_H, z_L = self._init_z(B)
        steps = torch.zeros(B, device=self.device, dtype=torch.int32)
        valid = torch.zeros(B, device=self.device, dtype=torch.bool)

        H_init_single = self.model.H_init[0]
        L_init_single = self.model.L_init[0]

        puzzle_ids = torch.zeros(B, device=self.device, dtype=torch.int32)
        puz_state = _PuzzleIterState(iter(loader), self.max_eval_samples)

        for i in range(B):
            inp, lab, pid = _get_next_puzzle(puz_state)
            if inp is not None:
                assert lab is not None
                inputs[i] = inp
                labels[i] = lab
                puzzle_ids[i] = pid if pid is not None else 0
                valid[i] = True

        if not valid.any():
            print("  cell_acc=0.00%, puzzle_acc=0.00%")
            return 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            while valid.any():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self._eval_forward(inputs, z_H, z_L, puzzle_ids)

                z_H = out["z_H"]
                z_L = out["z_L"]
                logits = out["logits"]
                q_halt = out["q_halt"]

                preds = self._preds(logits)
                steps += valid.int()
                halted = valid & ((q_halt > 0) | (steps >= self.current_max_steps))

                if halted.any():
                    correct_mask = preds == labels
                    cell_correct_per_sample = correct_mask.sum(dim=1)
                    puzzle_correct_per_sample = cell_correct_per_sample == grid_len

                    n_halted = halted.sum().item()
                    total_puzzles += n_halted
                    total_correct += puzzle_correct_per_sample[halted].sum().item()
                    total_cell_correct += cell_correct_per_sample[halted].sum().item()
                    total_cells += grid_len * n_halted
                    total_steps += steps[halted].sum().item()

                    if self.eval_path_valid:
                        H, W = self.eval_grid_shape
                        nv, no = check_maze_paths(preds[halted], labels[halted], H, W)
                        total_path_valid += nv
                        total_path_optimal += no

                    halted_steps = steps[halted].cpu().tolist()
                    for s in halted_steps:
                        if 1 <= s <= len(halt_histogram):
                            halt_histogram[s - 1] += 1

                    halted_idx = halted.nonzero(as_tuple=True)[0]
                    for idx_tensor in halted_idx:
                        idx = int(idx_tensor.item())
                        inp, lab, pid = _get_next_puzzle(puz_state)
                        if inp is not None:
                            assert lab is not None
                            inputs[idx] = inp
                            labels[idx] = lab
                            puzzle_ids[idx] = pid if pid is not None else 0
                            z_H[idx] = H_init_single
                            z_L[idx] = L_init_single
                            steps[idx] = 0
                            valid[idx] = True
                        else:
                            valid[idx] = False

        cell_acc = 100 * total_cell_correct / total_cells if total_cells > 0 else 0.0
        exact_acc = 100 * total_correct / total_puzzles if total_puzzles > 0 else 0.0
        puzzle_acc = exact_acc
        if self.eval_path_valid and total_puzzles > 0:
            pv = 100 * total_path_valid / total_puzzles
            po = 100 * total_path_optimal / total_puzzles
            puzzle_acc = po
            print(f"  path_valid={pv:.2f}%, exact_match={exact_acc:.2f}%")
        if total_puzzles > 0:
            print(f"  eval_avg_halt_step: {total_steps / total_puzzles:.2f}")
        # halt_acc == final_acc: this method already evaluates at halt time.
        return cell_acc, puzzle_acc, cell_acc, puzzle_acc

    def evaluate_act_haltfast_wta(
        self, loader: Any, progress: bool = False
    ) -> tuple[float, float, float, float]:
        """WTA eval: run K heads, first to halt wins. Streaming replacement."""
        assert self.device is not None
        start_time = time.perf_counter()
        seq_len = self._get_seq_len()
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

        grid_len = self.config.num_puzzle_grid_tokens
        inputs = torch.zeros(B, grid_len, device=self.device, dtype=torch.int32)
        labels = torch.zeros(B, grid_len, device=self.device, dtype=torch.int32)
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
            B,
            grid_len,
            self.model.head.c_out,
            device=self.device,
            dtype=self.dtype,
        )

        H_init_single = self.model.H_init[0]
        L_inits = []
        for k in range(self.K):
            if hasattr(self.model, f"L_init_{k}"):
                L_inits.append(getattr(self.model, f"L_init_{k}")[0])
            else:
                L_inits.append(self.model.L_init[0])

        puzzle_ids = torch.zeros(B, device=self.device, dtype=torch.int32)
        puz_state = _PuzzleIterState(iter(loader), self.max_eval_samples)

        for i in range(B):
            puzzle_idx = puz_state.samples_seen
            inp, lab, pid = _get_next_puzzle(puz_state)
            if inp is not None:
                assert lab is not None
                puzzle_ids[i] = pid if pid is not None else 0
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
            return 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            while valid.any():
                z_H_batched = z_H.reshape(B * self.K, seq_len, -1)
                z_L_batched = z_L.reshape(B * self.K, seq_len, -1)
                inputs_batched = (
                    inputs.unsqueeze(1).expand(B, self.K, -1).reshape(B * self.K, -1)
                )
                pids_batched = (
                    puzzle_ids.unsqueeze(1).expand(B, self.K).reshape(B * self.K)
                )

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self._eval_forward(
                        inputs_batched, z_H_batched, z_L_batched, pids_batched
                    )

                z_H = out["z_H"].reshape(B, self.K, seq_len, -1)
                z_L = out["z_L"].reshape(B, self.K, seq_len, -1)
                q_halt_stack = out["q_halt"].reshape(B, self.K)
                logits_stack = out["logits"].reshape(B, self.K, grid_len, -1)

                steps += valid.int()
                can_win = valid & ~has_winner
                halted_mask = q_halt_stack > 0
                any_halted = can_win & halted_mask.any(dim=1)

                if any_halted.any():
                    q_for_selection = q_halt_stack.clone()
                    q_for_selection[~halted_mask] = float("-inf")
                    winner_head = q_for_selection.argmax(dim=1)
                    batch_idx = torch.arange(B, device=self.device)
                    selected_logits = logits_stack[batch_idx, winner_head]
                    winner_logits[any_halted] = selected_logits[any_halted]
                    has_winner[any_halted] = True

                done = valid & (has_winner | (steps >= self.current_max_steps))

                if done.any():
                    fallback = done & ~has_winner
                    if fallback.any():
                        winner_logits[fallback] = logits_stack[fallback, 0]

                    done_idx = done.nonzero(as_tuple=True)[0]
                    for idx_tensor in done_idx:
                        idx = int(idx_tensor.item())
                        preds = self._preds(winner_logits[idx])
                        correct = (preds == labels[idx]).sum().item()
                        total_cells_correct += correct
                        total_cells += grid_len
                        total_correct += 1 if correct == grid_len else 0
                        total_puzzles += 1

                        if progress and total_puzzles % 1000 == 0:
                            elapsed = time.perf_counter() - start_time
                            print(
                                f"  {total_puzzles} puzzles, {elapsed:.1f}s, {total_puzzles / elapsed:.1f} puz/s",
                                flush=True,
                            )

                        puzzle_idx = puz_state.samples_seen
                        inp, lab, pid = _get_next_puzzle(puz_state)
                        if inp is not None:
                            assert lab is not None
                            puzzle_ids[idx] = pid if pid is not None else 0
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
        # halt_acc == final_acc: this method already evaluates at halt time.
        return cell_acc, puzzle_acc, cell_acc, puzzle_acc

    def evaluate_act_full(self, loader: Any) -> tuple[float, float, float, float]:
        """Full eval: run all steps with diagnostics.

        Reports both final-step accuracy (run all steps) and halt-time
        accuracy (predictions at the step where q_halt first fires).
        """
        assert self.device is not None
        correct = torch.tensor(0, device=self.device)
        puzzles_correct = torch.tensor(0, device=self.device)
        halt_correct = torch.tensor(0, device=self.device)
        halt_puzzles_correct = torch.tensor(0, device=self.device)
        total = 0
        total_halt_steps = torch.tensor(0.0, device=self.device)
        total_samples = 0
        total_path_valid = 0
        total_path_optimal = 0
        last_batch = None
        total_stuck = 0
        total_helped = 0
        samples_seen = 0

        grid_len = self.config.num_puzzle_grid_tokens

        with torch.no_grad():
            for batch in loader:
                if self.max_eval_samples >= 0 and samples_seen >= self.max_eval_samples:
                    break
                last_batch = batch
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                puzzle_ids = batch[2].to(self.device)
                valid_count = batch[3]
                B = inputs.shape[0]

                z_H, z_L = self._init_z(B)

                has_halted = torch.zeros(B, device=self.device, dtype=torch.bool)
                halt_preds = torch.zeros(
                    B, grid_len, device=self.device, dtype=torch.long
                )
                halt_step = torch.full(
                    (B,),
                    self.current_max_steps,
                    device=self.device,
                    dtype=torch.int32,
                )

                out: dict[str, Tensor] = {}
                for step in range(self.current_max_steps):
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                        out = self._eval_forward(inputs, z_H, z_L, puzzle_ids)
                    z_H = out["z_H"]
                    z_L = out["z_L"]
                    q_halt = out["q_halt"]

                    newly_halted = ~has_halted & (q_halt > 0)
                    if newly_halted.any():
                        halt_preds[newly_halted] = self._preds(
                            out["logits"][newly_halted]
                        )
                        halt_step[newly_halted] = step + 1
                        has_halted[newly_halted] = True

                # Non-halted samples: use final-step preds as halt preds
                if not has_halted.all():
                    halt_preds[~has_halted] = self._preds(out["logits"][~has_halted])

                total_halt_steps += halt_step[:valid_count].float().sum()
                total_samples += valid_count
                samples_seen += valid_count

                preds = self._preds(out["logits"])

                if self.enable_reset_retry:
                    stuck_mask = ~has_halted
                    if stuck_mask.any():
                        n_stuck = int(stuck_mask.sum().item())
                        total_stuck += n_stuck
                        stuck_idx = stuck_mask.nonzero(as_tuple=True)[0]
                        orig_preds_stuck = preds[stuck_idx]
                        labels_stuck = labels[stuck_idx]
                        was_wrong = ~(orig_preds_stuck == labels_stuck).all(dim=-1)

                        z_H_retry, z_L_retry = self._init_z(n_stuck)
                        inputs_stuck = inputs[stuck_idx]

                        out_retry: dict[str, Tensor] = {}
                        assert self.device is not None
                        puzzle_ids_stuck = puzzle_ids[stuck_idx]
                        for _ in range(self.current_max_steps):
                            with torch.autocast(
                                device_type=self.device.type,
                                dtype=self.dtype,
                            ):
                                out_retry = self._eval_forward(
                                    inputs_stuck, z_H_retry, z_L_retry, puzzle_ids_stuck
                                )
                            z_H_retry = out_retry["z_H"]
                            z_L_retry = out_retry["z_L"]

                        retry_preds = self._preds(out_retry["logits"])
                        now_correct = (retry_preds == labels_stuck).all(dim=-1)
                        total_helped += (was_wrong & now_correct).sum().item()
                        preds[stuck_idx] = retry_preds

                preds_valid = preds[:valid_count]
                halt_preds_valid = halt_preds[:valid_count]
                labels_valid = labels[:valid_count]
                correct += (preds_valid == labels_valid).sum()
                puzzles_correct += (preds_valid == labels_valid).all(dim=-1).sum()
                halt_correct += (halt_preds_valid == labels_valid).sum()
                halt_puzzles_correct += (
                    (halt_preds_valid == labels_valid).all(dim=-1).sum()
                )
                total += labels_valid.numel()

                if self.eval_path_valid:
                    H, W = self.eval_grid_shape
                    nv, no = check_maze_paths(preds_valid, labels_valid, H, W)
                    total_path_valid += nv
                    total_path_optimal += no

        cell_acc = 100 * correct.item() / total
        exact_acc = 100 * puzzles_correct.item() / total_samples
        halt_cell_acc = 100 * halt_correct.item() / total
        halt_exact_acc = 100 * halt_puzzles_correct.item() / total_samples
        puzzle_acc = exact_acc
        if self.eval_path_valid and total_samples > 0:
            pv = 100 * total_path_valid / total_samples
            po = 100 * total_path_optimal / total_samples
            puzzle_acc = po
            print(f"  path_valid={pv:.2f}%, exact_match={exact_acc:.2f}%")

        if total_samples > 0:
            print(
                f"  eval_avg_halt_step: {total_halt_steps.item() / total_samples:.2f}"
            )

        if self.enable_reset_retry and total_stuck > 0:
            print(
                f"  reset_retry: n_stuck={total_stuck} ({100 * total_stuck / total_samples:.1f}%), n_helped={total_helped} ({100 * total_helped / total_stuck:.1f}% of stuck)"
            )

        if last_batch is not None:
            with torch.no_grad():
                sample_inputs = last_batch[0].to(self.device)
                sample_labels = last_batch[1].to(self.device)
                sample_puzzle_ids = last_batch[2].to(self.device)
                diag_valid = last_batch[3]
                diag_B = sample_inputs.shape[0]

                z_H, z_L = self._init_z(diag_B)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    out = self.model(sample_inputs, z_H, z_L, sample_puzzle_ids)
                all_logits = list(out["all_logits"])
                all_z_H = list(out["all_z_H"])
                print_diagnostics(
                    all_logits, sample_labels, {"z_H": all_z_H}, valid_count=diag_valid
                )

                z_H, z_L = self._init_z(diag_B)
                step_logits = []
                step_q_halts = []
                for _ in range(self.current_max_steps):
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                        out = self.model.step(
                            sample_inputs, z_H, z_L, sample_puzzle_ids
                        )
                    step_logits.append(out["logits"])
                    step_q_halts.append(out["q_halt"])
                    z_H = out["z_H"]
                    z_L = out["z_L"]

                cell_accs = []
                puzzle_accs = []
                for lg in step_logits:
                    preds = self._preds(lg[:diag_valid])
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

        return cell_acc, puzzle_acc, halt_cell_acc, halt_exact_acc

    def _get_seq_len(self) -> int:
        return self.config.total_seq_len

    def _init_z(self, batch_size: int) -> tuple[Tensor, Tensor]:
        total_seq_len = self.config.total_seq_len
        z_H = self.model.H_init[0].expand(batch_size, total_seq_len, -1).contiguous()
        z_L = self.model.L_init[0].expand(batch_size, total_seq_len, -1).contiguous()
        return z_H, z_L

    def make_z_L_single(self, puzzle_idx: int) -> Tensor:
        del puzzle_idx
        _, l_indices = self.model._sample_head_indices()  # noqa: SLF001
        S = self.config.total_seq_len
        return self.model.L_init[l_indices].unsqueeze(1).expand(-1, S, -1)

    def make_checkpoint(self) -> dict[str, object]:
        assert self.optimizer1 is not None
        assert self.optimizer2 is not None
        checkpoint: dict[str, object] = {
            "model": self.model.state_dict(),
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "step": self.current_step,
            "best_acc": self.best_acc,
        }
        if self.ema:
            checkpoint["ema"] = dict(self.ema.shadow)
        if self.puzzle_id_embed is not None:
            checkpoint["puzzle_id_embed"] = self.puzzle_id_embed.state_dict()
        return checkpoint


@dataclasses.dataclass(kw_only=True, slots=True)
class TrainingState:
    """State for sparse chain pooling."""

    num_chains: int
    num_puzzle_grid_tokens: int
    total_seq_len: int
    hidden_size: int
    K: int
    device: torch.device
    dtype: torch.dtype

    z_H: Tensor = field(init=False)
    z_L: Tensor = field(init=False)
    batch_index: Tensor = field(init=False)
    h_step: Tensor = field(init=False)
    inputs: Tensor = field(init=False)
    labels: Tensor = field(init=False)
    puzzle_ids: Tensor = field(init=False)
    chain_indices: Tensor = field(init=False)
    active: Tensor = field(init=False)

    def __post_init__(self) -> None:
        num_batch_elements = self.num_chains // self.K
        self.z_H = torch.zeros(
            self.num_chains,
            self.total_seq_len,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.z_L = torch.zeros_like(self.z_H)
        self.batch_index = torch.full(
            [self.num_chains], fill_value=-1, device=self.device, dtype=torch.long
        )
        self.h_step = torch.zeros(
            num_batch_elements, device=self.device, dtype=torch.long
        )
        self.inputs = torch.zeros(
            num_batch_elements,
            self.num_puzzle_grid_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self.labels = torch.zeros_like(self.inputs)
        self.puzzle_ids = torch.zeros(
            num_batch_elements, device=self.device, dtype=torch.long
        )
        self.chain_indices = torch.full(
            [num_batch_elements, self.K],
            fill_value=-1,
            device=self.device,
            dtype=torch.long,
        )
        self.active = torch.zeros(
            num_batch_elements, device=self.device, dtype=torch.bool
        )

    @property
    def max_fillable(self) -> int:
        counts = torch.stack(
            [(self.batch_index == -1).sum(), (~self.active).sum()]
        ).tolist()
        num_free, num_inactive = int(counts[0]), int(counts[1])
        return min(num_free // self.K, num_inactive)

    def fill(
        self,
        inputs: Tensor,
        labels: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> int:
        num_to_fill = len(inputs)
        if num_to_fill == 0:
            return 0
        free_chains = (self.batch_index == -1).nonzero(as_tuple=True)[0]
        inactive = (~self.active).nonzero(as_tuple=True)[0]
        max_possible = min(len(free_chains) // self.K, len(inactive))
        if num_to_fill > max_possible:
            num_to_fill = max_possible
            if num_to_fill == 0:
                return 0
        free_chains = free_chains[: num_to_fill * self.K].reshape(num_to_fill, self.K)
        inactive = inactive[:num_to_fill]
        self.z_H[free_chains.reshape(-1)] = z_H[:num_to_fill].reshape(
            -1, *z_H.shape[2:]
        )
        self.z_L[free_chains.reshape(-1)] = z_L[:num_to_fill].reshape(
            -1, *z_L.shape[2:]
        )
        self.batch_index[free_chains.reshape(-1)] = (
            inactive.unsqueeze(1).expand(-1, self.K).reshape(-1)
        )
        self.chain_indices[inactive] = free_chains
        self.active[inactive] = True
        self.h_step[inactive] = 0
        self.inputs[inactive] = inputs[:num_to_fill].long()
        self.labels[inactive] = labels[:num_to_fill].long()
        if puzzle_ids is not None:
            self.puzzle_ids[inactive] = puzzle_ids[:num_to_fill].long()
        else:
            self.puzzle_ids[inactive] = 0
        return num_to_fill

    def select_winners(self, losses: Tensor) -> tuple[Tensor, Tensor]:
        if not self.active.any():
            empty: Tensor = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty
        active = self.active.nonzero(as_tuple=True)[0]
        chains = self.chain_indices[active]
        valid = chains >= 0
        loss_mat = losses[chains.clamp(min=0)]
        loss_mat = torch.where(
            valid, loss_mat, torch.full_like(loss_mat, fill_value=math.inf)
        )
        winner_k = loss_mat.argmin(dim=1)
        winner_chain = chains[torch.arange(len(active), device=self.device), winner_k]
        return active, winner_chain

    def expunge(self, losses: Tensor, threshold: float, min_h_step: int) -> int:
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
            torch.ones(self.K, self.K, device=self.device, dtype=torch.bool), diagonal=1
        )
        has_pair = (above_threshold & upper_triangular.unsqueeze(0)).any(dim=(1, 2))
        if not has_pair.any():
            return 0
        loss_matrix = losses[chains.clamp(min=0).reshape(-1)].reshape(
            len(eligible_indices), self.K
        )
        loss_matrix = torch.where(
            valid, loss_matrix, torch.full_like(loss_matrix, fill_value=-math.inf)
        )
        in_pair = above_threshold.any(dim=2) | above_threshold.any(dim=1)
        loss_matrix = torch.where(
            in_pair, loss_matrix, torch.full_like(loss_matrix, fill_value=-math.inf)
        )
        expunge_k = loss_matrix.argmax(dim=1)
        expunge_chain = chains[
            torch.arange(len(eligible_indices), device=self.device), expunge_k
        ]
        mask = has_pair & (expunge_chain >= 0)
        if not mask.any():
            return 0
        chains_to_free = expunge_chain[mask]
        self.batch_index[chains_to_free] = -1
        self.chain_indices[eligible_indices[mask], expunge_k[mask]] = -1
        return int(mask.sum().item())

    def release(self, batch_indices: Tensor) -> None:
        chains = self.chain_indices[batch_indices]
        self.batch_index[chains[chains >= 0]] = -1
        self.active[batch_indices] = False
        self.chain_indices[batch_indices] = -1


@runtime_checkable
class ExperimentProtocol(Protocol):
    """Duck-typed interface for experiments compatible with train()/main()."""

    @property
    def model(self) -> nn.Module: ...

    current_step: int
    max_train_sec: float
    eval_every_steps: int
    max_eval_samples_train: int

    def setup_optimizers(self) -> None: ...
    def make_train_loader(self) -> Any: ...
    def make_test_loader(self) -> Any: ...
    def step(
        self,
        inputs: Tensor,
        labels: Tensor,
        puzzle_ids: Tensor | None = None,
        valid_count: int | None = None,
    ) -> dict[str, Tensor] | None: ...
    def evaluate(
        self,
        loader: Any,
        train_loader: Any | None = None,
    ) -> EvalResult: ...
    def make_checkpoint(self) -> dict[str, object]: ...


def train(experiment: ExperimentProtocol):
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
    prev_step = experiment.current_step
    done = False

    def do_eval():
        nonlocal epoch_losses
        eval_start = time.perf_counter()

        train_eval_loader = (
            iter(trainloader) if experiment.max_eval_samples_train > 0 else None
        )
        er = experiment.evaluate(iter(testloader), train_loader=train_eval_loader)

        eval_time = time.perf_counter() - eval_start
        epoch_elapsed = eval_start - epoch_start
        train_loss = ""
        extra_stats = ""
        main_keys = {"lm", "n_halted"}
        if epoch_losses:
            first = epoch_losses[0]
            if isinstance(first, dict):
                keys: list[str] = list(first.keys())  # ty: ignore[invalid-assignment]
                main_parts = []
                extra_parts = []
                dicts: list[dict[str, Tensor]] = [  # ty: ignore[invalid-assignment]
                    d for d in epoch_losses if isinstance(d, dict)
                ]
                for k in keys:
                    vals = torch.stack([d[k] for d in dicts])
                    stat = f"{k}={vals.mean():.4f}±{vals.std():.4f}ϵ[{vals.min():.4f},{vals.max():.4f}]"
                    if k in main_keys:
                        main_parts.append(stat)
                    else:
                        extra_parts.append(stat)
                train_loss = "  Train " + " ".join(main_parts)
                if extra_parts:
                    extra_stats = "\n  " + " ".join(extra_parts)
            else:
                tensors = [t for t in epoch_losses if isinstance(t, Tensor)]
                losses = torch.stack(tensors)
                train_loss = f"  Train {losses.mean():.4f}±{losses.std():.4f} [{losses.min():.4f}, {losses.max():.4f}]"
            epoch_losses = []

        train_acc_str = ""
        if experiment.max_eval_samples_train > 0:
            train_acc_str = (
                f"  TrainAcc {er.train_cell_acc:5.2f}%/{er.train_puzzle_acc:5.2f}%"
                f" (halt: {er.train_halt_cell_acc:5.2f}%/{er.train_halt_puzzle_acc:5.2f}%)"
            )

        step = f"Step{experiment.current_step:6d}"
        acc_str = (
            f"  Test {er.cell_acc:5.2f}% / {er.puzzle_acc:5.2f}%"
            f" (halt: {er.halt_cell_acc:5.2f}%/{er.halt_puzzle_acc:5.2f}%)"
            f"{train_loss}{train_acc_str}"
        )
        timing = f"  ({eval_time:5.3f}s / {epoch_elapsed:6.3f}s)"

        print(f"{step}{acc_str}{timing}{extra_stats}", flush=True)
        return eval_start

    while True:
        done = done or (time.perf_counter() - train_start >= max_train_sec)

        if done:
            break

        try:
            samples, targets, puzzle_ids, valid_count = next(loader)
        except StopIteration:
            loader = iter(trainloader)
            continue

        prev_step = experiment.current_step
        loss = experiment.step(samples, targets, puzzle_ids, valid_count)
        if loss is None:
            done = True
        else:
            epoch_losses.append({k: v.detach() for k, v in loss.items()})

        step_changed = experiment.current_step > prev_step
        if (
            eval_every_steps
            and step_changed
            and experiment.current_step % eval_every_steps == 0
        ):
            eval_start = do_eval()
            epoch_start = time.perf_counter()
            train_start += epoch_start - eval_start
            experiment.model.train()

    return experiment.model


def main(
    experiment: ExperimentProtocol,
):
    """TRM main entry point."""
    script_path = pathlib.Path(sys.argv[0])
    with (
        Tee(script_path.with_suffix(".log"))
        if script_path.name
        else contextlib.nullcontext()
    ):
        experiment.setup_optimizers()
        return train(experiment)


def resume_from_checkpoint(
    exp: Any,
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

    # Reset carry state
    if hasattr(exp, "_state"):
        exp._state = TrainingState(  # noqa: SLF001
            num_chains=exp.num_chains,
            num_puzzle_grid_tokens=exp.config.num_puzzle_grid_tokens,
            total_seq_len=exp.config.total_seq_len,
            hidden_size=exp.config.hidden_size,
            K=exp.K,
            device=exp.device,
            dtype=exp.dtype,
        )
    print(f"Resumed from {ckpt_path} at step {exp.current_step}")


def cross_entropy_with_batched_smoothing(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float | Tensor = 0.0,
) -> Tensor:
    """Like ``nn.functional.cross_entropy``, but ``label_smoothing`` may be a per-element Tensor.

    When ``label_smoothing`` is a scalar float the call is forwarded to
    ``nn.functional.cross_entropy`` unchanged.  When it is a ``Tensor`` of the
    same shape as ``target``, each element receives its own smoothing value.

    """
    # Resolve deprecated size_average / reduce to a reduction string.
    if size_average is not None or reduce is not None:
        if reduce is not None and not reduce:
            reduction = "none"
        elif size_average is not None and not size_average:
            reduction = "sum"
        else:
            reduction = "mean"

    # Scalar smoothing: delegate to PyTorch.
    if isinstance(label_smoothing, (int, float)):
        return nn.functional.cross_entropy(
            input,
            target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    # --- Per-element (tensor) smoothing ---
    log_probs = input.log_softmax(dim=-1)
    nll = nn.functional.nll_loss(
        log_probs, target, ignore_index=ignore_index, reduction="none"
    )
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss

    mask = (target != ignore_index).float()
    loss = loss * mask

    if weight is not None:
        sample_weight: Tensor | None = weight[target.clamp(min=0)] * mask
        loss = loss * sample_weight
    else:
        sample_weight = None

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    # "mean"
    if sample_weight is not None:
        return loss.sum() / sample_weight.sum().clamp(min=1)
    return loss.sum() / mask.sum().clamp(min=1)


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
        if (
            p.ndim >= 2
            and "embed" not in n
            and "head" not in n
            and "register_tokens" not in n
        ):
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
            },
        )
    if muon_3d_params:
        muon_param_groups.append(
            {
                "params": list(muon_3d_params.values()),
                "ensemble_dims": 1,  # Per-head orthogonalization
            },
        )
    if muon_3d_qk_params:
        muon_param_groups.append(
            {
                "params": list(muon_3d_qk_params.values()),
                "ensemble_dims": 1,
                "weight_decay": qk_wd,
            },
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
