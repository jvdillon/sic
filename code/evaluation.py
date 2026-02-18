"""TRM evaluation utilities.

Provides diagnostics and metrics for TRM experiments.
"""

from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor
from torch.nn import functional

import torch


class _EvalModelProtocol(Protocol):
    @property
    def H_init(self) -> Tensor: ...

    @property
    def L_init(self) -> Tensor: ...

    @property
    def config(self) -> Any: ...

    def step(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Any]: ...


# Re-export for backwards compat
__all__ = [
    "HALT_TOKEN_ID",
    "cells_fixed_broken",
    "evaluate_multistart",
    "evaluate_single_start",
    "h_cosine_similarities",
    "per_h_accuracy",
    "print_diagnostics",
    "run_act_steps",
    "z_h_deltas",
]


HALT_TOKEN_ID = 11


def run_act_steps(
    model: _EvalModelProtocol,
    inputs: Tensor,
    z_H: Tensor,
    z_L: Tensor,
    max_steps: int,
    device: torch.device,  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    dtype: torch.dtype,  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
) -> tuple[Tensor, Tensor, Tensor]:
    """Run ACT steps and return predictions, q_halt, and final z_H."""
    out: dict[str, Tensor] = {}
    for _ in range(max_steps):
        out = model.step(inputs, z_H, z_L)
        z_H = out["z_H"]
        z_L = out["z_L"]

    preds = out["logits"].argmax(dim=-1)
    q_halt = out["q_halt"]
    return preds, q_halt, z_H


def evaluate_single_start(
    model: _EvalModelProtocol,
    inputs: Tensor,
    labels: Tensor,
    max_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Evaluate model with single start point. Returns (cell_acc, puzzle_acc)."""
    B = inputs.shape[0]
    seq_len = model.config.seq_len

    z_H = model.H_init.expand(B, seq_len, -1)
    z_L = model.L_init.expand(B, seq_len, -1)

    with torch.no_grad():
        preds, _, _ = run_act_steps(model, inputs, z_H, z_L, max_steps, device, dtype)

    cell_acc = (preds == labels).float().mean().item() * 100
    puzzle_acc = (preds == labels).all(dim=-1).float().mean().item() * 100
    return cell_acc, puzzle_acc


def evaluate_multistart(
    model: _EvalModelProtocol,
    inputs: Tensor,
    labels: Tensor,
    max_steps: int,
    n_starts: int,
    z_L_noise: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Evaluate model with multiple start points (noise in z_L). Returns (cell_acc, puzzle_acc)."""
    B = inputs.shape[0]
    seq_len = model.config.seq_len
    hidden_size = model.config.hidden_size

    best_preds = None
    best_q_halt = torch.full((B,), float("-inf"), device=device)

    for _ in range(n_starts):
        z_H = model.H_init.expand(B, seq_len, -1)
        z_L = model.L_init.expand(B, seq_len, -1)
        if z_L_noise > 0:
            noise = torch.randn(B, seq_len, hidden_size, device=device, dtype=dtype)
            z_L = z_L + noise * z_L_noise

        with torch.no_grad():
            preds, q_halt, _ = run_act_steps(
                model,
                inputs,
                z_H,
                z_L,
                max_steps,
                device,
                dtype,
            )

        better = q_halt > best_q_halt
        if best_preds is None:
            best_preds = preds
        else:
            best_preds[better] = preds[better]
        best_q_halt = torch.maximum(best_q_halt, q_halt)

    cell_acc = (best_preds == labels).float().mean().item() * 100  # ty: ignore[unresolved-attribute]
    puzzle_acc = (best_preds == labels).all(dim=-1).float().mean().item() * 100  # ty: ignore[unresolved-attribute]
    return cell_acc, puzzle_acc


def per_h_accuracy(
    all_logits: list[Tensor],
    labels: Tensor,
    valid_count: int | None = None,
) -> list[float]:
    """Accuracy at each H iteration."""
    accs = []
    v = valid_count if valid_count is not None else labels.shape[0]
    for logits in all_logits:
        preds = logits[:v].argmax(dim=-1)
        acc = (preds == labels[:v]).float().mean().item() * 100
        accs.append(acc)
    return accs


def h_cosine_similarities(
    all_logits: list[Tensor],
    valid_count: int | None = None,
) -> list[float]:
    """Cosine similarity between consecutive H iterations."""
    v = valid_count if valid_count is not None else all_logits[0].shape[0]
    cos_sims = []
    for i in range(1, len(all_logits)):
        z_prev = all_logits[i - 1][:v].reshape(v, -1)
        z_curr = all_logits[i][:v].reshape(v, -1)
        cos = functional.cosine_similarity(z_prev, z_curr, dim=1).mean().item()
        cos_sims.append(cos)
    return cos_sims


def z_h_deltas(z_H: list[Tensor], valid_count: int | None = None) -> list[float]:
    """L2 norm of z_H changes between iterations."""
    v = valid_count if valid_count is not None else z_H[0].shape[0]
    deltas = []
    for i in range(1, len(z_H)):
        delta = torch.norm(z_H[i][:v] - z_H[i - 1][:v], dim=-1).mean().item()
        deltas.append(delta)
    return deltas


def cells_fixed_broken(
    all_logits: list[Tensor],
    labels: Tensor,
    valid_count: int | None = None,
) -> tuple[list[int], list[int]]:
    """Count cells fixed/broken at each H iteration."""
    v = valid_count if valid_count is not None else labels.shape[0]
    preds_list = [logits[:v].argmax(dim=-1) for logits in all_logits]
    labels_v = labels[:v]
    correct_list = [p == labels_v for p in preds_list]
    cells_fixed, cells_broken = [], []
    for h in range(1, len(all_logits)):
        fixed = (~correct_list[h - 1] & correct_list[h]).sum().item()
        broken = (correct_list[h - 1] & ~correct_list[h]).sum().item()
        cells_fixed.append(fixed)
        cells_broken.append(broken)
    return cells_fixed, cells_broken


def print_diagnostics(
    all_logits: list[Tensor],
    labels: Tensor,
    z_info: dict[str, list[Tensor]],
    valid_count: int | None = None,
) -> None:
    """Print standard TRM diagnostics.

    Args:
        all_logits: List of logits at each H cycle (halt token already stripped).
        labels: Ground truth labels (B, seq_len).
        z_info: Dict with 'z_H' key containing list of z_H states.
        valid_count: Number of valid samples (excludes padding). If None, uses all.

    """
    accs = per_h_accuracy(all_logits, labels, valid_count)
    print(f"  per_H_acc: {[f'H{i}={a:.1f}%' for i, a in enumerate(accs)]}", flush=True)

    cos_sims = h_cosine_similarities(all_logits, valid_count)
    print(f"  H_cos: {[f'{c:.3f}' for c in cos_sims]}", flush=True)

    deltas = z_h_deltas(z_info["z_H"], valid_count)
    print(f"  z_H_delta: {[f'{n:.3f}' for n in deltas]}", flush=True)

    fixed, broken = cells_fixed_broken(all_logits, labels, valid_count)
    print(f"  cells: fixed={fixed} broken={broken}", flush=True)
