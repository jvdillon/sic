"""x005: x000l with per-cell regret loss weighting.

Cells that are persistently wrong across ACT steps get upweighted
in the loss: weight = 1 + alpha * wrong_count.
"""

from __future__ import annotations

import math

from experiment import ForwardResult, TrainingState, main
from maze.x000l import Experiment as Experiment000l
from torch import nn

import torch


class Experiment(Experiment000l):
    regret_alpha: float = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._init_wrong_count()

    def _init_wrong_count(self) -> None:
        num_elems = self._state.inputs.shape[0]
        seq_len = self.config.num_puzzle_grid_tokens
        self._wrong_count = torch.zeros(
            num_elems, seq_len, device=self.device, dtype=torch.float32
        )

    def reset_transient_state(self) -> None:
        super().reset_transient_state()
        self._init_wrong_count()

    def _forward(self, state: TrainingState) -> ForwardResult:
        # Reset wrong_count for newly filled samples (h_step == 0)
        if state.active.any():
            new_samples = state.active & (state.h_step == 0)
            if new_samples.any():
                self._wrong_count[new_samples] = 0

        result = super()._forward(state)
        chain_active = state.batch_index >= 0
        if not chain_active.any():
            return result

        batch_indices = state.batch_index.clamp(min=0)
        chain_labels = state.labels[batch_indices]
        # Recompute loss with regret weighting
        if self.label_smoothing_includes_pad_token:
            loss_labels = chain_labels.reshape(-1)
            loss_ignore = 0
        else:
            loss_labels = (chain_labels - 1).reshape(-1)
            loss_ignore = -1
        loss_per_token = nn.functional.cross_entropy(
            result["logits"].reshape(-1, result["logits"].shape[-1]),
            loss_labels,
            label_smoothing=self.label_smoothing,
            ignore_index=loss_ignore,
            reduction="none",
        ).reshape(state.num_chains, -1)

        weight = 1.0 + self.regret_alpha * self._wrong_count[batch_indices]
        loss_per_token = loss_per_token * weight

        valid_mask = chain_labels != 0
        weight_sum = (weight * valid_mask).sum(dim=-1).clamp(min=1)
        loss = loss_per_token.sum(dim=-1) / weight_sum
        result["losses"] = torch.where(
            chain_active,
            loss,
            torch.full_like(loss, fill_value=math.inf),
        )

        # Update wrong_count for next ACT step (K=1: chains map 1:1 to batches)
        with torch.no_grad():
            preds = result["logits"][chain_active].detach().argmax(dim=-1)
            active_batch = batch_indices[chain_active]
            active_labels = state.labels[active_batch]
            wrong = (preds != active_labels) & (active_labels != 0)
            self._wrong_count[active_batch] += wrong.float()

        return result


if __name__ == "__main__":
    main(Experiment())
