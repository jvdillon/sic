"""x11: x07a + per-ACT-step label smoothing decay.

Motivation: Early ACT steps have noisy predictions -- high label smoothing
prevents overconfident gradients from bad reasoning states. Later steps
have refined predictions, so harder targets let the model sharpen.

Fix: Linearly decay label_smoothing from smooth_start to smooth_end
across ACT steps. Step 0 gets maximum smoothing; the final step gets
minimum (or zero).

References:
- Zheng et al. (2022), "Confidence-aware label smoothing"
- Furlanello et al. (2018), "Born-Again Networks" -- soft-to-hard targets
Novel application: schedule tied to ACT step index, not training epoch.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import cross_entropy_with_batched_smoothing, main
from maze.x07a import Experiment as Experiment07a
from torch import nn

import torch


if TYPE_CHECKING:
    from experiment import ForwardResult, TrainingState
    from torch import Tensor


class Experiment(Experiment07a):
    smooth_start: float = 0.375
    smooth_end: float = 0.0625

    def _compute_loss(
        self,
        forward_result: ForwardResult,
        state: TrainingState,
        active_samples: Tensor,
        winner_chains: Tensor,
        train_q_halt: bool,
    ) -> Tensor:
        # Get logits and labels for winner chains.
        w_logits = forward_result["logits"][winner_chains]
        w_labels = state.labels[active_samples]

        # Per-sample smoothing: linear interpolation based on ACT step.
        h_steps = state.h_step[active_samples].float()
        max_step = self.max_reasoning_steps - 1
        t = (h_steps / max(max_step, 1)).clamp(max=1.0)
        smoothing = self.smooth_start + t * (self.smooth_end - self.smooth_start)

        # Recompute LM loss with per-sample batched smoothing.
        if self.label_smoothing_includes_pad_token:
            loss_labels = w_labels.reshape(-1)
            loss_ignore = 0
        else:
            loss_labels = (w_labels - 1).reshape(-1)
            loss_ignore = -1

        W, S = w_labels.shape
        smoothing_per_token = smoothing.unsqueeze(1).expand(W, S).reshape(-1)

        per_token_loss = cross_entropy_with_batched_smoothing(
            w_logits.reshape(-1, w_logits.shape[-1]),
            loss_labels,
            ignore_index=loss_ignore,
            reduction="none",
            label_smoothing=smoothing_per_token,
        ).reshape(W, S)

        loss_count = (w_labels != 0).sum(dim=-1).clamp(min=1)
        per_chain_loss = per_token_loss.sum(dim=-1) / loss_count

        reduction = "sum" if self.loss_sum_normalize else "mean"
        loss = getattr(per_chain_loss, reduction)()

        # q_halt loss (same as base).
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


if __name__ == "__main__":
    main(Experiment())
