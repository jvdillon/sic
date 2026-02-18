"""x10: x07a + PCGrad between ACT steps.

Motivation: Late ACT steps see only hard (unhalted) puzzles, producing
gradients that conflict with early-step gradients (which see all puzzles).
This destroys the shared reasoning block's performance on easy puzzles,
causing post-saturation oscillation.

Fix: Store the gradient from ACT step 0. Before applying subsequent steps'
gradients, project out the component that conflicts with step 0's direction.
This prevents late-step updates from undoing early-step progress.

References:
- Yu et al. (2020), "Gradient Surgery for Multi-Task Learning" (PCGrad)
- Liu et al. (2021), "Conflict-Averse Gradient Descent" (CAGrad)
Novel application to ACT steps (each step treated as a separate task).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x07a import Experiment as Experiment07a

import torch


if TYPE_CHECKING:
    from experiment import ForwardResult, TrainingState
    from torch import Tensor


class Experiment(Experiment07a):
    def _backward(
        self,
        forward_result: ForwardResult,
        state: TrainingState,
        active_samples: Tensor,
        winner_chains: Tensor,
        train_q_halt: bool,
    ) -> None:
        # Partition active samples by h_step.
        h_steps = state.h_step[active_samples]
        unique_steps = h_steps.unique(sorted=True)

        if len(unique_steps) <= 1:
            # All samples at same step — no conflict possible, use base path.
            super()._backward(
                forward_result, state, active_samples, winner_chains, train_q_halt
            )
            return

        ref_grad: dict[str, Tensor] | None = None

        for step_val in unique_steps:
            mask = h_steps == step_val
            group_active = active_samples[mask]
            group_winners = winner_chains[mask]

            # Compute and backward this group's loss.
            assert self.device is not None
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                loss = self._compute_loss(
                    forward_result, state, group_active, group_winners, train_q_halt
                )
            if self.loss_sum_normalize:
                (loss / self.batch_size).backward()
            else:
                loss.backward()

            if step_val == 0:
                # Store step-0 gradient as reference direction.
                ref_grad = {
                    n: p.grad.detach().clone()
                    for n, p in self.model.named_parameters()
                    if p.grad is not None
                }
            elif ref_grad is not None:
                # Project out conflicting component of current gradient.
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if p.grad is None or n not in ref_grad:
                            continue
                        g = p.grad
                        r = ref_grad[n]
                        dot = (g * r).sum()
                        if dot < 0:
                            p.grad = g - (dot / (r.norm().square() + 1e-12)) * r


if __name__ == "__main__":
    main(Experiment())
