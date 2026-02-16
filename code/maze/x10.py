"""x10: x07 + PCGrad between ACT steps.

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
from maze.x07 import Experiment as Experiment07


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment07):
    def __init__(self) -> None:
        super().__init__()
        self._ref_grad: dict[str, Tensor] | None = None

    def reset_transient_state(self) -> None:
        super().reset_transient_state()
        self._ref_grad = None

    def _act_post_backward(
        self,
        carry: dict[str, Tensor],
        was_running: Tensor,
    ) -> None:
        step = int(carry["steps"][was_running][0].item())
        if step == 0:
            # Store step-0 gradient as reference direction.
            self._ref_grad = {
                n: p.grad.detach().clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None
            }
            return

        ref = self._ref_grad
        if ref is None:
            return

        # Project out conflicting component of current gradient.
        for n, p in self.model.named_parameters():
            if p.grad is None or n not in ref:
                continue
            g = p.grad
            r = ref[n]
            dot = (g * r).sum()
            if dot < 0:
                p.grad = g - (dot / (r.norm().square() + 1e-12)) * r


if __name__ == "__main__":
    main(Experiment())
