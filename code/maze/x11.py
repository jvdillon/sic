"""x13: x07a + per-ACT-step label smoothing decay.

Motivation: Early ACT steps have noisy predictions — high label smoothing
prevents overconfident gradients from bad reasoning states. Later steps
have refined predictions, so harder targets let the model sharpen.

Fix: Linearly decay label_smoothing from smooth_start to smooth_end
across ACT steps. Step 0 gets maximum smoothing; the final step gets
minimum (or zero).

References:
- Zheng et al. (2022), "Confidence-aware label smoothing"
- Furlanello et al. (2018), "Born-Again Networks" — soft-to-hard targets
Novel application: schedule tied to ACT step index, not training epoch.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x07a import Experiment as Experiment07a


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment07a):
    smooth_start: float = 0.375
    smooth_end: float = 0.0625

    def _compute_act_loss(
        self,
        carry: dict[str, Tensor],
        was_running: Tensor,
        logits: Tensor,
        all_logits: list[Tensor],
        q_halt: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # Per-step label smoothing: linearly interpolate based on ACT step.
        step = int(carry["steps"][was_running][0].item())
        max_step = self.max_reasoning_steps - 1
        t = min(step / max(max_step, 1), 1.0)
        old_smooth = self.label_smoothing
        self.label_smoothing = self.smooth_start + t * (
            self.smooth_end - self.smooth_start
        )
        result = super()._compute_act_loss(  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            carry,
            was_running,
            logits,
            all_logits,
            q_halt,
        )
        self.label_smoothing = old_smooth
        return result


if __name__ == "__main__":
    main(Experiment())
