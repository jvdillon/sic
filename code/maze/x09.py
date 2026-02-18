"""x11: x07a + EMA parameter mixing after each ACT step.

Motivation: Late ACT steps produce sparse, high-variance gradients from
hard puzzles that push the shared reasoning block away from the region
that works for easy puzzles. This causes post-saturation oscillation.

Fix: After each optimizer step, mix current parameters toward an EMA:
theta <- (1 - alpha) * theta + alpha * theta_ema. This prevents any
single ACT step from moving parameters too far from the stable region.
Unlike weight decay (which pulls toward zero), this pulls toward the
running average — a much better attractor.

References:
- Herbster & Warmuth (1998), "Tracking the Best Expert" (Fixed-Share)
- Polyak & Juditsky (1992), iterate averaging for SGD
Analogy: Fixed-Share mixes expert weights uniformly after each round
to track shifting targets. Here we mix parameters toward their EMA to
damp high-variance late-step updates.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x07a import Experiment as Experiment07a

import torch


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment07a):
    act_ema_decay: float = 0.999
    act_ema_mix: float = 0.01

    def __init__(self) -> None:
        super().__init__()
        self._act_ema: dict[str, Tensor] | None = None

    def reset_transient_state(self) -> None:
        super().reset_transient_state()
        self._act_ema = None

    def _act_post_backward(
        self,
        carry: dict[str, Tensor],
        was_running: Tensor,
    ) -> None:
        del carry, was_running

    def _update_weights(self) -> None:
        super()._update_weights()
        # Initialize EMA on first call.
        if self._act_ema is None:
            self._act_ema = {
                n: p.data.clone() for n, p in self.model.named_parameters()
            }
            return
        ema = self._act_ema
        decay = self.act_ema_decay
        mix = self.act_ema_mix
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                # Update EMA: ema <- decay * ema + (1-decay) * theta
                ema[n].lerp_(p.data, 1.0 - decay)
                # Mix parameters toward EMA: theta <- (1-mix)*theta + mix*ema
                p.data.lerp_(ema[n], mix)


if __name__ == "__main__":
    main(Experiment())
