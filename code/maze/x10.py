"""x12: x07a + Lipschitz penalty on reasoning block.

Motivation: The shared reasoning block is applied iteratively (L_cycles
per H-cycle, H_cycles per ACT step). If its Jacobian spectral radius
drifts above 1 as it fits hard puzzles, small perturbations amplify
across iterations, causing post-saturation oscillation.

Fix: Penalize the local Lipschitz constant of the reasoning block via
a single random-direction finite-difference probe per ACT step:
  L_lip = (||f(x+eps) - f(x)|| / ||eps||)^2
This encourages the reasoning map to be contractive, stabilizing the
iterative application without expensive eigenvalue computation.

References:
- Bai et al. (2022), "Stabilizing Equilibrium Models by Jacobian
  Regularization" — penalizes spectral radius of DEQ iteration map
- Miller & Hardt (2019), "Stable Recurrent Models" — constrains
  recurrent Jacobian spectral radius < 1 for stability
- Behrmann et al. (2019), "Invertible Residual Networks" — Lip(g) < 1
  ensures contractivity of residual blocks

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x07a import Experiment as Experiment07a

import torch


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment07a):
    lip_weight: float = 0.1
    lip_eps: float = 1e-3

    def _compute_act_loss(
        self,
        carry: dict[str, Tensor],
        was_running: Tensor,
        logits: Tensor,
        all_logits: list[Tensor],
        q_halt: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        total_loss, loss_dict = super()._compute_act_loss(  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
            carry,
            was_running,
            logits,
            all_logits,
            q_halt,
        )
        # Probe reasoning block Lipschitz constant via finite difference.
        x = carry["z_L"][:1].detach()
        eps_vec = torch.randn_like(x)
        eps_vec = eps_vec / eps_vec.norm() * self.lip_eps
        cos_sin = self.model._get_cos_sin(x.device)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert self.device is not None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            f_x = self.model.reasoning(x, cos_sin)
            f_x_eps = self.model.reasoning(x + eps_vec, cos_sin)
        ratio = (f_x_eps - f_x).norm() / self.lip_eps
        lip_loss = self.lip_weight * ratio.square()
        return total_loss + lip_loss, loss_dict


if __name__ == "__main__":
    main(Experiment())
