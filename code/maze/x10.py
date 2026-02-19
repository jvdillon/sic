"""x10: x07a + Lipschitz penalty on reasoning block.

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
  Regularization" -- penalizes spectral radius of DEQ iteration map
- Miller & Hardt (2019), "Stable Recurrent Models" -- constrains
  recurrent Jacobian spectral radius < 1 for stability
- Behrmann et al. (2019), "Invertible Residual Networks" -- Lip(g) < 1
  ensures contractivity of residual blocks

"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import dataclasses

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, TRM3ConfigProtocol

import torch


if TYPE_CHECKING:
    from experiment import ForwardResult, TrainingState
    from torch import Tensor


class Experiment(Experiment07a):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07a.config),
        block_kwargs_by_layer={
            0: {"checkpoint": True},
            1: {"checkpoint": True},
        },
    )

    lip_weight: float = 0.1
    lip_eps: float = 1e-3
    log_lip: bool = True

    def _compute_loss(
        self,
        forward_result: ForwardResult,
        state: TrainingState,
        active_samples: Tensor,
        winner_chains: Tensor,
        train_q_halt: bool,
    ) -> Tensor:
        loss = super()._compute_loss(
            forward_result,
            state,
            active_samples,
            winner_chains,
            train_q_halt,
        )
        # Penalize reasoning block Lipschitz constant via finite difference
        # at each active sample's winner chain. z_H_pre and z_L_pre are the
        # inputs to the last H-cycle's core() call -- the operating point
        # where contractivity matters most.
        x = (
            forward_result["z_H_pre"][winner_chains].detach()
            + forward_result["z_L_pre"][winner_chains].detach()
        )
        eps_vec = torch.randn_like(x)
        norms = eps_vec.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
        eps_vec = eps_vec / norms.clamp(min=1e-12) * self.lip_eps
        cos_sin = self.model._get_cos_sin(x.device)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert self.device is not None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            f_x = self.model.reasoning(x, cos_sin)
            f_x_eps = self.model.reasoning(x + eps_vec, cos_sin)
        per_sample = (f_x_eps - f_x).flatten(1).norm(dim=1) / self.lip_eps
        lip_loss = self.lip_weight * per_sample.square().mean()
        return loss + lip_loss


if __name__ == "__main__":
    main(Experiment())
