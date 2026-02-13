"""x12: x07 + Lipschitz penalty on reasoning block.

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

import functools

from experiment import (
    Experiment as ExperimentBase,
    main,
    setup_muon_optimizers,
)
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_

import torch

from data import get_puzzle_config


if TYPE_CHECKING:
    from torch import Tensor

_CFG = get_puzzle_config("maze")


class Experiment(ExperimentBase):
    batch_size: int = 128
    lip_weight: float = 0.1
    lip_eps: float = 1e-3

    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    augment_sudoku: bool = False
    eval_every_steps: int = 500
    eval_batch_size: int | None = 256
    K: int = 1
    q_halt_weight: float = 0.05
    total_train_steps: int = 16_000
    use_ema: bool = False
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 1.0
    loss_ignore_index: int | None = 0
    cast_model_to_dtype: bool = False
    loss_sum_normalize: bool = True

    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        num_puzzle_grid_tokens=_CFG.grid_len,
        num_layers=2,
        H_cycles=3,
        L_cycles=4,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=False,
        use_rope=True,
        num_heads=8,
        num_puzzle_id_tokens=1,
        num_puzzle_ids=1,
        num_register_tokens=15,
        register_token_init_std=0.0,
        register_tokens_learnable=False,
        q_halt_seq_index=0,
        cast_model_to_dtype=False,
        block_fn=functools.partial(
            TransformerBlock,
            multiple_of=128,
            num_heads=8,
            mlp_init_weight_fn=trunc_normal_init_,
            mlp_muon_modified=True,
            attn_init_weight_fn=trunc_normal_init_,
            attn_muon_modified=True,
            attn_checkpoint_muon_norm=True,
            attn_qk_norm=False,
        ),
    )

    def _compute_act_loss(
        self,
        carry: dict[str, Tensor],
        was_running: Tensor,
        logits: Tensor,
        all_logits: list[Tensor],
        q_halt: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        total_loss, loss_dict = super()._compute_act_loss(  # pyright: ignore[reportAttributeAccessIssue]
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
        cos_sin = None if self.model.rope is None else self.model.rope()
        assert self.device is not None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            f_x = self.model.reasoning(x, cos_sin)
            f_x_eps = self.model.reasoning(x + eps_vec, cos_sin)
        ratio = (f_x_eps - f_x).norm() / self.lip_eps
        lip_loss = self.lip_weight * ratio.square()
        return total_loss + lip_loss, loss_dict

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
