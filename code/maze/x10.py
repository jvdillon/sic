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

import functools

from experiment import (
    Experiment as ExperimentBase,
    main,
    setup_muon_optimizers,
)
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_

from data import get_puzzle_config


if TYPE_CHECKING:
    from torch import Tensor

_CFG = get_puzzle_config("maze")


class Experiment(ExperimentBase):
    batch_size: int = 128

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

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
