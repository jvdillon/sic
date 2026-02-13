"""x11: x07 + EMA parameter mixing after each ACT step.

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
    act_ema_decay: float = 0.999
    act_ema_mix: float = 0.01

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

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
