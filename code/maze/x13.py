"""x13: x07 + per-ACT-step label smoothing decay.

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
    smooth_start: float = 0.375
    smooth_end: float = 0.0625

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
        # Per-step label smoothing: linearly interpolate based on ACT step.
        step = int(carry["steps"][was_running][0].item())
        max_step = self.max_reasoning_steps - 1
        t = min(step / max(max_step, 1), 1.0)
        old_smooth = self.label_smoothing
        self.label_smoothing = self.smooth_start + t * (
            self.smooth_end - self.smooth_start
        )
        result = super()._compute_act_loss(
            carry, was_running, logits, all_logits, q_halt,
        )
        self.label_smoothing = old_smooth
        return result

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
