"""x000f: x000 with attn_muon_modified=True."""

from typing import cast

import dataclasses
import functools

from experiment import main
from maze.x000 import Experiment as Experiment000
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_


class Experiment(Experiment000):
    # label_smoothing: float = 0.0

    use_ema: bool = False
    # ema_decay: float = 0.999
    # ema_warmup_steps: int = 0

    # q_halt_weight: float = 0.05  # 0.5
    lr_warmup_steps: int = 0  # 2000
    lr_min_ratio: float = 1.0
    # grad_clip_max_norm: float | None = None
    loss_ignore_index: int | None = 0
    cast_model_to_dtype: bool = False
    loss_sum_normalize: bool = True

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
        vocab_size=6,
        L_cycles=4,
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
            attn_init_weight_fn=trunc_normal_init_,
            attn_muon_modified=True,
            attn_checkpoint_muon_norm=True,
        ),
    )


if __name__ == "__main__":
    main(Experiment())
