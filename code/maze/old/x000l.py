"""x000l: x000 + 16k steps, attn_muon_modified, no EMA, sum-norm loss."""

from typing import cast

import dataclasses
import functools

from experiment import main
from maze.old.x000 import Experiment as Experiment000
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_


class Experiment(Experiment000):
    total_train_steps: int = 16_000
    use_ema: bool = False
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 1.0
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
