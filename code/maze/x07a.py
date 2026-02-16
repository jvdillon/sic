"""x07a: x07 + attn_qk_norm=True, eval_method="fast".

x07.log is eval_method="fast".
x07.log.1 is eval_method="full".
"""

from typing import Literal

import functools

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_

from data import get_puzzle_config


_CFG = get_puzzle_config("maze")


class Experiment(Experiment07):
    eval_method: Literal["full", "fast", "wta"] = "fast"

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
        label_smoothing_includes_pad_token=False,
        block_fn=functools.partial(
            TransformerBlock,
            multiple_of=128,
            num_heads=8,
            mlp_init_weight_fn=trunc_normal_init_,
            mlp_muon_modified=True,
            attn_init_weight_fn=trunc_normal_init_,
            attn_muon_modified=True,
            attn_checkpoint_muon_norm=True,
            attn_qk_norm=True,
        ),
    )


if __name__ == "__main__":
    main(Experiment())
