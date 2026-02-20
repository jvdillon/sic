"""x000g: x000 + use_ema=False, attn_muon_modified, 2D RoPE."""

from typing import cast

import dataclasses

from experiment import main
from maze.old.x000 import Experiment as Experiment000
from model import (
    TRM3,
    Attention,
    SwiGLU,
    TransformerBlock,
    TRM3ConfigProtocol,
    trunc_normal_init_,
)


class Experiment(Experiment000):
    use_ema: bool = False
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
        block=TransformerBlock.Config(
            attn=Attention.Config(
                num_heads=8,
                muon_modified=True,
                checkpoint_muon_norm=True,
                init_weight_fn=trunc_normal_init_,
            ),
            ffn=SwiGLU.Config(
                multiple_of=128,
                init_weight_fn=trunc_normal_init_,
            ),
        ),
    )


if __name__ == "__main__":
    main(Experiment())
