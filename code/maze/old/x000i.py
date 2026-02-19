"""x000i: x000 + use_ema=False, attn_muon_modified, label_smoothing=0.01."""

from typing import cast

import dataclasses
import functools

from experiment import main
from maze.old.x000 import Experiment as Experiment000
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_


class Experiment(Experiment000):
    use_ema: bool = False
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
        rope_kwargs={"base": (10e3, 10e3)},
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
