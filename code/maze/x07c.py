"""x07c: x07 + 2D RoPE, register_token_init_std=1.0, attn_qk_norm=True."""

from __future__ import annotations

from typing import cast

import dataclasses
import functools

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TRM3ConfigProtocol


Config07 = cast(TRM3.Config, Experiment07.config)
Block07 = cast(functools.partial[TRM3ConfigProtocol], Config07.block_fn)


class Experiment(Experiment07):
    config: TRM3ConfigProtocol = dataclasses.replace(
        Config07,
        rope_kwargs={"base": (10e3, 10e3)},
        register_token_init_std=1.0,
        block_fn=functools.partial(
            Block07.func,
            **{
                **Block07.keywords,
                "attn_qk_norm": True,
            },
        ),
    )


if __name__ == "__main__":
    main(Experiment())
