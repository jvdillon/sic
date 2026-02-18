"""x07c: x07a + attn_qk_norm=True."""

from __future__ import annotations

from typing import cast

import dataclasses
import functools

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, TRM3ConfigProtocol


Config07a = cast(TRM3.Config, Experiment07a.config)
Block07a = cast(functools.partial[TRM3ConfigProtocol], Config07a.block_fn)


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    config: TRM3ConfigProtocol = dataclasses.replace(
        Config07a,
        block_fn=functools.partial(
            Block07a.func,
            **{
                **Block07a.keywords,
                "attn_qk_norm": True,
            },
        ),
    )


if __name__ == "__main__":
    main(Experiment())
