"""x07n: x07 + K=4, K_L=4, carry_H=copy_top1, z_L_init_svd, 2D RoPE, layer checkpointing."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07):
    K: int = 4
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07.config),
        K_L=4,
        carry_H="copy_top1",
        z_L_init_svd=True,
        # z_L_random_init=True,
        rope_kwargs={"base": (10e3, 10e3)},
        block_kwargs_by_layer={
            0: {"checkpoint": True},
            1: {"checkpoint": True},
        },
    )


if __name__ == "__main__":
    main(Experiment())
