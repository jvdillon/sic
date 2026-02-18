"""x07j: x07a + aug data, bundle_size=8, anchor_seq_index=1, AC on H-cycles."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"
    augmentation_random_bundle_max_size: int = 8

    K: int = 4
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07a.config),
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
