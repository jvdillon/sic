"""x07i: x07b + aug data, bundle_size=8, anchor_seq_index=1, AC on H-cycles."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07b import Experiment as Experiment07b
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07b):
    total_train_steps: int = 8_000
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"
    augmentation_random_bundle_max_size: int = 8

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07b.config),
        anchor_seq_index=1,
    )


if __name__ == "__main__":
    main(Experiment())
