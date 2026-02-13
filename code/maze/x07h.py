"""x08a: x08 + muon_lr=0.02."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07.config),
        use_rope=True,
        rope_2d_grid_shape=(30, 30),
        num_layers=2,
        H_cycles=3,  # 3,
        L_cycles=4,  # 4,
    )


if __name__ == "__main__":
    main(Experiment())
