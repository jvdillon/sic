"""x000h: x000 with 2D RoPE for the 30x30 maze grid."""

from typing import cast

import dataclasses

from experiment import main
from maze.x000 import Experiment as Experiment000
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment000):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
        rope_2d_grid_shape=(30, 30),
    )


if __name__ == "__main__":
    main(Experiment())
