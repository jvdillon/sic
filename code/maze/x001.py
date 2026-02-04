"""x001: Maze experiment wta, K_L=4."""

from typing import cast

import copy

from experiment import main
from maze.x000 import Experiment as Experiment000
from model import TRM3, TRM3ConfigProtocol


def _config() -> TRM3.Config:
    c = cast(TRM3.Config, copy.copy(Experiment000.config))
    c.K_L = 4
    c.carry_H = "copy_top1"
    return c


class Experiment(Experiment000):
    batch_size: int = 64 // 4  # Reduced due to 11x longer seq_len (901 vs 82)
    K: int = 4
    config: TRM3ConfigProtocol = _config()


if __name__ == "__main__":
    main(Experiment())
