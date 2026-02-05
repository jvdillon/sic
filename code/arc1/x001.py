"""x001: Maze experiment wta, K_L=4."""

from typing import cast

import copy

from arc1.x000 import Experiment as Experiment000
from experiment import main
from model import TRM3, TRM3ConfigProtocol


def _config() -> TRM3.Config:
    c = cast("TRM3.Config", copy.copy(Experiment000.config))
    c.K_L = 4
    c.carry_H = "copy_top1"
    return c


class Experiment(Experiment000):
    batch_size: int = Experiment000.batch_size // 4
    K: int = 4
    config: TRM3ConfigProtocol = _config()


if __name__ == "__main__":
    main(Experiment())
