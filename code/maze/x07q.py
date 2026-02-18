"""x07q: x07a + num_layers=3, AC on layer 0."""

from typing import cast

import dataclasses

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07a):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07a.config),
        block_kwargs_by_layer={0: {"checkpoint": True}},
        num_layers=3,
    )


if __name__ == "__main__":
    main(Experiment())
