"""x09: x07 + num_layers=3, AC on layer 0, register_token_init_std=1.0."""

from typing import cast

import dataclasses

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TRM3ConfigProtocol

from data import get_puzzle_config


_CFG = get_puzzle_config("maze")


class Experiment(Experiment07):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07.config),
        block_kwargs_by_layer={0: {"checkpoint": True}},
        num_layers=3,
        register_token_init_std=1.0,
        num_register_tokens=15,
        z_L_init_svd=False,
    )


if __name__ == "__main__":
    main(Experiment())
