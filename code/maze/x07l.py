"""x08a: x08 + muon_lr=0.02."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07.config),
        rope_kwargs={"base": 1e3},
        register_token_init_std=1.0,
    )


if __name__ == "__main__":
    main(Experiment())
