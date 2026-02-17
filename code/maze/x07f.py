"""x07f: x07 + 2D RoPE."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07 import Experiment as Experiment07
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07.config),
        rope_kwargs={"base": (10e3, 10e3)},
    )


if __name__ == "__main__":
    main(Experiment())
