"""x07l: x07a + rope_kwargs=1e3 (scalar, overrides 2D RoPE)."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07a.config),
        rope_kwargs={"base": 1e3},
    )


if __name__ == "__main__":
    main(Experiment())
