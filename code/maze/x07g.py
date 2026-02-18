"""x07g: x07a + max_steps_schedule, activation checkpointing on H-cycles."""

from __future__ import annotations

from typing import ClassVar

from experiment import main
from maze.x07b import Experiment as Experiment07b


class Experiment(Experiment07b):
    total_train_steps: int = 8_000
    max_steps_schedule: ClassVar[dict[int, tuple[int, bool]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        0: (1, False),
        1000: (2, False),
        2000: (4, False),
        3000: (6, True),
    }


if __name__ == "__main__":
    main(Experiment())
