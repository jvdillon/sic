"""x07f: x07a"""

from __future__ import annotations

from experiment import main
from maze.x07a import Experiment as Experiment07a


class Experiment(Experiment07a):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())
