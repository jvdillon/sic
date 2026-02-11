"""x002b: x002a with batch_size=176."""

from __future__ import annotations

from experiment import main
from maze.x002a import Experiment as Experiment002a


class Experiment(Experiment002a):
    batch_size: int = 176


if __name__ == "__main__":
    main(Experiment())
