"""x07k: x07a + label_smoothing=0.1."""

from __future__ import annotations

from experiment import main
from maze.x07a import Experiment as Experiment07a


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    label_smoothing: float = 0.1


if __name__ == "__main__":
    main(Experiment())
