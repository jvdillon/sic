"""x08a: x08 + muon_lr=0.02."""

from experiment import main
from maze.x07 import Experiment as Experiment07


class Experiment(Experiment07):
    label_smoothing: float = 0.1


if __name__ == "__main__":
    main(Experiment())
