"""x08a: x08 + muon_lr=0.02."""

from experiment import main
from maze.x07 import Experiment as Experiment07


class Experiment(Experiment07):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"


if __name__ == "__main__":
    main(Experiment())
