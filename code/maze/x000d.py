"""x000c: 125 groups × 8 augmentations = 1000 patterns (same as noaug)."""

from experiment import main
from maze.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    grad_accum_steps: int = 4
    batch_size: int = 30


if __name__ == "__main__":
    main(Experiment())
