"""x000d: x000 + grad_accum_steps=4, batch_size=30."""

from experiment import main
from maze.old.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    grad_accum_steps: int = 4
    batch_size: int = 30


if __name__ == "__main__":
    main(Experiment())
