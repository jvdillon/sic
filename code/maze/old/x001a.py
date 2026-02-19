"""x001a: x001 except noaug."""

from experiment import main
from maze.old.x001 import Experiment as Experiment001


class Experiment(Experiment001):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    grad_accum_steps: int = 4
    batch_size: int = 30


if __name__ == "__main__":
    main(Experiment())
