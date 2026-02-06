"""x000b: x000 except no aug."""

from experiment import main
from maze.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"


if __name__ == "__main__":
    main(Experiment())
