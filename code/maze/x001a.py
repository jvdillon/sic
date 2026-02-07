"""x001a: x001 except noaug."""

from experiment import main
from maze.x001 import Experiment as Experiment001


class Experiment(Experiment001):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"


if __name__ == "__main__":
    main(Experiment())
