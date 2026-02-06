"""x000c: 125 groups × 8 augmentations = 1000 patterns (same as noaug)."""

from experiment import main
from maze.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    num_instances: int | None = 125
    stratified: bool = False


if __name__ == "__main__":
    main(Experiment())
