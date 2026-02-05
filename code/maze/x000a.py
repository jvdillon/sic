"""x000a: x000 except larger lr."""

from experiment import main
from maze.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    batch_size: int = 120


if __name__ == "__main__":
    main(Experiment())
