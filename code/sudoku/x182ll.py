"""x182ll: x182j but seed=45"""

from research.projects.trm3.experiment import main
from research.projects.trm3.x182j import Experiment as Experiment182j


class Experiment(Experiment182j):
    seed: int = 45


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
