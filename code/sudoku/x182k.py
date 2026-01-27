"""x182k: x182j with seed=43."""

from experiment import main
from x182j import Experiment as Experiment182j


class Experiment(Experiment182j):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
