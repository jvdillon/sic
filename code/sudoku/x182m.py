"""x182m: x182j with seed=45."""

from experiment import main
from x182j import Experiment as Experiment182j


class Experiment(Experiment182j):
    seed: int = 45


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
