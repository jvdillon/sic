"""x182i: x182g with seed=43."""

from experiment import main
from x182g import Experiment as Experiment182g


class Experiment(Experiment182g):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
