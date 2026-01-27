"""x182h: x182f with seed=43."""

from experiment import main
from x182f import Experiment as Experiment182f


class Experiment(Experiment182f):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
