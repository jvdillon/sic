"""x182b: x182a with seed=43."""

from experiment import main
from x182a import Experiment as Experiment182a


class Experiment(Experiment182a):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
