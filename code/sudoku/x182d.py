"""x182c: x182a with seed=44."""

from experiment import main
from x182a import Experiment as Experiment182a


class Experiment(Experiment182a):
    seed: int = 45


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
