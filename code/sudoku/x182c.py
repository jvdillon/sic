"""x182c: x182a with seed=44."""

from research.projects.trm3.experiment import main
from research.projects.trm3.x182a import Experiment as Experiment182a


class Experiment(Experiment182a):
    seed: int = 44


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
