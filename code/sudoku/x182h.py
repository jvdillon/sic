"""x182h: x182f with seed=43."""

from research.projects.trm3.experiment import main
from research.projects.trm3.x182f import Experiment as Experiment182f


class Experiment(Experiment182f):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
