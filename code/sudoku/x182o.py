"""x182o: TRM baseline but way faster; seed=44."""

from experiment import main
from x182m import Experiment as Experiment182m


class Experiment(Experiment182m):
    seed: int = 44


if __name__ == "__main__":
    main(Experiment())
