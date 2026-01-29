"""x182n: TRM baseline but way faster; seed=43."""

from experiment import main
from x182m import Experiment as Experiment182m


class Experiment(Experiment182m):
    seed: int = 43


if __name__ == "__main__":
    main(Experiment())
