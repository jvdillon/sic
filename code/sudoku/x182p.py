"""x182p: TRM baseline but way faster; seed=45."""

from experiment import main
from x182m import Experiment as Experiment182m


class Experiment(Experiment182m):
    seed: int = 45


if __name__ == "__main__":
    main(Experiment())
