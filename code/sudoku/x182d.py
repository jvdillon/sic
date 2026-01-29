"""x182d: x182a with seed=45."""

from experiment import main
from x182a import Experiment as Experiment182a


class Experiment(Experiment182a):
    seed: int = 45


if __name__ == "__main__":
    main(Experiment())
