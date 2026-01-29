"""x182e: x182a with seed=46."""

from experiment import main
from x182a import Experiment as Experiment182a


class Experiment(Experiment182a):
    seed: int = 46


if __name__ == "__main__":
    main(Experiment())
