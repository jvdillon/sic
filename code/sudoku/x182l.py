"""x182g: x182a without random init and with q_halt warmup (force H=1 for first 4k steps)."""

from experiment import main
from x182j import Experiment as Experiment182j


class Experiment(Experiment182j):
    seed: int = 44


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
