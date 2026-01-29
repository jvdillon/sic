"""x179g: x179 except K_H=4, carry_H="all", carry_L="all"."""

from experiment import main
from model import TRM3, ModelConfigProtocol

from .x179 import Experiment as Experiment179


class Experiment(Experiment179):
    config: ModelConfigProtocol = TRM3.Config(
        K_H=4,
        carry_H="all",
        carry_L="all",
    )


if __name__ == "__main__":
    main(Experiment())
