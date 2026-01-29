"""x179e: x179 except K_H=4 and carry_L="top1"."""

from experiment import main
from model import TRM3, ModelConfigProtocol

from .x179 import Experiment as Experiment179


class Experiment(Experiment179):
    config: ModelConfigProtocol = TRM3.Config(
        K_H=4,
        carry_H="top1",
        carry_L="top1",
    )


if __name__ == "__main__":
    main(Experiment())
