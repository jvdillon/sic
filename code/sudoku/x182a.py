"""x182a: x182 with random z_L init."""

from research.projects.trm3.experiment import main
from research.projects.trm3.model import TRM3
from research.projects.trm3.x182 import Experiment as Experiment182


class Experiment(Experiment182):
    config: TRM3.Config = TRM3.Config(
        K_H=1,
        K_L=4,
        carry_H="copy_top1",
        carry_L="all",
        z_L_init_svd=True,
        z_L_random_init=True,
    )


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
