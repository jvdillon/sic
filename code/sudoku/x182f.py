"""x182f: x182a with q_halt warmup (force H=1 for first 4k steps)."""

from experiment import main
from model import TRM3, TRM3ConfigProtocol
from x182 import Experiment as Experiment182


class Experiment(Experiment182):
    max_steps_schedule: dict[int, tuple[int, bool]] = {  # noqa: RUF012
        0: (1, False),
        2000: (16, True),
    }
    config: TRM3ConfigProtocol = TRM3.Config(
        K_H=1,
        K_L=4,
        carry_H="copy_top1",
        carry_L="all",
        z_L_init_svd=True,
        z_L_random_init=True,
    )


if __name__ == "__main__":
    main(Experiment())
