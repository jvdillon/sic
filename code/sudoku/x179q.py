"""x179q: x177d copycat (K_H=1, K_L=4, z_L_init_svd=True, bs=192, 36k steps)."""

from experiment import main
from model import TRM3, ModelConfigProtocol
from x179 import Experiment as Experiment179


class Experiment(Experiment179):
    batch_size: int = 96 * 2
    total_train_steps: int = 36_000

    config: ModelConfigProtocol = TRM3.Config(
        K_H=1,
        K_L=4,
        carry_H="copy_top1",
        carry_L="all",
        z_L_init_svd=True,
        z_L_random_init=True,
    )


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
