"""x182m: TRM baseline but way faster; seed=42."""

from experiment import main
from model import TRM3
from x182 import Experiment as Experiment182


class Experiment(Experiment182):
    total_train_steps: int = 8_000
    batch_size: int = 384
    eval_every_steps: int = 500
    K: int = 1  # Must match K_H * K_L

    config: TRM3.Config = TRM3.Config(
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=True,
        z_L_random_init=False,
    )


if __name__ == "__main__":
    main(Experiment())  # pyright: ignore[reportArgumentType]
