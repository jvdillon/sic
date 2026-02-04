"""x000: Maze experiment based on x182 sparse chain pooling."""

import functools

from experiment import main
from model import (
    TRM3,
    MLPMixerBlock,
    TRM3ConfigProtocol,
    trunc_normal_init_,
)
from sudoku.x182 import Experiment as Experiment182

from data import get_puzzle_config


_CFG = get_puzzle_config("maze")


class Experiment(Experiment182):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    augment_sudoku: bool = False
    total_train_steps: int = 8_000
    eval_every_steps: int = 500
    batch_size: int = 16  # Reduced due to 11x longer seq_len (901 vs 82)
    eval_batch_size: int | None = 64
    K: int = 1

    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        seq_len=_CFG.seq_len,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=True,
        block_fn=functools.partial(
            MLPMixerBlock,
            seq_len=_CFG.seq_len,
            init_weight_fn=trunc_normal_init_,
        ),
    )


if __name__ == "__main__":
    main(Experiment())
