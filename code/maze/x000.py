"""x000: Maze experiment baseline.

Uses TransformerBlock with RoPE (matching TRM paper config for maze).
TRM maze config: H_cycles=3, L_cycles=4, L_layers=2, attention+RoPE.
"""

import functools

from experiment import main, setup_muon_optimizers
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_
from sudoku.x182 import Experiment as Experiment182

from data import get_puzzle_config


_CFG = get_puzzle_config("maze")


class Experiment(Experiment182):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    augment_sudoku: bool = False
    total_train_steps: int = 8_000
    eval_every_steps: int = 500
    batch_size: int = 128
    eval_batch_size: int | None = 256
    K: int = 1
    q_halt_weight: float = 0.05

    # TRM paper maze config: H_cycles=3, L_cycles=4, attention+RoPE
    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        num_puzzle_grid_tokens=_CFG.grid_len,
        H_cycles=3,
        L_cycles=6,  # Or should it be 4?
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=False,
        use_rope=True,
        num_heads=8,  # Only used for RoPE dim calculation
        block_fn=functools.partial(
            TransformerBlock,
            multiple_of=128,
            num_heads=8,
            mlp_init_weight_fn=trunc_normal_init_,
            attn_init_weight_fn=trunc_normal_init_,
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
