"""x000: Maze experiment baseline.

Uses TransformerBlock with RoPE (matching TRM paper config for maze).
TRM maze config: H_cycles=3, L_cycles=4, L_layers=2, attention+RoPE.
"""

import functools

from experiment import main, setup_muon_optimizers
from model import TRM3, TransformerBlock, TRM3ConfigProtocol
from sudoku.x182 import Experiment as Experiment182

from data import get_puzzle_config


_CFG = get_puzzle_config("maze")


class Experiment(Experiment182):
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"
    augment_sudoku: bool = False
    total_train_steps: int = 8_000
    eval_every_steps: int = 500
    batch_size: int = 64  # Reduced due to 11x longer seq_len (901 vs 82)
    eval_batch_size: int | None = 256
    K: int = 1
    q_halt_weight: float = 0.05

    # TRM paper maze config: H_cycles=3, L_cycles=4, attention+RoPE
    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        num_puzzle_grid_tokens=_CFG.num_puzzle_grid_tokens,
        H_cycles=3,
        L_cycles=6,  # Or should it be 4?
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=False,
        use_rope=True,
        num_heads=8,
        block_fn=functools.partial(
            TransformerBlock,
            num_heads=8,
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.02 / 4,
        )


if __name__ == "__main__":
    main(Experiment())
