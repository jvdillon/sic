"""x000: ARC experiment baseline."""

# Slow compile?
# Try: rm -rf /tmp/torchinductor_* ~/.triton/cache

import functools

from experiment import (
    main,
    setup_muon_optimizers,
)
from model import (
    TRM3,
    MLPMixerBlock,
    TRM3ConfigProtocol,
    trunc_normal_init_,
)
from sudoku.x182 import Experiment as Experiment182

from data import get_puzzle_config


_CFG = get_puzzle_config("arc")


class Experiment(Experiment182):
    data_dir: str = "/opt/scratch/datasets/arc1concept-aug-1000"
    augment_sudoku: bool = False
    use_puzzle_identifier: bool = True  # Enable puzzle_id conditioning for ARC
    max_puzzle_ids_per_batch: int = 256  # Embedding table size for puzzle_ids
    max_eval_samples: int = 100  # 38_400
    total_train_steps: int = 8_000
    eval_every_steps: int = 500
    batch_size: int = 8  # Reduced further for ARC memory requirements
    eval_batch_size: int | None = 32
    K: int = 1

    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        seq_len=_CFG.seq_len,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=False,
        block_fn=functools.partial(
            MLPMixerBlock,
            seq_len=_CFG.seq_len,
            init_weight_fn=trunc_normal_init_,
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.02,
        )


if __name__ == "__main__":
    main(Experiment())
