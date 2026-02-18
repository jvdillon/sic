"""x000: ARC experiment baseline.

Uses TransformerBlock with RoPE (matching TRM paper config for ARC).
TRM ARC config: H_cycles=3, L_cycles=4, L_layers=2, attention+RoPE.
"""

# Slow compile?
# Try: rm -rf /tmp/torchinductor_* ~/.triton/cache

import functools

from experiment import (
    Experiment as Experiment182,
    main,
    setup_muon_optimizers,
)
from model import (
    TRM3,
    TransformerBlock,
    TRM3ConfigProtocol,
)

from data import get_puzzle_config


_CFG = get_puzzle_config("arc")


class Experiment(Experiment182):
    data_dir: str = "/opt/scratch/datasets/arc1concept-aug-1000"
    augment_sudoku: bool = False
    use_additive_puzzle_emb: bool = (
        False  # Disabled; using TRM-style puzzle_id_tokens instead
    )
    max_puzzle_ids_per_batch: int = 256  # Only used if use_additive_puzzle_emb=True
    max_eval_samples: int = 100  # 38_400
    total_train_steps: int = 8_000
    eval_every_steps: int = 500
    batch_size: int = 8  # Reduced further for ARC memory requirements
    eval_batch_size: int | None = 32
    K: int = 1

    # TRM paper ARC config: H_cycles=3, L_cycles=4, attention+RoPE
    # TRM-style prefix: 1 puzzle_id token + 15 register tokens (zeros)
    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        puzzle_grid_shape=_CFG.grid_shape,
        H_cycles=3,
        L_cycles=4,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=False,
        use_rope=True,
        num_heads=8,  # Must match block_fn num_heads (for RoPE dim)
        # NOTE: Prefix tokens disabled because ExperimentBase eval methods don't pass puzzle_ids.
        num_puzzle_id_tokens=0,  # TRM-style: disabled until eval supports puzzle_ids
        num_register_tokens=0,  # TRM-style: disabled until eval supports puzzle_ids
        num_puzzle_ids=256,  # Embedding table size (unused when num_puzzle_id_tokens=0)
        block_fn=functools.partial(
            TransformerBlock,
            num_heads=8,
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.02,
        )


if __name__ == "__main__":
    main(Experiment())
