"""x000q: x000l with core_damping=0.5 for iterative stability."""

from typing import cast

import dataclasses

from experiment import main, setup_muon_optimizers
from maze.old.x000 import Experiment as Experiment000
from model import (
    TRM3,
    Attention,
    SwiGLU,
    TransformerBlock,
    TRM3ConfigProtocol,
    trunc_normal_init_,
)


class Experiment(Experiment000):
    total_train_steps: int = 16_000
    use_ema: bool = False
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 1.0
    cast_model_to_dtype: bool = False
    loss_sum_normalize: bool = True

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
        vocab_size=6,
        L_cycles=4,
        num_puzzle_id_tokens=1,
        num_puzzle_ids=1,
        num_register_tokens=15,
        register_token_init_std=0.0,
        register_tokens_learnable=False,
        q_halt_seq_index=0,
        cast_model_to_dtype=False,
        core_damping=0.5,
        block=TransformerBlock.Config(
            attn=Attention.Config(
                num_heads=8,
                muon_modified=True,
                checkpoint_muon_norm=True,
                init_weight_fn=trunc_normal_init_,
            ),
            ffn=SwiGLU.Config(
                multiple_of=128,
                init_weight_fn=trunc_normal_init_,
            ),
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
