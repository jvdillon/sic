"""x000f: x000 with attn_muon_modified=True."""

from typing import cast

import dataclasses
import functools

from experiment import (
    Experiment as ExperimentBase,
    main,
    setup_muon_optimizers,
)
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_

from data import get_puzzle_config


_CFG = get_puzzle_config("maze")


class Experiment(ExperimentBase):
    total_train_steps: int = 16_000
    use_ema: bool = False
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 1.0
    cast_model_to_dtype: bool = False
    loss_sum_normalize: bool = True

    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    augment_sudoku: bool = False
    eval_every_steps: int = 500
    batch_size: int = 42
    eval_batch_size: int | None = 256
    K: int = 4
    q_halt_weight: float = 0.05

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, ExperimentBase.config),
        vocab_size=_CFG.vocab_size,
        num_puzzle_grid_tokens=_CFG.grid_len,
        H_cycles=3,
        L_cycles=4,
        K_H=1,
        K_L=4,
        carry_H="copy_top1",
        carry_L="all",
        z_L_init_svd=True,
        use_rope=True,
        num_heads=8,  # Only used for RoPE dim calculation
        num_puzzle_id_tokens=1,
        num_puzzle_ids=1,
        num_register_tokens=15,
        register_token_init_std=0.0,
        register_tokens_learnable=False,
        q_halt_seq_index=0,
        cast_model_to_dtype=False,
        block_fn=functools.partial(
            TransformerBlock,
            multiple_of=128,
            num_heads=8,
            mlp_init_weight_fn=trunc_normal_init_,
            attn_init_weight_fn=trunc_normal_init_,
            attn_muon_modified=True,
            attn_checkpoint_muon_norm=True,
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.001,
            muon_wd=0.02,  # 1e-4/0.005=0.01
        )


if __name__ == "__main__":
    main(Experiment())
