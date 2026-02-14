"""x07: x000l (standalone, no x000 inheritance)."""

from typing import Literal

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
    batch_size: int = 128

    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k"
    augment_sudoku: bool = False
    eval_every_steps: int = 500
    eval_batch_size: int | None = 256
    K: int = 1
    q_halt_weight: float = 0.05
    total_train_steps: int = 16_000
    use_ema: bool = False
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 1.0
    cast_model_to_dtype: bool = False
    loss_sum_normalize: bool = True
    eval_method: Literal["standard", "fast", "wta"] = "fast"
    eval_path_valid: bool = True
    label_smoothing_includes_pad_token: bool = False

    config: TRM3ConfigProtocol = TRM3.Config(
        vocab_size=_CFG.vocab_size,
        num_puzzle_grid_tokens=_CFG.grid_len,
        num_layers=2,
        H_cycles=3,
        L_cycles=4,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        z_L_init_svd=False,
        use_rope=True,
        num_heads=8,
        num_puzzle_id_tokens=1,
        num_puzzle_ids=1,
        num_register_tokens=15,
        register_token_init_std=0.0,
        register_tokens_learnable=False,
        q_halt_seq_index=0,
        cast_model_to_dtype=False,
        label_smoothing_includes_pad_token=False,
        block_fn=functools.partial(
            TransformerBlock,
            multiple_of=128,
            num_heads=8,
            mlp_init_weight_fn=trunc_normal_init_,
            mlp_muon_modified=True,
            attn_init_weight_fn=trunc_normal_init_,
            attn_muon_modified=True,
            attn_checkpoint_muon_norm=True,
            attn_qk_norm=True,
        ),
    )

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
