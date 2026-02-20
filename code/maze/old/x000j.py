"""x000j: x000l + batch_size=176, muon_lr=0.010."""

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
    use_ema: bool = False
    label_smoothing: float = 0.01
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
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
