"""x08a: x08 + muon_lr=0.02."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07b import Experiment as Experiment07b
from model import TRM3, TRM3ConfigProtocol


class Experiment(Experiment07b):
    augmentation_random_bundle_max_size: int = 8

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07b.config),
        use_rope=True,
        rope_theta=1000.0,
        num_layers=2,
        H_cycles=3,  # 3,
        L_cycles=4,  # 4,
        #
        # anchor_seq_index=1,
        num_puzzle_id_tokens=1,
        num_puzzle_ids=1,
        num_register_tokens=15,
        register_token_init_std=1.0,
        register_tokens_learnable=False,
        q_halt_seq_index=0,
    )


if __name__ == "__main__":
    main(Experiment())
