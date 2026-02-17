"""x07i: x07b + bundle_size=8, 2D RoPE, anchor_seq_index=1, register_token_init_std=1.0."""

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
        rope_kwargs={"base": (10e3, 10e3)},
        anchor_seq_index=1,
        register_token_init_std=1.0,
    )


if __name__ == "__main__":
    main(Experiment())
