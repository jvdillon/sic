"""x07c: x07a + attn_qk_norm=True."""

from __future__ import annotations

from typing import cast

import dataclasses

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, MLPMixerBlock, SwiGLU, TRM3ConfigProtocol


_config = cast(TRM3.Config, Experiment07a.config)
_block = cast(MLPMixerBlock.Config, _config.block)
_attn = cast(SwiGLU.Config, _block.attn)


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    config: TRM3ConfigProtocol = dataclasses.replace(
        _config,
        block=dataclasses.replace(
            _block,
            attn=dataclasses.replace(_attn, qk_norm=True),
        ),
    )


if __name__ == "__main__":
    main(Experiment())
