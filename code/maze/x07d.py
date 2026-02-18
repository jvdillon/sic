"""x07d: x07a + use_rope=False, activation checkpointing on H-cycles.

Outcome: Learning fails.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import dataclasses

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3, TRM3ConfigProtocol
from torch.utils.checkpoint import checkpoint


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07a.config),
        use_rope=False,
    )

    def _run_h_cycles(
        self,
        core: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]],
        embeddings: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        for _ in range(self.model.config.H_cycles - 1):
            result = checkpoint(
                core,
                embeddings,
                z_H,
                z_L,
                cos_sin,
                use_reentrant=False,
            )
            assert result is not None
            _logits, _q_halt, z_H, z_L = result
        return core(embeddings, z_H, z_L, cos_sin)


if __name__ == "__main__":
    main(Experiment())
