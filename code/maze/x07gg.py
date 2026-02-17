"""x07gg: x07b + bundle_size=8, max_steps_schedule, 2D RoPE, activation checkpointing."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast

import dataclasses

from experiment import main
from maze.x07b import Experiment as Experiment07b
from model import TRM3, TRM3ConfigProtocol
from torch.utils.checkpoint import checkpoint


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment07b):
    augmentation_random_bundle_max_size: int = 8

    max_steps_schedule: ClassVar[dict[int, tuple[int, bool]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        0: (1, False),
        1000: (2, False),
        2000: (4, False),
        3000: (6, True),
    }

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment07b.config),
        rope_kwargs={"base": (10e3, 10e3)},
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
