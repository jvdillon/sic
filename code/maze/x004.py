"""x004: x000l with full backprop through all H-cycles.

Baseline uses no_grad + detach for intermediate H-cycles. This experiment
removes that truncation so gradients flow through the entire H-cycle chain.
Uses activation checkpointing per H-cycle to avoid OOM.
Isolates whether H-cycle gradient truncation limits performance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from experiment import main
from maze.x000l import Experiment as Experiment000l
from torch.utils.checkpoint import checkpoint


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment000l):
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
