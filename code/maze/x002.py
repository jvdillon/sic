"""x002: x000l + prediction feedback.

Each H-cycle feeds embed(argmax(logits_prev).detach()) back into
the input embeddings, letting the model see and correct its
previous prediction.
"""

from __future__ import annotations

from collections.abc import Callable

from experiment import main
from maze.x000l import Experiment as Experiment000l
from torch import Tensor
from torch.nn import functional

import torch


class Experiment(Experiment000l):
    def _run_h_cycles(
        self,
        core: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]],
        embeddings: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
        cos_sin_detach: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        base_emb = embeddings
        for _ in range(self.model.config.H_cycles - 1):
            with torch.no_grad():
                logits, _q_halt, z_H, z_L = core(
                    embeddings.detach(),
                    z_H.detach(),
                    z_L.detach(),
                    cos_sin_detach,
                )
                pred_emb = self.model.embed_scale * self.model.embed_tokens(
                    logits[:, n_prefix:].argmax(dim=-1),
                    cfg.dtype,
                )
                embeddings = base_emb + functional.pad(pred_emb, (0, 0, n_prefix, 0))
        return core(embeddings, z_H, z_L, cos_sin)


if __name__ == "__main__":
    main(Experiment())
