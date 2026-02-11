"""x002: x000l + prediction feedback.

Each H-cycle feeds embed(argmax(logits_prev).detach()) back into
the input embeddings, letting the model see and correct its
previous prediction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from experiment import main
from maze.x000l import Experiment as Experiment000l
from torch import Tensor
from torch.nn import functional

import torch


class Experiment(Experiment000l):
    def _pred_feedback(
        self,
        base_emb: Tensor,
        logits: Tensor,
        n_prefix: int,
    ) -> Tensor:
        """Add embed(argmax(logits)) to base_emb, padding prefix positions."""
        cfg = self.config
        pred_emb = self.model.embed_scale * self.model.embed_tokens(
            logits[:, n_prefix:].argmax(dim=-1),
            cfg.dtype,
        )
        return base_emb + functional.pad(pred_emb, (0, 0, n_prefix, 0))

    def _run_h_cycles(
        self,
        core: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]],
        embeddings: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
        cos_sin_detach: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n_prefix = self.config.num_puzzle_id_tokens + self.config.num_register_tokens
        base_emb = embeddings
        for _ in range(self.model.config.H_cycles - 1):
            with torch.no_grad():
                logits, _q_halt, z_H, z_L = core(
                    embeddings.detach(),
                    z_H.detach(),
                    z_L.detach(),
                    cos_sin_detach,
                )
                embeddings = self._pred_feedback(base_emb, logits, n_prefix)
        return core(embeddings, z_H, z_L, cos_sin)

    def _eval_forward(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        """Eval forward with prediction feedback between H-cycles."""
        m = self.model
        cfg = m.config
        input_emb = m.embed_scale * m.embed_tokens(input_ids, cfg.dtype)
        input_emb = m._prepend_prefix(input_emb, puzzle_ids)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        cos_sin = None if m.rope is None else m.rope()
        core = m.core_compiled if cfg.compile_core else m.core
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        base_emb = input_emb
        embeddings = input_emb
        for _ in range(cfg.H_cycles - 1):
            logits, _q_halt, z_H, z_L = core(embeddings, z_H, z_L, cos_sin)
            embeddings = self._pred_feedback(base_emb, logits, n_prefix)
        logits, q_halt, z_H, z_L = core(embeddings, z_H, z_L, cos_sin)
        if n_prefix > 0:
            logits = logits[:, n_prefix:]
        return {
            "logits": logits,
            "q_halt": q_halt,
            "z_H": z_H.detach(),
            "z_L": z_L.detach(),
        }


if __name__ == "__main__":
    main(Experiment())
