"""x003: x002a + teacher-forced feedback.

During training, randomly replaces the model's own argmax predictions
with correct labels before feeding back (prob=feedback_teacher_prob).
Prevents the model from learning a self-consistent fixed point.
At eval, uses model's own predictions (labels=None path).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from experiment import ForwardResult, HCycleResult, main
from maze.x002 import Experiment as Experiment002
from torch import Tensor
from torch.nn import functional

import torch


class Experiment(Experiment002):
    _train_labels: Tensor | None = None
    feedback_preserves_emb_grad: bool = True
    feedback_teacher_prob: float = 0.5

    def _pred_feedback(
        self,
        base_emb: Tensor,
        logits: Tensor,
        n_prefix: int,
        labels: Tensor | None = None,
    ) -> Tensor:
        """Feedback with optional teacher forcing."""
        cfg = self.config
        preds = logits[:, n_prefix:].argmax(dim=-1)

        if labels is not None and self.model.training:
            mask = (
                torch.rand_like(preds, dtype=torch.float32) < self.feedback_teacher_prob
            )
            feedback_ids = torch.where(mask, labels, preds)
        else:
            feedback_ids = preds

        pred_emb = self.model.embed_scale * self.model.embed_tokens(
            feedback_ids,
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
    ) -> HCycleResult:
        n_prefix = self.config.num_puzzle_id_tokens + self.config.num_register_tokens
        labels = getattr(self, "_train_labels", None)
        base_emb = embeddings
        for _ in range(self.model.config.H_cycles - 1):
            with torch.no_grad():
                logits, _q_halt, z_H, z_L = core(
                    embeddings.detach(),
                    z_H.detach(),
                    z_L.detach(),
                    cos_sin,
                )
            embeddings = self._pred_feedback(base_emb, logits, n_prefix, labels)
        z_H_pre, z_L_pre = z_H, z_L
        logits, q_halt, z_H, z_L = core(embeddings, z_H, z_L, cos_sin)
        return HCycleResult(logits, q_halt, z_H, z_L, z_H_pre, z_L_pre)

    def _forward(self, state: Any) -> ForwardResult:
        batch_indices = state.batch_index.clamp(min=0)
        self._train_labels = state.labels[batch_indices]
        try:
            return super()._forward(state)
        finally:
            self._train_labels = None


if __name__ == "__main__":
    main(Experiment())
