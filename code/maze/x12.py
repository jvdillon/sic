"""x12: x07b + PCGrad across H-cycles.

Motivation: x07b gives gradients through all H-cycles, which helps (+2.7pp
over x07a). x08 shows that gradient surgery (PCGrad) between conflicting
gradient sources helps (+1.9pp). x12 combines both: full-gradient H-cycles
with PCGrad applied at the H-cycle level.

Early H-cycles learn coarse structure; later H-cycles refine. Without PCGrad,
refinement gradients can conflict with and degrade the coarse representation.
PCGrad projects later H-cycle gradients against earlier ones, preventing
later refinement from undoing early structure.

References:
- Yu et al. (2020), "Gradient Surgery for Multi-Task Learning" (PCGrad)
- Combines x07b (AC on H-cycles) with x08 (PCGrad between tasks)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3AC, TRM3ConfigProtocol

import torch
import torch._functorch.config


torch._functorch.config.donated_buffer = False  # noqa: SLF001

if TYPE_CHECKING:
    from experiment import ForwardResult, TrainingState
    from torch import Tensor


class Experiment(Experiment07a):
    collect_h_cycle_intermediates: bool = True

    config: TRM3ConfigProtocol = TRM3AC.Config().update(
        Experiment07a.config,
        block_kwargs_by_layer={
            0: {"checkpoint": True},
        },
    )

    def _backward(
        self,
        forward_result: ForwardResult,
        state: TrainingState,
        active_samples: Tensor,
        winner_chains: Tensor,
        train_q_halt: bool,
    ) -> None:
        hc = self._last_hc_result
        h_logits = list(hc.all_logits) if hc is not None else None
        if h_logits is None or len(h_logits) <= 1:
            super()._backward(
                forward_result,
                state,
                active_samples,
                winner_chains,
                train_q_halt,
            )
            return

        # Compute per-H-cycle CE losses on winner chains.
        chain_labels = state.labels[state.batch_index[winner_chains].clamp(min=0)]
        if self.label_smoothing_includes_pad_token:
            loss_labels = chain_labels.reshape(-1)
            loss_ignore = 0
        else:
            loss_labels = (chain_labels - 1).reshape(-1)
            loss_ignore = -1

        n_prefix = self.config.num_puzzle_id_tokens + self.config.num_register_tokens
        loss_count = (chain_labels != 0).sum(dim=-1).clamp(min=1).float()

        h_losses: list[Tensor] = []
        for lg in h_logits:
            lg_w = lg[winner_chains]
            if n_prefix > 0:
                lg_w = lg_w[:, n_prefix:]
            per_token = torch.nn.functional.cross_entropy(
                lg_w.reshape(-1, lg_w.shape[-1]),
                loss_labels,
                label_smoothing=self.label_smoothing,
                ignore_index=loss_ignore,
                reduction="none",
            ).reshape(len(winner_chains), -1)
            h_losses.append((per_token.sum(dim=-1) / loss_count).sum())

        # Add q_halt loss to the final H-cycle only.
        if train_q_halt:
            with torch.no_grad():
                predictions = self._preds(forward_result["logits"][winner_chains])
                correct = (
                    (predictions == state.labels[active_samples]).all(dim=-1).float()
                )
            h_losses[-1] = h_losses[-1] + self.q_halt_weight * (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    forward_result["q_halt"][winner_chains],
                    correct,
                    reduction="sum",
                )
            )

        # Save any previously accumulated gradients.
        saved_grad: dict[str, Tensor] = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                saved_grad[n] = p.grad.detach().clone()

        # PCGrad across H-cycles: earliest H-cycle is reference.
        ref_grad: dict[str, Tensor] | None = None
        accum_grad: dict[str, Tensor] = {}

        for i, loss_h in enumerate(h_losses):
            self.model.zero_grad(set_to_none=True)
            retain = i < len(h_losses) - 1
            (loss_h / self.batch_size).backward(retain_graph=retain)

            if ref_grad is None:
                ref_grad = {
                    n: p.grad.detach().clone()
                    for n, p in self.model.named_parameters()
                    if p.grad is not None
                }
            else:
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        if p.grad is None or n not in ref_grad:
                            continue
                        g = p.grad
                        r = ref_grad[n]
                        dot = (g * r).sum()
                        if dot < 0:
                            p.grad = g - (dot / (r.norm().square() + 1e-12)) * r

            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    if n in accum_grad:
                        accum_grad[n].add_(p.grad)
                    else:
                        accum_grad[n] = p.grad.detach().clone()

        # Restore saved + accumulated gradients.
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                parts = [
                    t for t in [saved_grad.get(n), accum_grad.get(n)] if t is not None
                ]
                if parts:
                    p.grad = parts[0] if len(parts) == 1 else parts[0] + parts[1]


if __name__ == "__main__":
    main(Experiment())
