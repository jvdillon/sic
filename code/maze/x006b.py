"""x006b: True BPTT across ACT steps with activation checkpointing.

Forward T ACT steps without detaching z, wrapping each step's
forward in torch.utils.checkpoint. Accumulate losses across all
steps, then single backward — gradients flow through z across steps.

Memory: O(T) checkpoint boundaries. Compute: ~2x forward (recomputation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x000l import Experiment as Experiment000l
from torch.utils.checkpoint import checkpoint

import torch


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment000l):
    act_steps: int = 4
    max_steps_schedule: dict[int, tuple[int, bool]] = {0: (1, False)}  # noqa: RUF012

    def _act_step(self, z_H: Tensor, z_L: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One ACT step. Returns (loss, new_z_H, new_z_L)."""
        state = self._state
        state.z_H = z_H
        state.z_L = z_L
        result = self._forward(state)
        active_samples, winner_chains = state.select_winners(result["losses"])
        if len(active_samples) > 0:
            loss = result["losses"][winner_chains].sum()
        else:
            loss = z_H.new_zeros((), requires_grad=True)
        return loss, result["z_H"], result["z_L"]

    def step(
        self,
        inputs: Tensor,
        labels: Tensor,
        puzzle_ids: Tensor | None = None,
        valid_count: int | None = None,
    ) -> dict[str, Tensor] | None:
        if self.current_step >= self.total_train_steps:
            return None
        if valid_count is None:
            valid_count = inputs.shape[0]

        state = self._state
        self._enqueue(
            inputs[:valid_count],
            labels[:valid_count],
            puzzle_ids[:valid_count] if puzzle_ids is not None else None,
        )
        self._fill_pending()

        if not (state.batch_index >= 0).any():
            return {
                "lm": torch.tensor(0.0, device=state.device),
                "n_active": torch.tensor(0.0, device=state.device),
                "n_halted": torch.tensor(0.0, device=state.device),
            }

        z_H = state.z_H
        z_L = state.z_L
        total_loss = z_H.new_zeros((), requires_grad=True)

        for t in range(self.act_steps):
            if t < self.act_steps - 1:
                result = checkpoint(
                    self._act_step,
                    z_H,
                    z_L,
                    use_reentrant=False,
                )
                assert result is not None
                loss, z_H, z_L = result
            else:
                loss, z_H, z_L = self._act_step(z_H, z_L)
            total_loss = total_loss + loss

        (total_loss / (self.act_steps * self.batch_size)).backward()
        self._update_weights()

        # Carry z forward (detached)
        state.z_H = z_H.detach()
        state.z_L = z_L.detach()

        # Halt all — fixed ACT steps, no q_halt
        active_samples = state.active.nonzero(as_tuple=True)[0]
        if len(active_samples) > 0:
            for h_step in state.h_step[active_samples].cpu().tolist():
                if h_step > 0:
                    self.act_steps_history.append(h_step)
            state.release(active_samples)
        self._fill_pending()

        return {
            "lm": (total_loss / self.act_steps).detach(),
            "n_active": torch.tensor(
                float(state.active.sum().item()), device=state.device
            ),
            "n_halted": torch.tensor(float(len(active_samples)), device=state.device),
        }


if __name__ == "__main__":
    main(Experiment())
