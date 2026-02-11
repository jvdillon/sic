"""x006a: Rewind algorithm — online theta updates with memoized z.

Forward T ACT steps (no_grad) to collect z trajectory.
Then rewind T→1: at each step t, forward from z[t] through two
model applications to get loss[t+1], so gradients flow through
z[t] → z[t+1] → loss. Update theta after each rewind step.

Each rewind step is a 2-step BPTT window sliding backward in time.
Theta is updated online, so earlier rewind steps benefit from
corrections made at later steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import main
from maze.x000l import Experiment as Experiment000l

import torch


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment000l):
    act_steps: int = 4
    max_steps_schedule: dict[int, tuple[int, bool]] = {0: (1, False)}  # noqa: RUF012

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

        active = state.batch_index >= 0
        if not active.any():
            return {
                "lm": torch.tensor(0.0, device=state.device),
                "n_active": torch.tensor(0.0, device=state.device),
                "n_halted": torch.tensor(0.0, device=state.device),
            }

        # Phase 1: Forward T steps, memoize z trajectory (no grad)
        z_H_traj: list[Tensor] = [state.z_H.detach().clone()]
        z_L_traj: list[Tensor] = [state.z_L.detach().clone()]
        with torch.no_grad():
            for _ in range(self.act_steps):
                result = self._forward(state)
                state.z_H = result["z_H"].detach()
                state.z_L = result["z_L"].detach()
                z_H_traj.append(state.z_H.clone())
                z_L_traj.append(state.z_L.clone())

        # Phase 2: Rewind with 2-step BPTT windows
        # At step t: forward(z[t]) → z[t+1]_new → forward(z[t+1]_new) → loss
        # Gradients flow: loss → z[t+1] → theta (how theta at step t affects future)
        losses = []
        for t in reversed(range(self.act_steps)):
            # Step 1: recompute z[t+1] from memoized z[t] (with grad)
            state.z_H = z_H_traj[t]
            state.z_L = z_L_traj[t]
            result_t = self._forward(state)

            # Step 2: forward from recomputed z[t+1] to get loss
            state.z_H = result_t["z_H"]
            state.z_L = result_t["z_L"]
            result_t1 = self._forward(state)

            active_samples, winner_chains = state.select_winners(result_t1["losses"])
            if len(active_samples) > 0:
                loss = result_t1["losses"][winner_chains].sum()
                (loss / self.batch_size).backward()
                losses.append(loss.detach())
            self._update_weights()

        # Carry final z forward (from end of trajectory)
        state.z_H = z_H_traj[-1]
        state.z_L = z_L_traj[-1]

        # Halt all — fixed ACT steps, no q_halt
        active_samples = state.active.nonzero(as_tuple=True)[0]
        if len(active_samples) > 0:
            for h_step in state.h_step[active_samples].cpu().tolist():
                if h_step > 0:
                    self.act_steps_history.append(h_step)
            state.release(active_samples)
        self._fill_pending()

        avg_loss = (
            torch.stack(losses).mean()
            if losses
            else torch.tensor(0.0, device=state.device)
        )
        return {
            "lm": avg_loss,
            "n_active": torch.tensor(
                float(state.active.sum().item()), device=state.device
            ),
            "n_halted": torch.tensor(float(len(active_samples)), device=state.device),
        }


if __name__ == "__main__":
    main(Experiment())
