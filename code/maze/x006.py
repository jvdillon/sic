"""x006: Forward ACT + rewind carrying adjoint of final loss.

Two-phase training per sample:

Forward phase (like BP through unrolled skip connections):
  Each ACT step independently minimizes its own loss. Theta gets
  gradient directly from each step's loss; z is detached between
  steps. Memoize z at each step boundary.

Rewind phase (like BP through unrolled non-skip connections):
  After q_halt, propagate the final loss's adjoint (grad_loss_at_halt)
  backward through the memoized z chain. At each rewind step t,
  forward from z[t] with current theta, backward with the carried
  adjoint, update theta. The adjoint naturally decays through the
  chain — early steps get weak signal, which is correct since the
  forward phase already handled them directly.

Staleness: theta is updated between rewind steps, so the adjoint
drifts from what it would be under a single fixed theta. This is
the core approximation — the bet is that the drift is small relative
to the useful gradient signal.

Note: rewind steps consume optimizer steps from total_train_steps.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import dataclasses

from maze.x000l import Experiment as Experiment000l
from model import TRM3, TRM3ConfigProtocol
from torch import nn
from torch.utils.checkpoint import checkpoint

import torch


if TYPE_CHECKING:
    from torch import Tensor


class Experiment(Experiment000l):
    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000l.config),
        max_num_compile_core=4,
    )

    def __init__(self) -> None:
        super().__init__()
        self._z_traj: dict[int, list[tuple[Tensor, Tensor]]] = defaultdict(list)
        self._traj_tokens: dict[int, Tensor] = {}

    def reset_transient_state(self) -> None:
        super().reset_transient_state()
        self._z_traj.clear()
        self._traj_tokens.clear()

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

    def _core_forward(
        self,
        tokens: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Model forward: tokens + z_in -> (logits_grid, z_H_out, z_L_out).

        logits are sliced to grid-only (prefix removed).
        """
        cfg = self.config
        m = self.model
        assert self.device is not None
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            emb = m.embed_scale * m.embed_tokens(tokens, cfg.dtype)
            puzzle_ids = torch.zeros(
                tokens.shape[0], device=tokens.device, dtype=torch.long
            )
            emb = m._prepend_prefix(emb, puzzle_ids)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            core = m.core_compiled if cfg.compile_core else m.core
            cos_sin = m._get_cos_sin(emb.device)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            logits, _q_halt, z_H_out, z_L_out = self._run_h_cycles(
                core,
                emb,
                z_H,
                z_L,
                cos_sin,
            )
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        return logits[:, n_prefix:], z_H_out, z_L_out

    def _do_rewind(
        self,
        halt_batch_indices: list[int],
    ) -> None:
        """Rewind phase for halted samples."""
        trajs = [self._z_traj[i] for i in halt_batch_indices]
        tokens = torch.stack([self._traj_tokens[i] for i in halt_batch_indices])
        T = min(len(t) for t in trajs)
        if T < 2:
            return

        # Seed adjoint: forward from z[T-2], compute loss, backward
        z_H_in = torch.stack([trajs[j][T - 2][0] for j in range(len(trajs))])
        z_L_in = torch.stack([trajs[j][T - 2][1] for j in range(len(trajs))])
        z_H_in = z_H_in.detach().requires_grad_(True)
        z_L_in = z_L_in.detach().requires_grad_(True)

        state = self._state
        labels = torch.stack([state.labels[i] for i in halt_batch_indices])
        logits, _z_H_out, _z_L_out = self._core_forward(tokens, z_H_in, z_L_in)
        assert self.device is not None
        B = len(halt_batch_indices)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            if self.label_smoothing_includes_pad_token:
                _loss_labels = labels.reshape(-1)
                _loss_ignore = 0
            else:
                _loss_labels = (labels - 1).reshape(-1)
                _loss_ignore = -1
            loss_per_token = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                _loss_labels,
                label_smoothing=self.label_smoothing,
                ignore_index=_loss_ignore,
                reduction="none",
            ).reshape(B, -1)
            loss_count = (labels != 0).sum(dim=-1).clamp(min=1)
            loss = (loss_per_token.sum(dim=-1) / loss_count).sum() / self.batch_size
        loss.backward()

        assert z_H_in.grad is not None
        assert z_L_in.grad is not None
        grad_H = z_H_in.grad.detach()
        grad_L = z_L_in.grad.detach()
        # Seed only extracts the adjoint; forward phase already updated
        # theta for this step. Zero grads to avoid double-counting.
        self.model.zero_grad(set_to_none=True)

        # Carry adjoint backward through remaining steps
        for t in reversed(range(T - 2)):
            z_H_in = torch.stack([trajs[j][t][0] for j in range(len(trajs))])
            z_L_in = torch.stack([trajs[j][t][1] for j in range(len(trajs))])
            z_H_in = z_H_in.detach().requires_grad_(True)
            z_L_in = z_L_in.detach().requires_grad_(True)

            _logits, z_H_out, z_L_out = self._core_forward(tokens, z_H_in, z_L_in)
            torch.autograd.backward([z_H_out, z_L_out], [grad_H, grad_L])

            assert z_H_in.grad is not None
            assert z_L_in.grad is not None
            grad_H = z_H_in.grad.detach()
            grad_L = z_L_in.grad.detach()
            self._update_weights()

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
        if self.augment_sudoku:
            from experiment import augment_sudoku  # noqa: PLC0415

            inputs, labels = augment_sudoku(inputs, labels)

        valid_keys = [k for k in self.max_steps_schedule if k <= self.current_step]
        max_h_steps, train_q_halt = self.max_steps_schedule[max(valid_keys)]
        state = self._state

        self._enqueue(
            inputs[:valid_count],
            labels[:valid_count],
            puzzle_ids[:valid_count] if puzzle_ids is not None else None,
        )
        self._fill_pending()

        # Snapshot z for active chains before forward
        active_mask = state.batch_index >= 0
        if active_mask.any():
            for c in active_mask.nonzero(as_tuple=True)[0].tolist():
                bi = int(state.batch_index[c].item())
                self._z_traj[bi].append(
                    (
                        state.z_H[c].detach().clone(),
                        state.z_L[c].detach().clone(),
                    )
                )
                if bi not in self._traj_tokens:
                    self._traj_tokens[bi] = state.inputs[bi].clone()

        # --- Standard forward ACT step ---
        forward_result = self._forward(state)
        active_at_forward = state.batch_index >= 0
        active_samples, winner_chains = state.select_winners(forward_result["losses"])

        if len(active_samples) > 0:
            assert self.device is not None
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                loss = forward_result["losses"][winner_chains].sum()
                if train_q_halt:
                    with torch.no_grad():
                        predictions = forward_result["logits"][winner_chains].argmax(
                            dim=-1
                        )
                        correct = (
                            (predictions == state.labels[active_samples])
                            .all(dim=-1)
                            .float()
                        )
                    loss = (
                        loss
                        + self.q_halt_weight
                        * nn.functional.binary_cross_entropy_with_logits(
                            forward_result["q_halt"][winner_chains],
                            correct,
                            reduction="sum",
                        )
                    )
                (loss / self.batch_size).backward()

        self._update_weights()

        # Carry z
        if len(active_samples) > 0:
            chains = state.chain_indices[active_samples]
            valid = chains >= 0
            valid_chains = chains[valid]
            state.z_H[valid_chains] = forward_result["z_H"][valid_chains].detach()
            state.z_L[valid_chains] = forward_result["z_L"][valid_chains].detach()
            state.h_step[active_samples] += 1

        # --- Halt detection ---
        num_halted = 0
        if len(active_samples) > 0:
            h_steps = state.h_step[active_samples]
            at_max = h_steps >= max_h_steps
            if train_q_halt:
                q_halt_positive = forward_result["q_halt"][winner_chains] > 0
                force_continue = (
                    torch.rand(len(h_steps), device=state.device)
                    < self.halt_exploration_prob
                )
                min_random_steps = torch.randint(
                    low=2,
                    high=max(3, max_h_steps + 1),
                    size=(len(h_steps),),
                    device=state.device,
                )
                halt = (at_max | q_halt_positive) & (
                    ~force_continue | (h_steps >= min_random_steps)
                )
            else:
                halt = at_max
            if halt.any():
                halt_indices = active_samples[halt]
                num_halted = int(halt.sum().item())
                for h_step in state.h_step[halt_indices].cpu().tolist():
                    if h_step > 0:
                        self.act_steps_history.append(h_step)
                        if 1 <= h_step <= len(self.halt_steps_histogram):
                            self.halt_steps_histogram[h_step - 1] += 1

                # Append final z to trajectories
                halt_winner = winner_chains[halt]
                halt_batch_list = halt_indices.tolist()
                for bi, c in zip(halt_batch_list, halt_winner.tolist(), strict=False):
                    self._z_traj[bi].append(
                        (
                            forward_result["z_H"][c].detach().clone(),
                            forward_result["z_L"][c].detach().clone(),
                        )
                    )

                # Rewind
                self._do_rewind(halt_batch_list)

                # Clean up trajectories
                for bi in halt_batch_list:
                    self._z_traj.pop(bi, None)
                    self._traj_tokens.pop(bi, None)

                state.release(halt_indices)

        num_expunged = state.expunge(
            forward_result["losses"].detach(),
            self.expunge_threshold,
            self.min_expunge_step,
        )
        self._fill_pending()

        num_active = int(state.active.sum().item())
        return {
            "lm": (
                forward_result["losses"][active_at_forward].mean().detach()
                if active_at_forward.any()
                else torch.tensor(0.0, device=state.device)
            ),
            "n_active": torch.tensor(
                float(num_active * self.K),
                device=state.device,
            ),
            "n_expunged": torch.tensor(float(num_expunged), device=state.device),
            "n_halted": torch.tensor(float(num_halted), device=state.device),
            "pool_util": torch.tensor(
                num_active * self.K / state.num_chains,
                device=state.device,
            ),
        }


if __name__ == "__main__":
    import pathlib

    exp = Experiment()
    exp.setup_optimizers()
    _ckpt_path = pathlib.Path(__file__).parent / "ckpts" / "x000l.1" / "best.pt"
    _ckpt = torch.load(_ckpt_path, weights_only=False)
    exp.model.load_state_dict(_ckpt["model"])
    print(f"Loaded model weights from {_ckpt_path}")
    from experiment import train

    train(exp)
