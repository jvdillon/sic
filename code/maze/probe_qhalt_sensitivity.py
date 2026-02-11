# pyright: reportPrivateUsage=false
"""Probe: measure q_halt logit sensitivity to a single optimizer step.

For each of N batches:
  1. Forward pass on batch → record q_halt logits (before)
  2. Full training step (forward + backward + optimizer.step)
  3. Forward same batch again → record q_halt logits (after)
  4. Compute delta, sign flips, etc.

This measures how much a single weight update destabilizes the halt decision.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv --quiet --project ~/projects/sic run python \
        maze/probe_qhalt_sensitivity.py --exp x000m --step 8500
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import resume_from_checkpoint


def stats(xs: list[float]) -> tuple[float, float, float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, math.sqrt(v), min(xs), max(xs)


def run_probe(exp_name: str, step: int, n_trials: int = 100, warmup: int = 30):
    ckpt_path = f"maze/ckpts/{exp_name}/step{step:05d}.pt"
    print(f"Loading {ckpt_path}...")

    if exp_name.startswith("x000m"):
        from maze.x000m import Experiment  # noqa: PLC0415
    elif exp_name.startswith("x000l"):
        from maze.x000l import Experiment  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    exp = Experiment()
    exp.setup_optimizers()
    resume_from_checkpoint(exp, ckpt_path)
    exp.total_train_steps = exp.current_step + warmup + n_trials * 2 + 100

    loader = exp.make_train_loader()
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # Warmup: fill the training state pool
    print(f"Warming up ({warmup} steps)...", flush=True)
    for i in range(warmup):
        batch = next_batch()
        inputs, labels, puzzle_ids, valid_count = batch
        exp.step(inputs, labels, puzzle_ids, valid_count)
        if i % 10 == 0:
            print(f"  warmup step {i}", flush=True)

    # Now run sensitivity trials
    # Each trial: save state → forward for q_halt → train step → forward again
    print(f"Running {n_trials} sensitivity trials...", flush=True)

    all_mean_abs_delta = []
    all_std_delta = []
    all_frac_flipped = []
    all_n_positive_before = []
    all_n_positive_after = []
    all_mean_qhalt_before = []
    all_margin_before = []  # mean |q_halt| — how far from threshold

    for trial in range(n_trials):
        batch = next_batch()
        inputs, labels, puzzle_ids, valid_count = batch

        # Get current state for a clean forward pass to read q_halt
        state = exp._state  # noqa: SLF001
        active_mask = state.batch_index >= 0
        n_active = int(active_mask.sum().item())

        if n_active == 0:
            # Enqueue and fill to get active chains
            exp._enqueue(  # noqa: SLF001
                inputs[:valid_count],
                labels[:valid_count],
                puzzle_ids[:valid_count],
            )
            exp._fill_pending()  # noqa: SLF001
            active_mask = state.batch_index >= 0
            n_active = int(active_mask.sum().item())

        if n_active == 0:
            continue

        # Forward pass to get q_halt BEFORE the training step
        with torch.no_grad():
            fwd_before = exp._forward(state)  # noqa: SLF001
        q_halt_snap = fwd_before["q_halt"].float().cpu()
        del fwd_before
        torch.cuda.empty_cache()

        # Do one real training step (this updates weights)
        result = exp.step(inputs, labels, puzzle_ids, valid_count)
        if result is None:
            break

        # Forward pass to get q_halt AFTER the training step
        state2 = exp._state  # noqa: SLF001
        active_mask2 = state2.batch_index >= 0
        with torch.no_grad():
            fwd_after = exp._forward(state2)  # noqa: SLF001
        q_halt_snap2 = fwd_after["q_halt"].float().cpu()
        del fwd_after
        torch.cuda.empty_cache()

        # Find chains that were active both before and after
        both_active = (active_mask & active_mask2).cpu()
        if both_active.sum() == 0:
            continue

        q_before = q_halt_snap[both_active]
        q_after = q_halt_snap2[both_active]
        delta = q_after - q_before

        n_both = len(delta)
        mean_abs_delta = delta.abs().mean().item()
        std_delta = delta.std().item() if n_both > 1 else 0.0
        sign_before = (q_before > 0).float()
        sign_after = (q_after > 0).float()
        frac_flipped = (sign_before != sign_after).float().mean().item()
        n_pos_before = int((q_before > 0).sum().item())
        n_pos_after = int((q_after > 0).sum().item())
        mean_qhalt = q_before.mean().item()
        margin = q_before.abs().mean().item()

        all_mean_abs_delta.append(mean_abs_delta)
        all_std_delta.append(std_delta)
        all_frac_flipped.append(frac_flipped)
        all_n_positive_before.append(n_pos_before / max(n_both, 1))
        all_n_positive_after.append(n_pos_after / max(n_both, 1))
        all_mean_qhalt_before.append(mean_qhalt)
        all_margin_before.append(margin)

        if trial % 20 == 0:
            print(
                f"  trial {trial}: |delta|={mean_abs_delta:.4f} "
                f"flipped={frac_flipped:.3f} "
                f"pos_before={n_pos_before}/{n_both} "
                f"pos_after={n_pos_after}/{n_both} "
                f"margin={margin:.4f}",
                flush=True,
            )

    # Report
    n = len(all_mean_abs_delta)
    print(f"\n{'=' * 60}")
    print(f"Q_HALT SENSITIVITY: {exp_name} @ step {step}  (n={n} trials)")
    print(f"{'=' * 60}")

    for name, vals in [
        ("|delta| per step", all_mean_abs_delta),
        ("std(delta)", all_std_delta),
        ("frac sign-flipped", all_frac_flipped),
        ("frac positive (before)", all_n_positive_before),
        ("frac positive (after)", all_n_positive_after),
        ("mean q_halt (before)", all_mean_qhalt_before),
        ("mean |q_halt| (margin)", all_margin_before),
    ]:
        m, s, mn, mx = stats(vals)
        print(f"  {name:30s}: mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}")

    # Key diagnostic: is the step-to-step delta large relative to the margin?
    if len(all_mean_abs_delta) > 0 and len(all_margin_before) > 0:
        avg_delta = sum(all_mean_abs_delta) / len(all_mean_abs_delta)
        avg_margin = sum(all_margin_before) / len(all_margin_before)
        ratio = avg_delta / avg_margin if avg_margin > 0 else float("inf")
        print(f"\n  delta/margin ratio: {ratio:.4f}")
        print(
            "    (>1.0 = single step moves q_halt more than its distance to threshold)"
        )
        print("    (<0.1 = q_halt is stable, threshold crossings are rare)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    run_probe(args.exp, args.step, args.n_trials, args.warmup)
