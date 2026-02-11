"""Probe: measure correlation between n_halted and gradient norm.

Loads a checkpoint, runs ~500 training steps recording per-step:
n_halted, loss, grad_norm (total, muon, adam), mean sample age.

We monkey-patch _update_weights to capture grad norms before zeroing.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv --quiet --project ~/projects/sic run python \
        maze/probe_halt_variance.py --exp x000l --step 3000
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import argparse
import csv
import math
import pathlib
import sys


if TYPE_CHECKING:
    import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import resume_from_checkpoint


def grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.float().norm().item() ** 2
    return total**0.5


def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def stats(xs: list[float]) -> tuple[float, float, float, float]:
    n = len(xs)
    if n == 0:
        return 0, 0, 0, 0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, math.sqrt(v), min(xs), max(xs)


def run_probe(exp_name: str, step: int, n_steps: int = 500, warmup: int = 50):
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
    exp.total_train_steps = exp.current_step + warmup + n_steps + 100

    # Identify param groups
    muon_params, adam_params = [], []
    for n, p in exp.model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim >= 2
            and "embed" not in n
            and "head" not in n
            and "register_tokens" not in n
        ):
            muon_params.append(p)
        else:
            adam_params.append(p)

    # Monkey-patch: capture grads, zero without stepping optimizers
    _captured = {}

    def _patched_update_weights():
        _captured["gn_total"] = grad_norm(list(exp.model.parameters()))
        _captured["gn_muon"] = grad_norm(muon_params)
        _captured["gn_adam"] = grad_norm(adam_params)
        exp.model.zero_grad()

    exp._update_weights = _patched_update_weights  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    loader = exp.make_train_loader()
    data_iter = iter(loader)

    out_path = pathlib.Path(f"maze/probe_{exp_name}_step{step:05d}.csv")
    f = out_path.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        [
            "probe_step",
            "n_halted",
            "n_active",
            "n_expunged",
            "loss",
            "grad_norm_total",
            "grad_norm_muon",
            "grad_norm_adam",
            "mean_age",
            "std_age",
            "min_age",
            "max_age",
        ]
    )

    # Collect data for analysis
    rows = []

    print(f"Running {warmup} warmup + {n_steps} probe steps...", flush=True)
    total = warmup + n_steps

    for i in range(total):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        inputs, labels, puzzle_ids, valid_count = batch
        _captured.clear()

        result = exp.step(inputs, labels, puzzle_ids, valid_count)
        if result is None:
            print(f"step() returned None at iteration {i}")
            break

        n_halted = int(result["n_halted"].item())
        n_active = int(result["n_active"].item())
        n_expunged = int(result["n_expunged"].item())
        loss = result["lm"].item()

        state = exp._state  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        active_mask = state.active
        if active_mask.any():
            ages = state.h_step[active_mask].float()
            mean_age = ages.mean().item()
            std_age = ages.std().item() if len(ages) > 1 else 0.0
            min_age = ages.min().item()
            max_age = ages.max().item()
        else:
            mean_age = std_age = min_age = max_age = 0.0

        gn_total = _captured.get("gn_total", 0.0)
        gn_muon = _captured.get("gn_muon", 0.0)
        gn_adam = _captured.get("gn_adam", 0.0)

        if i >= warmup:
            row = {
                "n_halted": n_halted,
                "n_active": n_active,
                "n_expunged": n_expunged,
                "loss": loss,
                "gn_total": gn_total,
                "gn_muon": gn_muon,
                "gn_adam": gn_adam,
                "mean_age": mean_age,
                "std_age": std_age,
                "min_age": min_age,
                "max_age": max_age,
            }
            rows.append(row)
            writer.writerow(
                [
                    i - warmup,
                    n_halted,
                    n_active,
                    n_expunged,
                    f"{loss:.6f}",
                    f"{gn_total:.6f}",
                    f"{gn_muon:.6f}",
                    f"{gn_adam:.6f}",
                    f"{mean_age:.2f}",
                    f"{std_age:.2f}",
                    f"{min_age:.0f}",
                    f"{max_age:.0f}",
                ]
            )

        if i % 50 == 0:
            phase = "warmup" if i < warmup else "probe"
            print(
                f"  [{phase}] iter {i}: n_halted={n_halted} n_active={n_active} "
                f"loss={loss:.4f} grad_norm={gn_total:.4f} mean_age={mean_age:.1f}",
                flush=True,
            )

    f.close()
    print(f"\nSaved {out_path}")

    # Analysis
    B = exp.batch_size
    nh = [r["n_halted"] for r in rows]
    losses = [r["loss"] for r in rows]
    gn_totals = [r["gn_total"] for r in rows]
    gn_muons = [r["gn_muon"] for r in rows]
    gn_adams = [r["gn_adam"] for r in rows]
    ages = [r["mean_age"] for r in rows]

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {exp_name} @ step {step}  (n={len(rows)} samples, B={B})")
    print(f"{'=' * 60}")

    print("\nDescriptive stats:")
    for name, vals in [
        ("n_halted", nh),
        ("loss", losses),
        ("grad_norm_total", gn_totals),
        ("mean_age", ages),
    ]:
        m, s, mn, mx = stats(vals)
        print(f"  {name:20s}: mean={m:.4f} std={s:.4f} min={mn:.4f} max={mx:.4f}")

    print("\nCorrelations (n_halted vs):")
    for name, vals in [
        ("loss", losses),
        ("grad_total", gn_totals),
        ("grad_muon", gn_muons),
        ("grad_adam", gn_adams),
    ]:
        r = pearson_r(nh, vals)
        print(f"  {name:20s}: r={r:+.4f}  r²={r**2:.4f}")

    print("\nCorrelations (mean_age vs):")
    for name, vals in [
        ("loss", losses),
        ("grad_total", gn_totals),
        ("grad_muon", gn_muons),
        ("grad_adam", gn_adams),
    ]:
        r = pearson_r(ages, vals)
        print(f"  {name:20s}: r={r:+.4f}  r²={r**2:.4f}")

    # Bucket analysis
    sorted_nh = sorted(nh)
    q33 = sorted_nh[len(sorted_nh) // 3]
    q67 = sorted_nh[2 * len(sorted_nh) // 3]
    buckets = {"low": [], "mid": [], "high": []}
    for r in rows:
        if r["n_halted"] <= q33:
            buckets["low"].append(r)
        elif r["n_halted"] <= q67:
            buckets["mid"].append(r)
        else:
            buckets["high"].append(r)

    print(f"\nBucket analysis (n_halted <= {q33} / {q33}-{q67} / > {q67}):")
    for label in ["low", "mid", "high"]:
        sub = buckets[label]
        if len(sub) > 0:
            gm, gs, _, _ = stats([r["gn_total"] for r in sub])
            lm, ls, _, _ = stats([r["loss"] for r in sub])
            am, _, _, _ = stats([r["mean_age"] for r in sub])
            print(
                f"  {label:4s} (n={len(sub):3d}): "
                f"grad_norm={gm:.4f}±{gs:.4f}  "
                f"loss={lm:.4f}±{ls:.4f}  "
                f"mean_age={am:.1f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()
    run_probe(args.exp, args.step, args.n_steps, args.warmup)
