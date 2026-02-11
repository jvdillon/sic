"""Probe: does q_halt correlate with puzzle difficulty?

For each test puzzle, run the model at fixed depths 1..max_depth (ignoring
q_halt). Record the minimum depth at which it gets 100% correct ("intrinsic
difficulty"). Then run with q_halt and record actual halt depth. Compute
correlation.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv --quiet --project ~/projects/sic run python \
        maze/probe_difficulty_vs_halt.py --exp x000l --step 15000
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import resume_from_checkpoint


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


def run_probe(exp_name: str, step: int, max_depth: int = 16):
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

    loader = exp.make_test_loader()
    model = exp.model
    device = exp.device
    assert device is not None
    dtype = exp.dtype

    # Disable torch.compile to avoid recompilation issues with varying batch sizes
    model.config.compile_core = False

    # Collect all test puzzles
    all_inputs = []
    all_labels = []
    for batch in loader:
        inputs, labels, _puzzle_ids, valid_count = batch
        for i in range(valid_count):
            all_inputs.append(inputs[i])
            all_labels.append(labels[i])
    n_puzzles = len(all_inputs)
    print(f"Loaded {n_puzzles} test puzzles")

    intrinsic_difficulty = []  # min depth for 100% correct (or max_depth+1 if never)
    qhalt_halt_depth = []  # depth at which q_halt first goes positive

    batch_size = min(256, n_puzzles)
    H_init = model.H_init[0]
    L_init = model.L_init[0]

    with torch.no_grad():
        for start in range(0, n_puzzles, batch_size):
            end = min(start + batch_size, n_puzzles)
            B = end - start
            inputs = torch.stack(all_inputs[start:end])
            labels = torch.stack(all_labels[start:end])

            z_H = H_init.unsqueeze(0).expand(B, -1, -1).clone()
            z_L = L_init.unsqueeze(0).expand(B, -1, -1).clone()

            first_correct_depth = torch.full((B,), max_depth + 1, device=device)
            first_halt_depth = torch.full((B,), max_depth + 1, device=device)

            puzzle_ids = torch.zeros(B, device=device, dtype=torch.int32)
            for d in range(max_depth):
                with torch.autocast(device_type=device.type, dtype=dtype):
                    out = model(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                logits = out["logits"]
                q_halt = out["q_halt"]

                preds = logits.argmax(dim=-1)
                correct = (preds == labels).all(dim=-1)

                newly_correct = correct & (first_correct_depth > d + 1)
                first_correct_depth[newly_correct] = d + 1

                halt_positive = (q_halt > 0) & (first_halt_depth > d + 1)
                first_halt_depth[halt_positive] = d + 1

            for i in range(B):
                intrinsic_difficulty.append(first_correct_depth[i].item())
                qhalt_halt_depth.append(first_halt_depth[i].item())

            if start % 512 == 0:
                print(f"  processed {start}/{n_puzzles}", flush=True)

    # Analysis
    n = len(intrinsic_difficulty)
    diff = [float(x) for x in intrinsic_difficulty]
    halt = [float(x) for x in qhalt_halt_depth]

    print(f"\n{'=' * 60}")
    print(f"DIFFICULTY vs HALT: {exp_name} @ step {step}  (n={n} puzzles)")
    print(f"{'=' * 60}")

    # Distribution of intrinsic difficulty
    diff_counts: dict[int, int] = {}
    for d in intrinsic_difficulty:
        d_int = int(d)
        diff_counts[d_int] = diff_counts.get(d_int, 0) + 1
    print("\nIntrinsic difficulty distribution (min depth for 100% correct):")
    for depth in sorted(diff_counts):
        label = f"depth={depth}" if depth <= max_depth else "never"
        pct = 100 * diff_counts[depth] / n
        print(f"  {label:12s}: {diff_counts[depth]:5d} ({pct:5.1f}%)")

    # Distribution of q_halt halt depth
    halt_counts: dict[int, int] = {}
    for h in qhalt_halt_depth:
        h_int = int(h)
        halt_counts[h_int] = halt_counts.get(h_int, 0) + 1
    print("\nq_halt halt depth distribution (first depth q_halt > 0):")
    for depth in sorted(halt_counts):
        label = f"depth={depth}" if depth <= max_depth else "never"
        pct = 100 * halt_counts[depth] / n
        print(f"  {label:12s}: {halt_counts[depth]:5d} ({pct:5.1f}%)")

    # Correlation
    r = pearson_r(diff, halt)
    print(
        f"\nCorrelation(intrinsic_difficulty, q_halt_depth): r={r:+.4f}  r²={r**2:.4f}"
    )

    # Cross-tabulation: for puzzles solvable at depth 1, what does q_halt do?
    print("\nCross-tab: intrinsic difficulty vs q_halt halt depth")
    print(
        f"  {'diff':>6s} | {'n':>5s} | {'halt=1':>6s} | {'halt=2':>6s} | {'halt≥3':>6s} | {'never':>6s} | mean_halt"
    )
    for target_diff in sorted(diff_counts):
        if target_diff > max_depth:
            continue
        indices = [
            i for i, d in enumerate(intrinsic_difficulty) if int(d) == target_diff
        ]
        if not indices:
            continue
        halts_for_diff = [qhalt_halt_depth[i] for i in indices]
        n_d = len(indices)
        h1 = sum(1 for h in halts_for_diff if h == 1)
        h2 = sum(1 for h in halts_for_diff if h == 2)
        h3p = sum(1 for h in halts_for_diff if 3 <= h <= max_depth)
        hn = sum(1 for h in halts_for_diff if h > max_depth)
        mean_h = sum(min(h, max_depth) for h in halts_for_diff) / n_d
        print(
            f"  {target_diff:>6d} | {n_d:>5d} | {100 * h1 / n_d:>5.1f}% | {100 * h2 / n_d:>5.1f}% | "
            f"{100 * h3p / n_d:>5.1f}% | {100 * hn / n_d:>5.1f}% | {mean_h:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--max_depth", type=int, default=16)
    args = parser.parse_args()
    run_probe(args.exp, args.step, args.max_depth)
