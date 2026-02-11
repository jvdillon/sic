"""Probe: measure model confidence on correct vs incorrect predictions.

For unsolved puzzles, compare the softmax probability on correct vs wrong cells.
Is the model uncertain at error sites, or confidently wrong?

Usage:
    CUDA_VISIBLE_DEVICES=0 uv --quiet --project ~/projects/sic run python \
        maze/probe_confidence.py --exp x000l --step 15000
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import resume_from_checkpoint


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
    model.config.compile_core = False

    all_inputs, all_labels = [], []
    for batch in loader:
        inputs, labels, _puzzle_ids, valid_count = batch
        for i in range(valid_count):
            all_inputs.append(inputs[i])
            all_labels.append(labels[i])
    n_puzzles = len(all_inputs)

    H_init = model.H_init[0]
    L_init = model.L_init[0]
    batch_size = min(256, n_puzzles)

    # Collect per-cell confidence at depth 1 (since accuracy is flat)
    correct_probs = []  # P(true label) for cells predicted correctly
    wrong_probs = []  # P(true label) for cells predicted incorrectly
    wrong_pred_probs = []  # P(predicted label) for cells predicted incorrectly
    wrong_entropies = []  # entropy at error sites

    # Also track per-puzzle: solved vs unsolved
    solved_conf = []  # mean P(true label) for solved puzzles
    unsolved_conf = []  # mean P(true label) for unsolved puzzles

    with torch.no_grad():
        for start in range(0, n_puzzles, batch_size):
            end = min(start + batch_size, n_puzzles)
            B = end - start
            inputs = torch.stack(all_inputs[start:end])
            labels = torch.stack(all_labels[start:end])

            z_H = H_init.unsqueeze(0).expand(B, -1, -1).clone()
            z_L = L_init.unsqueeze(0).expand(B, -1, -1).clone()
            puzzle_ids = torch.zeros(B, device=device, dtype=torch.int32)

            # Run max_depth iterations, collect logits at each
            out = None
            for _ in range(max_depth):
                with torch.autocast(device_type=device.type, dtype=dtype):
                    out = model(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]

            # Use final depth logits
            assert out is not None
            logits = out["logits"].float()  # [B, seq, vocab]
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)  # [B, seq]

            for i in range(B):
                p = probs[i]  # [seq, vocab]
                pred = preds[i]  # [seq]
                lab = labels[i]  # [seq]

                # Per-cell true-label probability
                true_prob = p[torch.arange(len(lab), device=device), lab]

                correct_mask = pred == lab
                wrong_mask = ~correct_mask
                all_correct = correct_mask.all()

                if all_correct:
                    solved_conf.append(true_prob.mean().item())
                else:
                    unsolved_conf.append(true_prob.mean().item())

                    # Entropy at error sites
                    if wrong_mask.any():
                        p_wrong = p[wrong_mask]  # [n_wrong, vocab]
                        entropy = -(p_wrong * (p_wrong + 1e-10).log()).sum(dim=-1)
                        wrong_entropies.extend(entropy.tolist())

                        # P(true label) at error sites
                        true_at_wrong = true_prob[wrong_mask]
                        wrong_probs.extend(true_at_wrong.tolist())

                        # P(predicted label) at error sites
                        pred_at_wrong = p_wrong[
                            torch.arange(len(p_wrong), device=device), pred[wrong_mask]
                        ]
                        wrong_pred_probs.extend(pred_at_wrong.tolist())

                    # P(true label) at correct sites
                    if correct_mask.any():
                        correct_probs.extend(true_prob[correct_mask].tolist())

            if start % 512 == 0:
                print(f"  processed {start}/{n_puzzles}", flush=True)

    def stats(xs: list[float]) -> tuple[float, float, float, float]:
        if not xs:
            return 0, 0, 0, 0
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return m, math.sqrt(v), min(xs), max(xs)

    n_solved = len(solved_conf)
    n_unsolved = len(unsolved_conf)
    print(f"\n{'=' * 60}")
    print(f"CONFIDENCE ANALYSIS: {exp_name} @ step {step}")
    print(f"  {n_solved} solved, {n_unsolved} unsolved (of {n_puzzles})")
    print(f"{'=' * 60}")

    m, s, mn, mx = stats(solved_conf)
    print(f"\nSolved puzzles — mean P(true label):   {m:.4f} +/- {s:.4f}")
    m, s, mn, mx = stats(unsolved_conf)
    print(f"Unsolved puzzles — mean P(true label): {m:.4f} +/- {s:.4f}")

    print(f"\nAt ERROR sites ({len(wrong_probs)} cells across {n_unsolved} puzzles):")
    m, s, mn, mx = stats(wrong_probs)
    print(f"  P(true label):      {m:.4f} +/- {s:.4f}  (min={mn:.4f} max={mx:.4f})")
    m, s, mn, mx = stats(wrong_pred_probs)
    print(f"  P(predicted label): {m:.4f} +/- {s:.4f}  (min={mn:.4f} max={mx:.4f})")
    m, s, mn, mx = stats(wrong_entropies)
    print(f"  Entropy:            {m:.4f} +/- {s:.4f}  (min={mn:.4f} max={mx:.4f})")
    print(f"  (max possible entropy for vocab=6: {math.log(6):.4f})")

    print(f"\nAt CORRECT sites of unsolved puzzles ({len(correct_probs)} cells):")
    m, s, mn, mx = stats(correct_probs)
    print(f"  P(true label):      {m:.4f} +/- {s:.4f}")

    # Histogram of P(predicted label) at error sites
    bins = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.01]
    print("\nConfidence histogram at error sites — P(predicted label):")
    for j in range(len(bins) - 1):
        count = sum(1 for p in wrong_pred_probs if bins[j] <= p < bins[j + 1])
        pct = 100 * count / max(len(wrong_pred_probs), 1)
        print(f"  [{bins[j]:.2f}, {bins[j + 1]:.2f}): {count:5d} ({pct:5.1f}%)")

    # How confident is the model at errors vs correct cells?
    # Histogram of P(true label) at error sites
    print("\nP(true label) histogram at error sites:")
    for j in range(len(bins) - 1):
        count = sum(1 for p in wrong_probs if bins[j] <= p < bins[j + 1])
        pct = 100 * count / max(len(wrong_probs), 1)
        print(f"  [{bins[j]:.2f}, {bins[j + 1]:.2f}): {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--max_depth", type=int, default=16)
    args = parser.parse_args()
    run_probe(args.exp, args.step, args.max_depth)
