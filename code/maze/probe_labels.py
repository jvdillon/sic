"""Quick probe: what do labels actually look like, and what does the loss see?"""

from __future__ import annotations

import math
import pathlib
import sys

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import resume_from_checkpoint
from maze.x000l import Experiment


def run():
    exp = Experiment()
    exp.setup_optimizers()
    resume_from_checkpoint(exp, "maze/ckpts/x000l/step15000.pt")
    model = exp.model
    model.config.compile_core = False
    device = exp.device
    assert device is not None
    dtype = exp.dtype

    loader = exp.make_test_loader()
    batch = next(iter(loader))
    inputs, labels, _pids, vc = batch

    inp = inputs[0]  # [900]
    lab = labels[0]  # [900]

    print("Input token distribution:")
    for t in range(7):
        c = (inp == t).sum().item()
        if c > 0:
            print(f"  token {t}: {c}")

    print("\nLabel token distribution:")
    for t in range(7):
        c = (lab == t).sum().item()
        if c > 0:
            print(f"  token {t}: {c}")

    print(f"\nPositions where input != label: {(inp != lab).sum().item()}")
    diff_mask = inp != lab
    print(f"  Input tokens at diff positions: {inp[diff_mask].unique().tolist()}")
    print(f"  Label tokens at diff positions: {lab[diff_mask].unique().tolist()}")

    # Check loss_ignore_index
    print(f"\nloss_ignore_index = {exp.loss_ignore_index}")
    print(f"Positions with label=0: {(lab == 0).sum().item()}")

    # Now check loss_sum_normalize
    print(f"loss_sum_normalize = {exp.loss_sum_normalize}")

    # Run model and compute loss manually to understand gradient weighting
    H_init = model.H_init[0]
    L_init = model.L_init[0]
    B = 1
    inp_batch = inp.unsqueeze(0)
    lab_batch = lab.unsqueeze(0)
    z_H = H_init.unsqueeze(0).expand(B, -1, -1).clone()
    z_L = L_init.unsqueeze(0).expand(B, -1, -1).clone()
    pids = torch.zeros(B, device=device, dtype=torch.int32)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype):
            out = model(inp_batch, z_H, z_L, pids)
        logits = out["logits"].float()
        preds = logits.argmax(dim=-1)[0]

        # Per-token CE loss
        ce = torch.nn.functional.cross_entropy(
            logits[0], lab_batch[0].long(), reduction="none"
        )

    correct = preds == lab
    wrong = ~correct
    n_wrong = wrong.sum().item()
    n_correct = correct.sum().item()

    print("\nPer-cell loss stats:")
    print(f"  Overall: mean={ce.mean().item():.4f}")
    if n_correct > 0:
        print(f"  Correct cells ({n_correct}): mean={ce[correct].mean().item():.6f}")
    if n_wrong > 0:
        print(f"  Wrong cells ({n_wrong}): mean={ce[wrong].mean().item():.4f}")

    # What's the loss contribution ratio?
    total_loss = ce.sum().item()
    if n_wrong > 0:
        wrong_loss = ce[wrong].sum().item()
        print(f"\n  Total loss: {total_loss:.4f}")
        print(
            f"  Loss from {n_wrong} wrong cells: {wrong_loss:.4f} ({100 * wrong_loss / total_loss:.1f}%)"
        )
        print(
            f"  Loss from {n_correct} correct cells: {total_loss - wrong_loss:.4f} ({100 * (1 - wrong_loss / total_loss):.1f}%)"
        )

    # Check across multiple unsolved puzzles
    print("\n--- Across all test puzzles ---")
    all_inputs, all_labels = [], []
    for batch in loader:
        inputs, labels, _pids, vc = batch
        for i in range(vc):
            all_inputs.append(inputs[i])
            all_labels.append(labels[i])

    # Add back first batch
    first_batch = next(iter(exp.make_test_loader()))
    inputs, labels, _pids, vc = first_batch
    for i in range(vc):
        all_inputs.insert(0, inputs[i])
        all_labels.insert(0, labels[i])

    n_puzzles = len(all_inputs)
    n_diff_positions = [
        (all_inputs[i] != all_labels[i]).sum().item() for i in range(n_puzzles)
    ]

    m = sum(n_diff_positions) / len(n_diff_positions)
    s = math.sqrt(
        sum((x - m) ** 2 for x in n_diff_positions) / (len(n_diff_positions) - 1)
    )
    print(
        f"Positions where input != label: mean={m:.1f} std={s:.1f} min={min(n_diff_positions)} max={max(n_diff_positions)}"
    )


if __name__ == "__main__":
    run()
