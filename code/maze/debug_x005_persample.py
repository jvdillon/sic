"""Per-sample analysis: which test puzzles break after the drop?"""

import random
import sys

from maze.x005 import Experiment
from util import numpy_rng

import torch


SNAPSHOT_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "maze/ckpts/debug_x005_drop/pre_step03565.pt"
)

snap = torch.load(SNAPSHOT_PATH, weights_only=False)
print(f"Loaded snapshot from {SNAPSHOT_PATH}, step={snap['step']}")


def restore_model_only(exp, snap):
    exp.model.load_state_dict(snap["model"])


def per_sample_eval(exp):
    """Return per-puzzle (num_correct_cells, grid_len, halt_step, preds)."""
    exp.model.eval()
    results = []
    with torch.no_grad():
        for batch in exp.make_test_loader():
            inputs = batch[0].to(exp.device)
            labels = batch[1].to(exp.device)
            puzzle_ids = batch[2]
            if puzzle_ids is not None:
                puzzle_ids = puzzle_ids.to(exp.device)
            valid_count = batch[3]
            B = inputs.shape[0]

            z_H, z_L = exp._init_z(B)
            has_halted = torch.zeros(B, device=exp.device, dtype=torch.bool)
            halt_step = torch.full(
                (B,), exp.max_reasoning_steps, device=exp.device, dtype=torch.int32,
            )

            out = {}
            for step in range(exp.max_reasoning_steps):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                q_halt = out["q_halt"]

                newly_halted = ~has_halted & (q_halt > 0)
                if newly_halted.any():
                    halt_step[newly_halted] = step + 1
                    has_halted[newly_halted] = True

            preds = out["logits"].argmax(dim=-1)
            grid_len = inputs.shape[1]

            for i in range(valid_count):
                correct_cells = (preds[i] == labels[i]).sum().item()
                results.append({
                    "correct_cells": correct_cells,
                    "grid_len": grid_len,
                    "halt_step": halt_step[i].item(),
                    "halted": has_halted[i].item(),
                    "preds": preds[i].cpu(),
                    "labels": labels[i].cpu(),
                    "inputs": inputs[i].cpu(),
                })
    exp.model.train()
    return results


exp = Experiment()
exp.setup_optimizers()

# PRE eval
print("\n=== PRE model (before drop step) ===")
restore_model_only(exp, snap)
pre_results = per_sample_eval(exp)
pre_correct = sum(1 for r in pre_results if r["correct_cells"] == r["grid_len"])
print(f"  puzzle_acc={100*pre_correct/len(pre_results):.2f}% "
      f"({pre_correct}/{len(pre_results)})")

# POST eval: apply one training step
print("\n=== Applying one training step... ===")
exp.model.load_state_dict(snap["model"])
exp.optimizer1.load_state_dict(snap["optimizer1"])
exp.optimizer2.load_state_dict(snap["optimizer2"])
exp.current_step = snap["step"]
exp.total_train_steps = snap["step"] + 10
state = exp._state
for key in ["z_H", "z_L", "batch_index", "h_step", "inputs", "labels",
            "puzzle_ids", "chain_indices", "active"]:
    getattr(state, key).copy_(snap[key].to(getattr(state, key).device))
exp._wrong_count.copy_(snap["_wrong_count"].to(exp._wrong_count.device))
exp._pending_inputs = snap["_pending_inputs"].to(exp.device)
exp._pending_labels = snap["_pending_labels"].to(exp.device)
exp._pending_puzzle_ids = snap["_pending_puzzle_ids"].to(exp.device)

random.setstate(snap["rng_python"])
numpy_rng.bit_generator.state = snap["rng_numpy"]
torch.set_rng_state(snap["rng_torch"])
for i, t in enumerate(snap["rng_cuda"]):
    torch.cuda.set_rng_state(t, i)

trainloader = exp.make_train_loader()
samples, targets, puzzle_ids, valid_count = next(iter(trainloader))
exp.step(
    samples.to(exp.device), targets.to(exp.device),
    puzzle_ids.to(exp.device) if puzzle_ids is not None else None, valid_count,
)

print("\n=== POST model (after drop step) ===")
post_results = per_sample_eval(exp)
post_correct = sum(1 for r in post_results if r["correct_cells"] == r["grid_len"])
print(f"  puzzle_acc={100*post_correct/len(post_results):.2f}% "
      f"({post_correct}/{len(post_results)})")

# Compare
print(f"\n=== COMPARISON ({len(pre_results)} puzzles) ===")
gained = []  # wrong->right
lost = []    # right->wrong
stayed_right = []
stayed_wrong = []

for i, (pre, post) in enumerate(zip(pre_results, post_results)):
    pre_ok = pre["correct_cells"] == pre["grid_len"]
    post_ok = post["correct_cells"] == post["grid_len"]
    if pre_ok and post_ok:
        stayed_right.append(i)
    elif pre_ok and not post_ok:
        lost.append(i)
    elif not pre_ok and post_ok:
        gained.append(i)
    else:
        stayed_wrong.append(i)

print(f"  stayed_right: {len(stayed_right)}")
print(f"  stayed_wrong: {len(stayed_wrong)}")
print(f"  gained (wrong->right): {len(gained)}")
print(f"  LOST (right->wrong): {len(lost)}")

# Detail on lost puzzles
if lost:
    print("\n=== LOST PUZZLES (right->wrong) detail ===")
    for idx in lost:
        pre = pre_results[idx]
        post = post_results[idx]
        diff_cells = (pre["preds"] != post["preds"]).sum().item()
        wrong_cells = post["grid_len"] - post["correct_cells"]
        print(f"  puzzle {idx:4d}: "
              f"pre_halt={pre['halt_step']:2d} post_halt={post['halt_step']:2d}  "
              f"post_wrong_cells={wrong_cells}  pred_diff_cells={diff_cells}")

# Halt step distribution
print("\n=== HALT STEP DISTRIBUTION ===")
for label, results in [("PRE", pre_results), ("POST", post_results)]:
    steps = [r["halt_step"] for r in results]
    halted = sum(1 for r in results if r["halted"])
    from collections import Counter
    step_counts = Counter(steps)
    print(f"  {label}: halted={halted}/{len(results)}  "
          f"steps={dict(sorted(step_counts.items()))}")

# Cell-level analysis on lost puzzles
if lost:
    print("\n=== CELL-LEVEL BREAKDOWN (lost puzzles) ===")
    total_newly_wrong = 0
    total_newly_right = 0
    total_stayed_wrong = 0
    for idx in lost:
        pre_correct_mask = pre_results[idx]["preds"] == pre_results[idx]["labels"]
        post_correct_mask = post_results[idx]["preds"] == post_results[idx]["labels"]
        newly_wrong = (pre_correct_mask & ~post_correct_mask).sum().item()
        newly_right = (~pre_correct_mask & post_correct_mask).sum().item()
        sw = (~pre_correct_mask & ~post_correct_mask).sum().item()
        total_newly_wrong += newly_wrong
        total_newly_right += newly_right
        total_stayed_wrong += sw
    print(f"  cells newly wrong: {total_newly_wrong}")
    print(f"  cells newly right: {total_newly_right}")
    print(f"  cells stayed wrong: {total_stayed_wrong}")

# stayed_wrong analysis
if stayed_wrong:
    print("\n=== STAYED-WRONG ANALYSIS ===")
    improved = 0
    worsened = 0
    same = 0
    for idx in stayed_wrong:
        pre_c = pre_results[idx]["correct_cells"]
        post_c = post_results[idx]["correct_cells"]
        if post_c > pre_c:
            improved += 1
        elif post_c < pre_c:
            worsened += 1
        else:
            same += 1
    print(f"  improved (more cells right): {improved}")
    print(f"  worsened (fewer cells right): {worsened}")
    print(f"  same: {same}")
