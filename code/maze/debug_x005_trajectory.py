"""Per-puzzle accuracy trajectory across ACT steps, PRE vs POST."""

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

MAX_STEPS = 16


def per_step_detail(exp):
    """Return per-puzzle: (grid_len, correct_per_step[16], wrong_positions_per_step)."""
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
            grid_len = inputs.shape[1]

            z_H, z_L = exp._init_z(B)
            # Track per-step: which cells are wrong
            step_wrong = torch.zeros(B, MAX_STEPS, grid_len, device=exp.device, dtype=torch.bool)

            for step in range(MAX_STEPS):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                preds = out["logits"].argmax(dim=-1)
                step_wrong[:, step] = preds != labels

            for i in range(valid_count):
                results.append(step_wrong[i].cpu())
    exp.model.train()
    return results


exp = Experiment()
exp.setup_optimizers()

# PRE
print("Evaluating PRE...")
exp.model.load_state_dict(snap["model"])
pre_wrong = per_step_detail(exp)

# POST
print("Applying training step...")
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

print("Evaluating POST...")
post_wrong = per_step_detail(exp)

n = len(pre_wrong)
grid_len = pre_wrong[0].shape[1]

# Classify puzzles
# "lost" = correct at best PRE step, wrong at all POST steps
pre_ever_perfect = []
post_ever_perfect = []
for i in range(n):
    pre_ever_perfect.append(any(pre_wrong[i][s].sum() == 0 for s in range(MAX_STEPS)))
    post_ever_perfect.append(any(post_wrong[i][s].sum() == 0 for s in range(MAX_STEPS)))

lost = [i for i in range(n) if pre_ever_perfect[i] and not post_ever_perfect[i]]
print(f"\nPuzzles perfect at some PRE step but never at any POST step: {len(lost)}")

# For lost puzzles: how many wrong cells at each step?
print("\n=== LOST PUZZLES: wrong cells per step ===")
print(f"{'puz':>4s}  " + "  ".join(f"s{s+1:02d}" for s in range(MAX_STEPS)))
for idx in lost[:50]:  # first 50
    pre_counts = [pre_wrong[idx][s].sum().item() for s in range(MAX_STEPS)]
    post_counts = [post_wrong[idx][s].sum().item() for s in range(MAX_STEPS)]
    pre_str = " ".join(f"{c:4d}" for c in pre_counts)
    post_str = " ".join(f"{c:4d}" for c in post_counts)
    print(f" {idx:4d} PRE:  {pre_str}")
    print(f"      POST: {post_str}")

# Do wrong cells accumulate or shift?
print("\n=== WRONG CELL STABILITY (POST, lost puzzles) ===")
# For each lost puzzle, check if wrong cells at step s are a subset/superset of step s+1
growing = 0  # wrong set grows monotonically
shifting = 0  # wrong set changes (some fixed, some break)
for idx in lost:
    is_growing = True
    for s in range(MAX_STEPS - 1):
        curr_wrong = set(post_wrong[idx][s].nonzero(as_tuple=True)[0].tolist())
        next_wrong = set(post_wrong[idx][s + 1].nonzero(as_tuple=True)[0].tolist())
        if not curr_wrong.issubset(next_wrong):
            is_growing = False
            break
    if is_growing:
        growing += 1
    else:
        shifting += 1
print(f"  monotonically growing wrong set: {growing}")
print(f"  shifting (some cells fix, others break): {shifting}")

# What fraction of POST wrong cells at step 16 were already wrong at step 2?
print("\n=== POST: wrong cell origin (step 2 vs step 16) ===")
inherited = 0
new_at_16 = 0
fixed_by_16 = 0
for idx in lost:
    w2 = set(post_wrong[idx][1].nonzero(as_tuple=True)[0].tolist())  # step 2 (idx 1)
    w16 = set(post_wrong[idx][15].nonzero(as_tuple=True)[0].tolist())  # step 16 (idx 15)
    inherited += len(w2 & w16)
    new_at_16 += len(w16 - w2)
    fixed_by_16 += len(w2 - w16)
print(f"  wrong at both step 2 and 16: {inherited}")
print(f"  wrong only at step 16 (new): {new_at_16}")
print(f"  wrong only at step 2 (fixed): {fixed_by_16}")
