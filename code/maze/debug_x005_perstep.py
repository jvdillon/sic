"""Per-ACT-step accuracy for PRE vs POST, ignoring halt signal."""

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


def per_step_eval(exp):
    """Return per-puzzle, per-step accuracy (ignoring halt)."""
    exp.model.eval()
    # per_step_correct[step][puzzle] = num correct cells at that step
    all_step_correct = []  # list of (grid_len, [step0_correct, step1_correct, ...])
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
            step_corrects = torch.zeros(B, MAX_STEPS, device=exp.device)

            for step in range(MAX_STEPS):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                preds = out["logits"].argmax(dim=-1)
                step_corrects[:, step] = (preds == labels).sum(dim=-1).float()

            for i in range(valid_count):
                all_step_correct.append(
                    (grid_len, step_corrects[i].cpu())
                )
    exp.model.train()
    return all_step_correct


exp = Experiment()
exp.setup_optimizers()

# PRE
print("\n=== PRE model ===")
exp.model.load_state_dict(snap["model"])
pre_data = per_step_eval(exp)

# POST: apply one step
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

print("\n=== POST model ===")
post_data = per_step_eval(exp)

# Aggregate: per-step cell_acc and puzzle_acc
print("\n=== PER-STEP ACCURACY (ignoring halt) ===")
print(f"{'step':>4s}  {'PRE_cell':>9s} {'PRE_puz':>9s}  {'POST_cell':>9s} {'POST_puz':>9s}  {'delta_puz':>9s}")
n = len(pre_data)
for s in range(MAX_STEPS):
    pre_cells = sum(d[1][s].item() for d in pre_data)
    post_cells = sum(d[1][s].item() for d in post_data)
    total_cells = sum(d[0] for d in pre_data)
    pre_puz = sum(1 for d in pre_data if d[1][s].item() == d[0])
    post_puz = sum(1 for d in post_data if d[1][s].item() == d[0])
    pre_ca = 100 * pre_cells / total_cells
    post_ca = 100 * post_cells / total_cells
    pre_pa = 100 * pre_puz / n
    post_pa = 100 * post_puz / n
    print(f"  {s+1:2d}    {pre_ca:7.2f}%  {pre_pa:7.2f}%   {post_ca:7.2f}%  {post_pa:7.2f}%   {post_pa - pre_pa:+7.2f}%")

# Per-puzzle best-step analysis
print("\n=== BEST-STEP ANALYSIS ===")
pre_best = sum(1 for d in pre_data if any(d[1][s].item() == d[0] for s in range(MAX_STEPS)))
post_best = sum(1 for d in post_data if any(d[1][s].item() == d[0] for s in range(MAX_STEPS)))
print(f"  PRE  puzzles correct at ANY step: {pre_best}/{n} ({100*pre_best/n:.1f}%)")
print(f"  POST puzzles correct at ANY step: {post_best}/{n} ({100*post_best/n:.1f}%)")

# Which step is best for each model?
pre_step_puz = [sum(1 for d in pre_data if d[1][s].item() == d[0]) for s in range(MAX_STEPS)]
post_step_puz = [sum(1 for d in post_data if d[1][s].item() == d[0]) for s in range(MAX_STEPS)]
pre_best_step = max(range(MAX_STEPS), key=lambda s: pre_step_puz[s])
post_best_step = max(range(MAX_STEPS), key=lambda s: post_step_puz[s])
print(f"  PRE  best step: {pre_best_step+1} ({pre_step_puz[pre_best_step]} puzzles)")
print(f"  POST best step: {post_best_step+1} ({post_step_puz[post_best_step]} puzzles)")
