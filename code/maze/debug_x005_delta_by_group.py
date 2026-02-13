"""Compare z_H deltas for lost vs stayed-right puzzles."""

from typing import Any

import random
import sys

from maze.x005 import Experiment
from util import numpy_rng

import torch


SNAPSHOT_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else ("maze/ckpts/debug_x005_drop/pre_step03565.pt")
)

snap = torch.load(SNAPSHOT_PATH, weights_only=False)
MAX_STEPS = 16


def collect_per_puzzle(exp: Any) -> Any:
    """Return per-puzzle: (step_deltas[15], step_correct[16], grid_len)."""
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

            z_H, z_L = exp._init_z(B)  # noqa: SLF001
            prev_z_H = None
            deltas = torch.zeros(B, MAX_STEPS - 1)
            corrects = torch.zeros(B, MAX_STEPS, dtype=torch.int32)

            for step in range(MAX_STEPS):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)  # noqa: SLF001
                z_H = out["z_H"]
                z_L = out["z_L"]
                preds = out["logits"].argmax(dim=-1)
                corrects[:, step] = (preds == labels).sum(dim=-1)

                if prev_z_H is not None:
                    for i in range(B):
                        d = (z_H[i].float() - prev_z_H[i].float()).norm(dim=-1).mean()
                        deltas[i, step - 1] = d.item()
                prev_z_H = z_H.clone()

            results.extend(
                (deltas[i], corrects[i], grid_len) for i in range(valid_count)
            )
    exp.model.train()
    return results


exp = Experiment()
exp.setup_optimizers()

# PRE
exp.model.load_state_dict(snap["model"])
pre = collect_per_puzzle(exp)

# POST
exp.model.load_state_dict(snap["model"])
assert exp.optimizer1 is not None
exp.optimizer1.load_state_dict(snap["optimizer1"])
assert exp.optimizer2 is not None
exp.optimizer2.load_state_dict(snap["optimizer2"])
exp.current_step = snap["step"]
exp.total_train_steps = snap["step"] + 10
state = exp._state  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
for key in [
    "z_H",
    "z_L",
    "batch_index",
    "h_step",
    "inputs",
    "labels",
    "puzzle_ids",
    "chain_indices",
    "active",
]:
    getattr(state, key).copy_(snap[key].to(getattr(state, key).device))
exp._wrong_count.copy_(snap["_wrong_count"].to(exp._wrong_count.device))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
exp._pending_inputs = snap["_pending_inputs"].to(exp.device)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
exp._pending_labels = snap["_pending_labels"].to(exp.device)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
exp._pending_puzzle_ids = snap["_pending_puzzle_ids"].to(exp.device)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
random.setstate(snap["rng_python"])
numpy_rng.bit_generator.state = snap["rng_numpy"]
torch.set_rng_state(snap["rng_torch"])
for i, t in enumerate(snap["rng_cuda"]):
    torch.cuda.set_rng_state(t, i)
trainloader = exp.make_train_loader()
samples, targets, puzzle_ids, valid_count = next(iter(trainloader))
exp.step(
    samples.to(exp.device),
    targets.to(exp.device),
    puzzle_ids.to(exp.device) if puzzle_ids is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
    valid_count,
)
post = collect_per_puzzle(exp)

# Classify puzzles
n = len(pre)
lost = []
stayed_right = []
stayed_wrong = []
gained = []
for i in range(n):
    pre_ever = any(pre[i][1][s] == pre[i][2] for s in range(MAX_STEPS))
    post_ever = any(post[i][1][s] == post[i][2] for s in range(MAX_STEPS))
    if pre_ever and not post_ever:
        lost.append(i)
    elif pre_ever and post_ever:
        stayed_right.append(i)
    elif not pre_ever and post_ever:
        gained.append(i)
    else:
        stayed_wrong.append(i)

print(
    f"lost={len(lost)} stayed_right={len(stayed_right)} "
    f"stayed_wrong={len(stayed_wrong)} gained={len(gained)}"
)


def group_mean_delta(indices: Any, data: Any, step_idx: Any) -> Any:
    if not indices:
        return 0.0
    return sum(data[i][0][step_idx].item() for i in indices) / len(indices)


# Compare late-step deltas by group
print("\n=== MEAN z_H DELTA AT LATE STEPS (POST model) ===")
print(
    f"{'step':>6s}  {'lost':>8s}  {'stayed_R':>8s}  {'stayed_W':>8s}  {'ratio_L/R':>10s}"
)
for s in range(MAX_STEPS - 1):
    loss_val = group_mean_delta(lost, post, s)
    r = group_mean_delta(stayed_right, post, s)
    w = group_mean_delta(stayed_wrong, post, s)
    ratio = loss_val / r if r > 0 else float("inf")
    print(f"  {s + 1}->{s + 2}  {loss_val:8.4f}  {r:8.4f}  {w:8.4f}  {ratio:10.3f}")

# Same for PRE
print("\n=== MEAN z_H DELTA AT LATE STEPS (PRE model) ===")
print(
    f"{'step':>6s}  {'lost':>8s}  {'stayed_R':>8s}  {'stayed_W':>8s}  {'ratio_L/R':>10s}"
)
for s in range(MAX_STEPS - 1):
    loss_val = group_mean_delta(lost, pre, s)
    r = group_mean_delta(stayed_right, pre, s)
    w = group_mean_delta(stayed_wrong, pre, s)
    ratio = loss_val / r if r > 0 else float("inf")
    print(f"  {s + 1}->{s + 2}  {loss_val:8.4f}  {r:8.4f}  {w:8.4f}  {ratio:10.3f}")

# Distribution of final delta (step 15->16) for each group
print("\n=== DISTRIBUTION OF FINAL DELTA (step 15->16, POST) ===")
for label, indices in [
    ("lost", lost),
    ("stayed_right", stayed_right),
    ("stayed_wrong", stayed_wrong),
]:
    vals = sorted([post[i][0][14].item() for i in indices])
    if vals:
        p25 = vals[len(vals) // 4]
        p50 = vals[len(vals) // 2]
        p75 = vals[3 * len(vals) // 4]
        print(
            f"  {label:12s}: p25={p25:.4f} p50={p50:.4f} p75={p75:.4f} "
            f"min={vals[0]:.4f} max={vals[-1]:.4f}"
        )
