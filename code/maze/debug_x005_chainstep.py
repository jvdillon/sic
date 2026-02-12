"""Check: what h_step were chains at during the drop step?

If many chains were at late ACT steps, the gradient backpropagates
through many iterations of the shared reasoning block, which could
cause a larger effective update and overshoot.
"""

import sys

import torch


SNAPSHOT_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "maze/ckpts/debug_x005_drop/pre_step03565.pt"
)

snap = torch.load(SNAPSHOT_PATH, weights_only=False)
print(f"Loaded snapshot from {SNAPSHOT_PATH}, step={snap['step']}")

h_step = snap["h_step"]
active = snap["active"]
batch_index = snap["batch_index"]

print(f"\nTraining state at step {snap['step']}:")
print(f"  active chains: {active.sum().item()}")
print("  h_step values for active chains:")

active_h = h_step[active]
print(f"    h_steps: {active_h.tolist()}")
print(f"    min={active_h.min().item()} max={active_h.max().item()} "
      f"mean={active_h.float().mean().item():.1f}")

from collections import Counter


counts = Counter(active_h.tolist())
print(f"    distribution: {dict(sorted(counts.items()))}")

# How many are at late steps (>= 4)?
late = (active_h >= 4).sum().item()
total = active_h.shape[0]
print(f"    late (h_step >= 4): {late}/{total} ({100*late/total:.1f}%)")

# Also check batch_index to see which training samples are active
active_bi = batch_index[active]
print(f"\n  active batch_indices: unique={active_bi.unique().shape[0]}")
