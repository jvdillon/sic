"""Ablate: which param changes cause the drop? Zero out each delta individually."""

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
print(f"Loaded snapshot from {SNAPSHOT_PATH}, step={snap['step']}")


def restore_model_only(exp: Any, snap: Any) -> None:
    exp.model.load_state_dict(snap["model"])


def restore_rng(snap: Any) -> None:
    random.setstate(snap["rng_python"])
    numpy_rng.bit_generator.state = snap["rng_numpy"]
    torch.set_rng_state(snap["rng_torch"])
    for i, t in enumerate(snap["rng_cuda"]):
        torch.cuda.set_rng_state(t, i)


exp = Experiment()
exp.setup_optimizers()

# Get PRE params
restore_model_only(exp, snap)
pre_params = {n: p.data.clone() for n, p in exp.model.named_parameters()}

# Get POST params by running one step
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
restore_rng(snap)

trainloader = exp.make_train_loader()
samples, targets, puzzle_ids, valid_count = next(iter(trainloader))
exp.step(
    samples.to(exp.device),
    targets.to(exp.device),
    puzzle_ids.to(exp.device) if puzzle_ids is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
    valid_count,
)
post_params = {n: p.data.clone() for n, p in exp.model.named_parameters()}

# Compute deltas
deltas = {}
for n, pre_p in pre_params.items():
    deltas[n] = post_params[n] - pre_p
    print(f"  {n:50s}  delta_norm={deltas[n].float().norm().item():.6f}")

# PRE baseline
restore_model_only(exp, snap)
cell_acc, puzzle_acc, *_ = exp.evaluate(iter(exp.make_test_loader()))
print(f"\nPRE:  cell={cell_acc:.2f}% puzzle={puzzle_acc:.2f}%")

# POST baseline
for n, p in exp.model.named_parameters():
    p.data.copy_(post_params[n])
cell_acc, puzzle_acc, *_ = exp.evaluate(iter(exp.make_test_loader()))
print(f"POST: cell={cell_acc:.2f}% puzzle={puzzle_acc:.2f}%")

# Ablation: apply all deltas EXCEPT one param at a time
print("\n=== LEAVE-ONE-OUT ABLATION ===")
print(
    "(Apply all deltas except named param; if puzzle_acc recovers, that param caused drop)"
)
for skip_name in pre_params:
    for n, p in exp.model.named_parameters():
        if n == skip_name:
            p.data.copy_(pre_params[n])  # keep PRE value
        else:
            p.data.copy_(post_params[n])  # use POST value
    cell_acc, puzzle_acc, *_ = exp.evaluate(iter(exp.make_test_loader()))
    print(f"  skip {skip_name:50s}  cell={cell_acc:.2f}% puzzle={puzzle_acc:.2f}%")

# Ablation: apply ONLY one param's delta at a time
print("\n=== APPLY-ONE-ONLY ABLATION ===")
print(
    "(Apply only named param's delta; if puzzle_acc drops, that param alone causes drop)"
)
for apply_name in pre_params:
    for n, p in exp.model.named_parameters():
        if n == apply_name:
            p.data.copy_(post_params[n])  # use POST value
        else:
            p.data.copy_(pre_params[n])  # keep PRE value
    cell_acc, puzzle_acc, *_ = exp.evaluate(iter(exp.make_test_loader()))
    print(f"  only {apply_name:50s}  cell={cell_acc:.2f}% puzzle={puzzle_acc:.2f}%")

# Skip both qkv_proj weights
print("\n=== SKIP BOTH QKV ===")
both_qkv = {"reasoning.0.attn.qkv_proj.weight", "reasoning.1.attn.qkv_proj.weight"}
for n, p in exp.model.named_parameters():
    p.data.copy_(pre_params[n] if n in both_qkv else post_params[n])
cell_acc, puzzle_acc, *_ = exp.evaluate(iter(exp.make_test_loader()))
print(f"  skip both qkv:  cell={cell_acc:.2f}% puzzle={puzzle_acc:.2f}%")
