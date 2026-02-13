"""Verify snapshot perfectly reproduces a training step.

Restores from snapshot twice, runs exp.step() each time, checks
all model params and training state match within tolerance.

Non-zero tolerance needed because bfloat16 matmul on CUDA has
inherent non-determinism in reduction order (~1e-4 max diff).
"""

from typing import Any

import os
import random
import sys


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


torch.use_deterministic_algorithms(True)

from maze.x005 import Experiment  # noqa: E402
from util import numpy_rng  # noqa: E402


SNAPSHOT_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else ("maze/ckpts/debug_x005_drop/pre_step03540.pt")
)
ATOL = 2e-4  # bfloat16 matmul non-determinism

snap = torch.load(SNAPSHOT_PATH, weights_only=False)
print(f"Loaded snapshot from {SNAPSHOT_PATH}, step={snap['step']}")


def restore_all(exp: Any, snap: Any) -> None:
    exp.model.load_state_dict(snap["model"])
    exp.optimizer1.load_state_dict(snap["optimizer1"])
    exp.optimizer2.load_state_dict(snap["optimizer2"])
    exp.current_step = snap["step"]
    exp.total_train_steps = snap["step"] + 10
    state = exp._state  # noqa: SLF001
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
    exp._wrong_count.copy_(snap["_wrong_count"].to(exp._wrong_count.device))  # noqa: SLF001
    exp._pending_inputs = snap["_pending_inputs"].to(exp.device)  # noqa: SLF001
    exp._pending_labels = snap["_pending_labels"].to(exp.device)  # noqa: SLF001
    exp._pending_puzzle_ids = snap["_pending_puzzle_ids"].to(exp.device)  # noqa: SLF001
    random.setstate(snap["rng_python"])
    numpy_rng.bit_generator.state = snap["rng_numpy"]
    torch.set_rng_state(snap["rng_torch"])
    for i, t in enumerate(snap["rng_cuda"]):
        torch.cuda.set_rng_state(t, i)


def run_one_step(exp: Any) -> dict[str, Any]:
    trainloader = exp.make_train_loader()
    samples, targets, puzzle_ids, valid_count = next(iter(trainloader))
    exp.step(
        samples.to(exp.device),
        targets.to(exp.device),
        puzzle_ids.to(exp.device) if puzzle_ids is not None else None,
        valid_count,
    )
    state = exp._state  # noqa: SLF001
    return {
        "params": {n: p.data.clone() for n, p in exp.model.named_parameters()},
        "z_H": state.z_H.clone(),
        "z_L": state.z_L.clone(),
        "h_step": state.h_step.clone(),
        "active": state.active.clone(),
        "batch_index": state.batch_index.clone(),
    }


exp = Experiment()
exp.setup_optimizers()

restore_all(exp, snap)
result_a = run_one_step(exp)

restore_all(exp, snap)
result_b = run_one_step(exp)

# Compare params
max_diff = 0.0
for name in result_a["params"]:
    diff = (
        (result_a["params"][name].float() - result_b["params"][name].float())
        .abs()
        .max()
        .item()
    )
    max_diff = max(max_diff, diff)
    if diff > ATOL:
        print(f"FAIL param {name}: max_diff={diff} > {ATOL}")

# Compare training state (integer tensors should be exact)
for key in ["h_step", "active", "batch_index"]:
    if not torch.equal(result_a[key], result_b[key]):
        print(f"FAIL state {key}: not equal")

# Compare carries (float, use tolerance)
for key in ["z_H", "z_L"]:
    diff = (result_a[key].float() - result_b[key].float()).abs().max().item()
    if diff > ATOL:
        print(f"FAIL state {key}: max_diff={diff} > {ATOL}")

if max_diff <= ATOL:
    print(f"PASS: max param diff={max_diff:.2e} <= {ATOL} (bf16 matmul noise)")
else:
    print(f"FAIL: max param diff={max_diff:.2e} > {ATOL}")
