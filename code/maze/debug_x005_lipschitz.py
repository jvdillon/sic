"""Estimate local Lipschitz constant of the reasoning map PRE vs POST."""

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

N_PERTURBATIONS = 10
EPS = 0.1


def estimate_lipschitz(exp: Any, n_puzzles: int = 100) -> Any:
    """Estimate local Lipschitz of one ACT step around the step-2 z_H."""
    exp.model.eval()
    ratios_all = []
    puzzle_count = 0

    with torch.no_grad():
        for batch in exp.make_test_loader():
            if puzzle_count >= n_puzzles:
                break
            inputs = batch[0].to(exp.device)
            _labels = batch[1].to(exp.device)
            puzzle_ids = batch[2]
            if puzzle_ids is not None:
                puzzle_ids = puzzle_ids.to(exp.device)
            valid_count = batch[3]
            B = inputs.shape[0]

            # Run 2 steps to get to a representative z_H
            z_H, z_L = exp._init_z(B)  # noqa: SLF001
            for _ in range(2):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)  # noqa: SLF001
                z_H = out["z_H"]
                z_L = out["z_L"]

            # Now estimate Lipschitz at this z_H
            for i in range(min(valid_count, B)):
                if puzzle_count >= n_puzzles:
                    break
                zh_i = z_H[i : i + 1]  # [1, seq, hidden]
                zl_i = z_L[i : i + 1]
                inp_i = inputs[i : i + 1]
                pid_i = puzzle_ids[i : i + 1] if puzzle_ids is not None else None

                # Get f(z_H)
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out0 = exp._eval_forward(inp_i, zh_i, zl_i, pid_i)  # noqa: SLF001
                zh_out0 = out0["z_H"].float()

                ratios = []
                for _ in range(N_PERTURBATIONS):
                    # Random perturbation
                    delta = torch.randn_like(zh_i) * EPS
                    zh_pert = zh_i + delta

                    with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                        out_p = exp._eval_forward(inp_i, zh_pert, zl_i, pid_i)  # noqa: SLF001
                    zh_out_p = out_p["z_H"].float()

                    # ||f(z+d) - f(z)|| / ||d||
                    out_diff = (zh_out_p - zh_out0).norm().item()
                    in_diff = delta.float().norm().item()
                    ratios.append(out_diff / in_diff)

                ratios_all.append(max(ratios))  # worst-case ratio per puzzle
                puzzle_count += 1

    return ratios_all


torch._dynamo.reset()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

exp = Experiment()
exp.setup_optimizers()

# First apply the training step to get POST params
print("Applying training step to get POST params...")
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
post_params = {n: p.data.clone() for n, p in exp.model.named_parameters()}

# PRE
print("Estimating Lipschitz for PRE model...")
exp.model.load_state_dict(snap["model"])
pre_lip = estimate_lipschitz(exp)

# POST
print("Estimating Lipschitz for POST model...")
for n, p in exp.model.named_parameters():
    p.data.copy_(post_params[n])
post_lip = estimate_lipschitz(exp)

# Report
pre_lip.sort()
post_lip.sort()
n = len(pre_lip)


def stats(vals: Any) -> Any:
    s = sorted(vals)
    n = len(s)
    return (
        s[0],
        s[n // 4],
        s[n // 2],
        s[3 * n // 4],
        s[-1],
        sum(s) / n,
        sum(1 for v in s if v > 1.0),
    )


pre_s = stats(pre_lip)
post_s = stats(post_lip)

print(
    f"\n=== LOCAL LIPSCHITZ ESTIMATE (max over {N_PERTURBATIONS} perturbations, eps={EPS}) ==="
)
print(
    f"{'':>6s}  {'min':>6s}  {'p25':>6s}  {'p50':>6s}  {'p75':>6s}  {'max':>6s}  {'mean':>6s}  {'>1':>4s}"
)
print(
    f"  PRE   {pre_s[0]:6.3f}  {pre_s[1]:6.3f}  {pre_s[2]:6.3f}  {pre_s[3]:6.3f}  {pre_s[4]:6.3f}  {pre_s[5]:6.3f}  {pre_s[6]:4d}"
)
print(
    f"  POST  {post_s[0]:6.3f}  {post_s[1]:6.3f}  {post_s[2]:6.3f}  {post_s[3]:6.3f}  {post_s[4]:6.3f}  {post_s[5]:6.3f}  {post_s[6]:4d}"
)
