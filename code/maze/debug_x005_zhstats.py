"""Compare z_H dynamics between PRE and POST across ACT steps."""

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


def collect_zh_stats(exp):
    """Collect z_H norm, delta, cosine sim between consecutive steps."""
    exp.model.eval()
    all_stats = []  # per puzzle: list of dicts per step
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
            prev_z_H = z_H.clone()

            for step in range(MAX_STEPS):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]

                if step == 0:
                    prev_z_H = z_H.clone()
                    continue

                # Per-sample stats (average over seq positions)
                for i in range(min(valid_count, B)):
                    zh = z_H[i].float()  # [seq, hidden]
                    pzh = prev_z_H[i].float()
                    delta = zh - pzh
                    delta_norm = delta.norm(dim=-1).mean().item()
                    zh_norm = zh.norm(dim=-1).mean().item()
                    cos = torch.nn.functional.cosine_similarity(
                        zh.reshape(-1), pzh.reshape(-1), dim=0
                    ).item()

                    if len(all_stats) <= i + (batch[3] if step > 1 else 0):
                        pass  # handled below

                prev_z_H = z_H.clone()

    exp.model.train()
    return all_stats


def collect_zh_trajectory(exp):
    """Simpler: collect per-step z_H norms and deltas, aggregated."""
    exp.model.eval()
    step_norms = [[] for _ in range(MAX_STEPS)]
    step_deltas = [[] for _ in range(MAX_STEPS - 1)]
    step_cosines = [[] for _ in range(MAX_STEPS - 1)]
    step_logit_entropy = [[] for _ in range(MAX_STEPS)]

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
            prev_z_H = None

            for step in range(MAX_STEPS):
                with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
                    out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                logits = out["logits"]

                for i in range(valid_count):
                    zh = z_H[i].float()
                    step_norms[step].append(zh.norm(dim=-1).mean().item())

                    # Logit entropy
                    p = torch.softmax(logits[i].float(), dim=-1)
                    ent = -(p * p.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
                    step_logit_entropy[step].append(ent)

                    if prev_z_H is not None:
                        pzh = prev_z_H[i].float()
                        delta = zh - pzh
                        step_deltas[step - 1].append(delta.norm(dim=-1).mean().item())
                        cos = torch.nn.functional.cosine_similarity(
                            zh.reshape(-1), pzh.reshape(-1), dim=0
                        ).item()
                        step_cosines[step - 1].append(cos)

                prev_z_H = z_H.clone()

    exp.model.train()
    return step_norms, step_deltas, step_cosines, step_logit_entropy


exp = Experiment()
exp.setup_optimizers()

# PRE
print("Evaluating PRE...")
exp.model.load_state_dict(snap["model"])
pre_norms, pre_deltas, pre_cos, pre_ent = collect_zh_trajectory(exp)

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
post_norms, post_deltas, post_cos, post_ent = collect_zh_trajectory(exp)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


print("\n=== z_H NORM PER STEP ===")
print(f"{'step':>4s}  {'PRE_norm':>10s}  {'POST_norm':>10s}")
for s in range(MAX_STEPS):
    print(f"  {s+1:2d}    {mean(pre_norms[s]):8.3f}    {mean(post_norms[s]):8.3f}")

print("\n=== z_H DELTA NORM (step-to-step) ===")
print(f"{'step':>6s}  {'PRE_delta':>10s}  {'POST_delta':>10s}")
for s in range(MAX_STEPS - 1):
    print(f"  {s+1}->{s+2}    {mean(pre_deltas[s]):8.3f}    {mean(post_deltas[s]):8.3f}")

print("\n=== z_H COSINE SIM (step-to-step) ===")
print(f"{'step':>6s}  {'PRE_cos':>10s}  {'POST_cos':>10s}")
for s in range(MAX_STEPS - 1):
    print(f"  {s+1}->{s+2}    {mean(pre_cos[s]):8.5f}    {mean(post_cos[s]):8.5f}")

print("\n=== LOGIT ENTROPY PER STEP ===")
print(f"{'step':>4s}  {'PRE_ent':>10s}  {'POST_ent':>10s}")
for s in range(MAX_STEPS):
    print(f"  {s+1:2d}    {mean(pre_ent[s]):8.5f}    {mean(post_ent[s]):8.5f}")
