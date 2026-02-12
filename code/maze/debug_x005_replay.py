"""Replay a bad step from saved full state with instrumentation."""

import random
import sys

from maze.x005 import Experiment
from torch import nn
from util import numpy_rng

import torch


SNAPSHOT_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "maze/ckpts/debug_x005_drop/pre_step03582.pt"
)


def restore_rng(snap):
    random.setstate(snap["rng_python"])
    numpy_rng.bit_generator.state = snap["rng_numpy"]
    torch.set_rng_state(snap["rng_torch"])
    for i, t in enumerate(snap["rng_cuda"]):
        torch.cuda.set_rng_state(t, i)


def restore_training_state(exp, snap):
    state = exp._state  # noqa: SLF001
    state.z_H.copy_(snap["z_H"].to(state.z_H.device))
    state.z_L.copy_(snap["z_L"].to(state.z_L.device))
    state.batch_index.copy_(snap["batch_index"].to(state.batch_index.device))
    state.h_step.copy_(snap["h_step"].to(state.h_step.device))
    state.inputs.copy_(snap["inputs"].to(state.inputs.device))
    state.labels.copy_(snap["labels"].to(state.labels.device))
    state.puzzle_ids.copy_(snap["puzzle_ids"].to(state.puzzle_ids.device))
    state.chain_indices.copy_(snap["chain_indices"].to(state.chain_indices.device))
    state.active.copy_(snap["active"].to(state.active.device))
    exp._wrong_count.copy_(snap["_wrong_count"].to(exp._wrong_count.device))
    exp._pending_inputs = snap["_pending_inputs"].to(exp.device)
    exp._pending_labels = snap["_pending_labels"].to(exp.device)
    exp._pending_puzzle_ids = snap["_pending_puzzle_ids"].to(exp.device)


# Load snapshot
snap = torch.load(SNAPSHOT_PATH, weights_only=False)
print(f"Loaded snapshot from {SNAPSHOT_PATH}, step={snap['step']}")

# Create experiment and restore everything
exp = Experiment()
exp.setup_optimizers()
exp.model.load_state_dict(snap["model"])
exp.optimizer1.load_state_dict(snap["optimizer1"])
exp.optimizer2.load_state_dict(snap["optimizer2"])
exp.current_step = snap["step"]
restore_training_state(exp, snap)
restore_rng(snap)

# Pre-step eval
cell_acc, puzzle_acc = exp.evaluate(iter(exp.make_test_loader()))
print(f"Step {exp.current_step:6d}  Test {cell_acc:5.2f}% / {puzzle_acc:5.2f}%  (PRE)")

# Restore RNG again (eval consumed it)
restore_rng(snap)

# ── Instrumented step ──
state = exp._state  # noqa: SLF001

pre_params = {
    n: p.data.detach().clone() for n, p in exp.model.named_parameters()
    if p.requires_grad
}

print("\n=== PRE-STEP PARAM NORMS ===")
for name, p in exp.model.named_parameters():
    if p.requires_grad:
        print(f"  {name:50s}  norm={p.data.float().norm().item():10.4f}")

print("\n=== PRE-STEP MUON MOMENTUM BUFFERS ===")
for group in exp.optimizer2.param_groups:
    for p in group["params"]:
        s = exp.optimizer2.state.get(p)
        if s and "momentum_buffer" in s:
            buf = s["momentum_buffer"]
            print(f"  shape={list(p.shape)!s:30s}"
                  f"  buf_norm={buf.float().norm().item():10.4f}"
                  f"  buf_max={buf.float().abs().max().item():10.6f}")

# Active state
active_mask = state.batch_index >= 0
print("\n=== PRE-STEP STATE ===")
print(f"  active chains: {active_mask.sum().item()}")
print(f"  h_step: {state.h_step[state.active].tolist()}")

valid_keys = [k for k in exp.max_steps_schedule if k <= exp.current_step]
max_h_steps, train_q_halt = exp.max_steps_schedule[max(valid_keys)]
print(f"  max_h_steps={max_h_steps}, train_q_halt={train_q_halt}")

# Training batch
trainloader = exp.make_train_loader()
samples, targets, puzzle_ids, valid_count = next(iter(trainloader))
print("\n=== TRAINING BATCH ===")
print(f"  samples shape: {list(samples.shape)}, valid_count: {valid_count}")
print(f"  sample hash: {samples.sum().item():.6f}")

if exp.augment_sudoku:
    from experiment import augment_sudoku
    samples, targets = augment_sudoku(samples, targets)

exp._enqueue(  # noqa: SLF001
    samples[:valid_count].to(exp.device),
    targets[:valid_count].to(exp.device),
    puzzle_ids[:valid_count].to(exp.device) if puzzle_ids is not None else None,
)
exp._fill_pending()  # noqa: SLF001

# Forward
forward_result = exp._forward(state)  # noqa: SLF001
active_samples, winner_chains = state.select_winners(forward_result["losses"])

print("\n=== FORWARD RESULT ===")
print(f"  active_samples: {len(active_samples)}")
print(f"  winner_chains: {winner_chains.tolist()[:10]}...")
print(f"  losses (winners): {forward_result['losses'][winner_chains].tolist()[:10]}...")
print(f"  q_halt (winners): {forward_result['q_halt'][winner_chains].tolist()[:10]}...")
print(f"  loss mean: {forward_result['losses'][winner_chains].mean().item():.6f}")
print(f"  loss sum/bs: {forward_result['losses'][winner_chains].sum().item() / exp.batch_size:.6f}")

with torch.no_grad():
    predictions = forward_result["logits"][winner_chains].argmax(dim=-1)
    correct = (predictions == state.labels[active_samples]).all(dim=-1).float()
    n_correct = int(correct.sum().item())
    n_total = len(correct)
print(f"  puzzles correct: {n_correct}/{n_total} ({100*n_correct/max(n_total,1):.1f}%)")

# Backward
assert exp.device is not None
with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):
    loss = forward_result["losses"][winner_chains].sum()
    if train_q_halt:
        loss = (
            loss
            + exp.q_halt_weight
            * nn.functional.binary_cross_entropy_with_logits(
                forward_result["q_halt"][winner_chains],
                correct,
                reduction="sum",
            )
        )
    total_loss = loss / exp.batch_size
print("\n=== LOSS ===")
print(f"  total_loss: {total_loss.item():.6f}")
total_loss.backward()

print("\n=== GRADIENTS (before clip) ===")
total_norm_sq = 0.0
for name, p in exp.model.named_parameters():
    if p.grad is not None:
        gnorm = p.grad.float().norm().item()
        gmax = p.grad.float().abs().max().item()
        total_norm_sq += gnorm ** 2
        print(f"  {name:50s}  grad_norm={gnorm:10.6f}  grad_max={gmax:10.6f}")
print(f"  {'TOTAL GRAD NORM':50s}  {total_norm_sq**0.5:10.6f}")

if exp.grad_clip_max_norm is not None:
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        exp.model.parameters(), max_norm=exp.grad_clip_max_norm
    )
    print("\n=== GRAD CLIP ===")
    print(f"  pre-clip norm: {pre_clip_norm.item():.6f}")
    print(f"  max_norm: {exp.grad_clip_max_norm}")
    print(f"  clipped: {pre_clip_norm.item() > exp.grad_clip_max_norm}")

lr_scale = exp._lr_scale()  # noqa: SLF001
print("\n=== LR ===")
print(f"  lr_scale: {lr_scale:.6f}")
for i, opt in enumerate([exp.optimizer1, exp.optimizer2]):
    for j, pg in enumerate(opt.param_groups):
        pg["lr"] = pg["initial_lr"] * lr_scale
        print(f"  opt{i+1} group{j} initial_lr={pg['initial_lr']:.6f}  effective_lr={pg['lr']:.6f}")

exp.optimizer1.step()
exp.optimizer2.step()
exp.model.zero_grad(set_to_none=True)
exp.current_step += 1

print("\n=== POST-STEP MUON MOMENTUM BUFFERS ===")
for group in exp.optimizer2.param_groups:
    for p in group["params"]:
        s = exp.optimizer2.state.get(p)
        if s and "momentum_buffer" in s:
            buf = s["momentum_buffer"]
            print(f"  shape={list(p.shape)!s:30s}"
                  f"  buf_norm={buf.float().norm().item():10.4f}"
                  f"  buf_max={buf.float().abs().max().item():10.6f}")

print("\n=== PARAM DELTAS ===")
max_delta_name = ""
max_delta_val = 0.0
for name, p in exp.model.named_parameters():
    if p.requires_grad and name in pre_params:
        delta = (p.data.float() - pre_params[name].float()).norm().item()
        rel = delta / (pre_params[name].float().norm().item() + 1e-8)
        print(f"  {name:50s}  delta_norm={delta:10.6f}  rel={rel:10.6f}")
        if delta > max_delta_val:
            max_delta_val = delta
            max_delta_name = name
print(f"  LARGEST DELTA: {max_delta_name}  {max_delta_val:.6f}")

# Carry + halt
if len(active_samples) > 0:
    chains = state.chain_indices[active_samples]
    valid = chains >= 0
    valid_chains = chains[valid]
    state.z_H[valid_chains] = forward_result["z_H"][valid_chains].detach()
    state.z_L[valid_chains] = forward_result["z_L"][valid_chains].detach()
    state.h_step[active_samples] += 1

num_halted = 0
if len(active_samples) > 0:
    h_steps = state.h_step[active_samples]
    at_max = h_steps >= max_h_steps
    if train_q_halt:
        q_halt_positive = forward_result["q_halt"][winner_chains] > 0
        force_continue = (
            torch.rand(len(h_steps), device=state.device)
            < exp.halt_exploration_prob
        )
        min_random_steps = torch.randint(
            low=2, high=max(3, max_h_steps + 1),
            size=(len(h_steps),), device=state.device,
        )
        halt = (at_max | q_halt_positive) & (~force_continue | (h_steps >= min_random_steps))
    else:
        halt = at_max
    num_halted = int(halt.sum().item())
    if halt.any():
        state.release(active_samples[halt])

print("\n=== HALTING ===")
print(f"  num_halted: {num_halted}")

state.expunge(forward_result["losses"].detach(), exp.expunge_threshold, exp.min_expunge_step)
exp._fill_pending()  # noqa: SLF001

# Post-step eval
cell_acc, puzzle_acc = exp.evaluate(iter(exp.make_test_loader()))
print(f"\n{'='*80}")
print(f"Step {exp.current_step:6d}  Test {cell_acc:5.2f}% / {puzzle_acc:5.2f}%  (POST)")
