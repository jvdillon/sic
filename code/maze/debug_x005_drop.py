"""Scan training steps, save full state before any big puzzle_acc drop."""

import pathlib
import random

from experiment import resume_from_checkpoint, train
from maze.x005 import Experiment
from util import numpy_rng

import torch


CKPT_PATH = pathlib.Path(__file__).parent / "ckpts" / "x005" / "step03500.pt"
SCAN_START = 3540
SCAN_END = 3600
DROP_THRESHOLD = 10.0  # percentage points
SAVE_DIR = pathlib.Path(__file__).parent / "ckpts" / "debug_x005_drop"


def save_full_state(exp):
    """Save everything needed to replay one step: model, optimizers, RNG, carries."""
    state = exp._state  # noqa: SLF001
    return {
        # Model + optimizers
        "model": {k: v.cpu().clone() for k, v in exp.model.state_dict().items()},
        "optimizer1": exp.optimizer1.state_dict(),
        "optimizer2": exp.optimizer2.state_dict(),
        "step": exp.current_step,
        # RNG
        "rng_python": random.getstate(),
        "rng_numpy": numpy_rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        # Training state (carries, active chains, etc)
        "z_H": state.z_H.cpu().clone(),
        "z_L": state.z_L.cpu().clone(),
        "batch_index": state.batch_index.cpu().clone(),
        "h_step": state.h_step.cpu().clone(),
        "inputs": state.inputs.cpu().clone(),
        "labels": state.labels.cpu().clone(),
        "puzzle_ids": state.puzzle_ids.cpu().clone(),
        "chain_indices": state.chain_indices.cpu().clone(),
        "active": state.active.cpu().clone(),
        # Pending sample queue
        "_pending_inputs": exp._pending_inputs.cpu().clone(),
        "_pending_labels": exp._pending_labels.cpu().clone(),
        "_pending_puzzle_ids": exp._pending_puzzle_ids.cpu().clone(),
        # x005-specific
        "_wrong_count": exp._wrong_count.cpu().clone(),
    }


exp = Experiment()
exp.setup_optimizers()
resume_from_checkpoint(exp, str(CKPT_PATH))

# Fast-forward to SCAN_START
exp.eval_every_steps = 999999
exp.total_train_steps = SCAN_START
train(exp)  # pyright: ignore[reportArgumentType]
exp.total_train_steps = SCAN_END + 1

prev_puzzle_acc = None
prev_snapshot = None
prev_step = None

for _ in range(SCAN_START, SCAN_END + 1):
    cell_acc, puzzle_acc = exp.evaluate(iter(exp.make_test_loader()))
    print(f"Step {exp.current_step:6d}  Test {cell_acc:5.2f}% / {puzzle_acc:5.2f}%")

    if prev_puzzle_acc is not None:
        drop = prev_puzzle_acc - puzzle_acc
        if drop > DROP_THRESHOLD:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVE_DIR / f"pre_step{prev_step:05d}.pt"
            torch.save(prev_snapshot, path)
            print(f"DROP {prev_puzzle_acc:.1f}% -> {puzzle_acc:.1f}% "
                  f"({-drop:+.1f}pp) at step {prev_step}->{exp.current_step}")
            print(f"Saved full state to {path}")
            break

    if exp.current_step >= SCAN_END:
        print("No drop found.")
        break

    # Save full state before next step
    prev_snapshot = save_full_state(exp)
    prev_step = exp.current_step
    prev_puzzle_acc = puzzle_acc

    trainloader = exp.make_train_loader()
    samples, targets, puzzle_ids, valid_count = next(iter(trainloader))
    exp.step(
        samples.to(exp.device),
        targets.to(exp.device),
        puzzle_ids.to(exp.device) if puzzle_ids is not None else None,
        valid_count,
    )
