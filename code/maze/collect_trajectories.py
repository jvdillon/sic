"""Collect full 16-step ACT trajectories for maze error analysis.

Evaluates 4 checkpoints × 2 splits, saves per-cell logits and q_halt at each step.
"""

from collections import deque
from pathlib import Path

import pickle
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_puzzle_dataset


def load_model(exp_module_name: str, ckpt_path: str, device: torch.device):
    """Load experiment and checkpoint."""
    import importlib
    mod = importlib.import_module(f"maze.{exp_module_name}")
    exp = mod.Experiment()
    # Move model to target device if needed
    if exp.device != device:
        exp.model = exp.model.to(device)
        exp.device = device

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        exp.model.load_state_dict(ckpt["model"])
        if exp.ema and "ema" in ckpt:
            for n, v in ckpt["ema"].items():
                if n in exp.ema.shadow:
                    exp.ema.shadow[n].copy_(v)
    else:
        exp.model.load_state_dict(ckpt)
    return exp


def collect_trajectories(
    exp,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    max_steps: int = 16,
    batch_size: int = 128,
) -> dict:
    """Run max_steps ACT steps, collect logits and q_halt at each step."""
    ema = exp.ema
    use_ema = ema is not None
    if use_ema:
        ema.apply(exp.model)

    exp.model.eval()
    N = len(inputs)
    grid_len = inputs.shape[1]
    vocab_size = exp.config.vocab_size

    all_logits = np.zeros((N, max_steps, grid_len, vocab_size), dtype=np.float16)
    all_q_halt = np.zeros((N, max_steps), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            inp = inputs[start:end].to(device)
            B = inp.shape[0]

            z_H, z_L = exp._init_z(B)
            puzzle_ids = torch.zeros(B, device=device, dtype=torch.long)

            for step in range(max_steps):
                with torch.autocast(device_type=device.type, dtype=exp.dtype):
                    out = exp.model(inp, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                logits = out["logits"]  # [B, grid_len, vocab]
                q_halt = out["q_halt"]  # [B]

                all_logits[start:end, step] = logits.cpu().float().numpy().astype(np.float16)
                all_q_halt[start:end, step] = q_halt.cpu().float().numpy()

            if (end - 1) % 500 < batch_size:
                print(f"  {end}/{N}", flush=True)

    exp.model.train()
    if use_ema:
        ema.restore(exp.model)

    return {
        "logits": all_logits,
        "q_halt": all_q_halt,
        "inputs": inputs.cpu().numpy(),
        "labels": labels.cpu().numpy(),
    }


def analyze_maze_structure(inp: np.ndarray, lab: np.ndarray) -> dict:
    """Compute maze structural features: path length, wrong turns, depths."""
    grid = inp.reshape(30, 30)
    sol = lab.reshape(30, 30)

    starts = np.argwhere(grid == 3)
    ends = np.argwhere(grid == 4)
    if len(starts) == 0 or len(ends) == 0:
        return {"path_len": 0, "wrong_turns": 0, "max_wrong_depth": 0,
                "mean_wrong_depth": 0, "total_wrong_cells": 0,
                "start": (-1, -1), "end": (-1, -1)}

    start = tuple(starts[0])
    end = tuple(ends[0])
    solution_cells = set(map(tuple, np.argwhere(sol == 5)))
    solution_cells.add(start)
    solution_cells.add(end)

    # Wrong turn analysis: for each solution cell, count non-solution open neighbors
    wrong_turn_depths = []
    for r, c in solution_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 30 and 0 <= nc < 30 and grid[nr, nc] >= 2 and (nr, nc) not in solution_cells:
                # BFS to measure depth of this dead-end branch
                visited = set()
                q = deque([(nr, nc)])
                visited.add((nr, nc))
                while q:
                    wr, wc = q.popleft()
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        wnr, wnc = wr + ddr, wc + ddc
                        if (0 <= wnr < 30 and 0 <= wnc < 30
                                and (wnr, wnc) not in visited
                                and grid[wnr, wnc] >= 2
                                and (wnr, wnc) not in solution_cells):
                            visited.add((wnr, wnc))
                            q.append((wnr, wnc))
                wrong_turn_depths.append(len(visited))

    # Distance from start along solution path for each solution cell
    # BFS along solution path only
    path_dist = {}
    pq = deque([start])
    path_dist[start] = 0
    while pq:
        r, c = pq.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in solution_cells and (nr, nc) not in path_dist:
                path_dist[(nr, nc)] = path_dist[(r, c)] + 1
                pq.append((nr, nc))

    return {
        "path_len": len(solution_cells),
        "wrong_turns": len(wrong_turn_depths),
        "max_wrong_depth": max(wrong_turn_depths) if wrong_turn_depths else 0,
        "mean_wrong_depth": float(np.mean(wrong_turn_depths)) if wrong_turn_depths else 0,
        "total_wrong_cells": sum(wrong_turn_depths),
        "start": start,
        "end": end,
        "path_dist": path_dist,
        "wrong_turn_depths": wrong_turn_depths,
    }


def main():
    device = torch.device("cuda")
    out_path = Path(__file__).parent / "trajectory_data.pkl"

    # Configuration: (exp_module, ckpt_step, data_dir, split, n_puzzles, label)
    configs = [
        # x07b (augmented) - train: first 250 instances = rows 0..1999
        ("x07b", 9500, "/opt/scratch/datasets/maze-30x30-hard-1k-aug", "train", 2000, "x07b_train_9500"),
        ("x07b", 7500, "/opt/scratch/datasets/maze-30x30-hard-1k-aug", "train", 2000, "x07b_train_7500"),
        # x07b test
        ("x07b", 9500, "/opt/scratch/datasets/maze-30x30-hard-1k-aug", "test", 250, "x07b_test_9500"),
        ("x07b", 7500, "/opt/scratch/datasets/maze-30x30-hard-1k-aug", "test", 250, "x07b_test_7500"),
        # x07 (non-augmented) - train
        ("x07", 12500, "/opt/scratch/datasets/maze-30x30-hard-1k", "train", 250, "x07_train_12500"),
        ("x07", 6500, "/opt/scratch/datasets/maze-30x30-hard-1k", "train", 250, "x07_train_6500"),
        # x07 test
        ("x07", 12500, "/opt/scratch/datasets/maze-30x30-hard-1k", "test", 250, "x07_test_12500"),
        ("x07", 6500, "/opt/scratch/datasets/maze-30x30-hard-1k", "test", 250, "x07_test_6500"),
    ]

    results = {}

    for exp_name, step, data_dir, split, n_puzzles, label in configs:
        print(f"\n{'='*60}")
        print(f"Collecting: {label}")
        print(f"{'='*60}")

        ckpt_path = str(Path(__file__).parent / "ckpts" / exp_name / f"step{step:05d}.pt")
        print(f"  Checkpoint: {ckpt_path}")

        # Load data
        data = load_puzzle_dataset(data_dir, split)
        inputs = data["inputs"][:n_puzzles]
        labels = data["labels"][:n_puzzles]
        print(f"  Data: {len(inputs)} puzzles from {data_dir}/{split}")

        # Load model
        exp = load_model(exp_name, ckpt_path, device)
        print("  Model loaded")

        # Collect trajectories
        traj = collect_trajectories(exp, inputs, labels, device)

        # Compute maze structure for base instances
        group_indices = data["group_indices"]
        if split == "train" and "aug" in data_dir:
            # Augmented: analyze only base instances (every 8th puzzle)
            n_instances = min(250, n_puzzles // 8)
            structures = []
            for i in range(n_instances):
                base_idx = i * 8
                structures.append(analyze_maze_structure(
                    traj["inputs"][base_idx], traj["labels"][base_idx]
                ))
        else:
            structures = []
            for i in range(min(250, len(inputs))):
                structures.append(analyze_maze_structure(
                    traj["inputs"][i], traj["labels"][i]
                ))

        traj["maze_structures"] = structures
        traj["group_indices"] = group_indices[:n_puzzles + 1].numpy() if len(group_indices) > n_puzzles else group_indices.numpy()
        results[label] = traj

        # Quick summary
        final_preds = traj["logits"][:, -1].argmax(axis=-1)  # [N, 900]
        cell_acc = (final_preds == traj["labels"]).mean() * 100
        puzzle_acc = (final_preds == traj["labels"]).all(axis=-1).mean() * 100
        print(f"  Step 16: cell_acc={cell_acc:.2f}%, puzzle_acc={puzzle_acc:.2f}%")

        del exp
        torch.cuda.empty_cache()

    print(f"\nSaving to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
