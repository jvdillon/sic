r"""Collect full ACT trajectories for puzzle error analysis.

Usage::

    python collect_trajectories.py \
        --type maze --split test --num_train 2000 --num_test 250 \
        --x 07b,7500 --x 07,2500,train

Each --x is: EXP,STEP[,SPLIT][,NUM][,TYPE]. Fields after STEP override
the corresponding global defaults.

TYPE is one of: maze, maze-aug, sudoku, arc.
The -aug suffix selects the augmented dataset variant.

Checkpoint path: <task>/ckpts/<exp>/step<STEP>.pt  (task = type without -aug)
Data dir: looked up from _DATA_DIRS[type]
Label: <exp>_<split>_<step>
"""

from collections import deque
from pathlib import Path
from typing import Any

import argparse
import importlib
import pickle
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_puzzle_dataset


_DATA_DIRS: dict[str, str] = {
    "maze": "/opt/scratch/datasets/maze-30x30-hard-1k",
    "maze-aug": "/opt/scratch/datasets/maze-30x30-hard-1k-aug",
    "sudoku": "/opt/scratch/datasets/sudoku-extreme-1k-aug-1000",
    "arc": "/opt/scratch/datasets/arc1concept-aug-1000",
}


def _task_from_type(type_str: str) -> str:
    """Strip -aug suffix to get the experiment module prefix."""
    return type_str.removesuffix("-aug")


def _parse_x(spec: str, defaults: argparse.Namespace) -> dict[str, Any]:
    """Parse an --x spec into a config dict.

    Format: EXP,STEP[,SPLIT][,NUM][,TYPE]
    """
    parts = spec.split(",")
    assert len(parts) >= 2, f"--x needs at least EXP,STEP: got {spec!r}"
    exp = parts[0]
    step = int(parts[1])
    split = parts[2] if len(parts) > 2 else defaults.split
    num = int(parts[3]) if len(parts) > 3 else None
    type_str = parts[4] if len(parts) > 4 else defaults.type
    if num is None:
        num = defaults.num_train if split == "train" else defaults.num_test
    return {
        "exp": exp,
        "step": step,
        "split": split,
        "num": num,
        "type": type_str,
    }


def load_model(
    task: str,
    exp_name: str,
    ckpt_path: str,
    device: torch.device,
) -> Any:
    """Load experiment and checkpoint."""
    mod = importlib.import_module(f"{task}.{exp_name}")
    exp = mod.Experiment()
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
    exp: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    max_steps: int = 16,
    batch_size: int = 128,
) -> dict[str, Any]:
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

            z_H, z_L = exp._init_z(B)  # noqa: SLF001
            puzzle_ids = torch.zeros(B, device=device, dtype=torch.long)

            for step in range(max_steps):
                with torch.autocast(device_type=device.type, dtype=exp.dtype):
                    out = exp.model(inp, z_H, z_L, puzzle_ids)
                z_H = out["z_H"]
                z_L = out["z_L"]
                logits = out["logits"]
                q_halt = out["q_halt"]

                all_logits[start:end, step] = (
                    logits.cpu().float().numpy().astype(np.float16)
                )
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


def analyze_maze_structure(inp: np.ndarray, lab: np.ndarray) -> dict[str, Any]:
    """Compute maze structural features: path length, wrong turns, depths."""
    grid = inp.reshape(30, 30)
    sol = lab.reshape(30, 30)

    starts = np.argwhere(grid == 3)
    ends = np.argwhere(grid == 4)
    if len(starts) == 0 or len(ends) == 0:
        return {
            "path_len": 0,
            "wrong_turns": 0,
            "max_wrong_depth": 0,
            "mean_wrong_depth": 0,
            "total_wrong_cells": 0,
            "start": (-1, -1),
            "end": (-1, -1),
        }

    start = tuple(starts[0])
    end = tuple(ends[0])
    solution_cells = set(map(tuple, np.argwhere(sol == 5)))
    solution_cells.add(start)
    solution_cells.add(end)

    wrong_turn_depths: list[int] = []
    for r, c in solution_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < 30
                and 0 <= nc < 30
                and grid[nr, nc] >= 2
                and (nr, nc) not in solution_cells
            ):
                visited: set[tuple[int, int]] = set()
                q: deque[tuple[int, int]] = deque([(nr, nc)])
                visited.add((nr, nc))
                while q:
                    wr, wc = q.popleft()
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        wnr, wnc = wr + ddr, wc + ddc
                        if (
                            0 <= wnr < 30
                            and 0 <= wnc < 30
                            and (wnr, wnc) not in visited
                            and grid[wnr, wnc] >= 2
                            and (wnr, wnc) not in solution_cells
                        ):
                            visited.add((wnr, wnc))
                            q.append((wnr, wnc))
                wrong_turn_depths.append(len(visited))

    path_dist: dict[tuple[int, int], int] = {}
    pq: deque[tuple[int, int]] = deque([start])
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
        "mean_wrong_depth": (
            float(np.mean(wrong_turn_depths)) if wrong_turn_depths else 0
        ),
        "total_wrong_cells": sum(wrong_turn_depths),
        "start": start,
        "end": end,
        "path_dist": path_dist,
        "wrong_turn_depths": wrong_turn_depths,
    }


def _compute_structures(
    task: str,
    traj: dict[str, Any],
    group_indices: torch.Tensor,
    n_puzzles: int,
) -> list[dict[str, Any]]:
    """Compute task-specific structural features for base instances."""
    if task != "maze":
        return []

    n_augs_per_instance = (
        int(group_indices[1] - group_indices[0]) if len(group_indices) > 1 else 1
    )
    if n_augs_per_instance > 1:
        n_instances = min(
            n_puzzles // n_augs_per_instance, len(traj["inputs"]) // n_augs_per_instance
        )
        return [
            analyze_maze_structure(
                traj["inputs"][i * n_augs_per_instance],
                traj["labels"][i * n_augs_per_instance],
            )
            for i in range(n_instances)
        ]
    return [
        analyze_maze_structure(traj["inputs"][i], traj["labels"][i])
        for i in range(min(n_puzzles, len(traj["inputs"])))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--type", default="maze", help="Puzzle type (default: maze)")
    parser.add_argument("--split", default="test", help="Default split (default: test)")
    parser.add_argument(
        "--num_train", type=int, default=250, help="Default num samples for train"
    )
    parser.add_argument(
        "--num_test", type=int, default=250, help="Default num samples for test"
    )
    parser.add_argument(
        "--x", action="append", required=True, help="EXP,STEP[,SPLIT][,NUM][,TYPE]"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle path (default: trajectory_data.pkl)",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    out_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "trajectory_data.pkl"
    )
    code_dir = Path(__file__).resolve().parent.parent

    configs = [_parse_x(spec, args) for spec in args.x]

    results: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}

    for cfg in configs:
        exp_name = cfg["exp"]
        step = cfg["step"]
        split = cfg["split"]
        n_puzzles = cfg["num"]
        type_str = cfg["type"]
        task = _task_from_type(type_str)
        label = f"{exp_name}_{split}_{step}"

        print(f"\n{'=' * 60}")
        print(f"Collecting: {label}  (type={type_str})")
        print(f"{'=' * 60}")

        data_dir = _DATA_DIRS[type_str]
        ckpt_path = str(code_dir / task / "ckpts" / exp_name / f"step{step:05d}.pt")
        print(f"  Checkpoint: {ckpt_path}")

        data = load_puzzle_dataset(data_dir, split)
        inputs = data["inputs"][:n_puzzles]
        labels = data["labels"][:n_puzzles]
        print(f"  Data: {len(inputs)} samples from {data_dir}/{split}")

        exp = load_model(task, exp_name, ckpt_path, device)
        print("  Model loaded")

        traj = collect_trajectories(exp, inputs, labels, device)

        group_indices = data["group_indices"]
        traj["maze_structures"] = _compute_structures(
            task,
            traj,
            group_indices,
            n_puzzles,
        )
        traj["group_indices"] = (
            group_indices[: n_puzzles + 1].numpy()
            if len(group_indices) > n_puzzles
            else group_indices.numpy()
        )
        results[label] = traj

        n_augs = (
            int(group_indices[1] - group_indices[0]) if len(group_indices) > 1 else 1
        )
        meta[label] = {
            "type": type_str,
            "task": task,
            "split": split,
            "exp": exp_name,
            "step": step,
            "n_augs": n_augs,
        }

        final_preds = traj["logits"][:, -1].argmax(axis=-1)
        cell_acc = (final_preds == traj["labels"]).mean() * 100
        puzzle_acc = (final_preds == traj["labels"]).all(axis=-1).mean() * 100
        print(f"  Final step: cell_acc={cell_acc:.2f}%, puzzle_acc={puzzle_acc:.2f}%")

        del exp
        torch.cuda.empty_cache()

    results["_meta"] = meta

    print(f"\nSaving to {out_path}...")
    with out_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
