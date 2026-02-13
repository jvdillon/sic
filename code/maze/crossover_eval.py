"""Crossover: evaluate x07 model on x07b's augmented training data."""

from pathlib import Path

import pickle
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maze.collect_trajectories import (
    analyze_maze_structure,
    collect_trajectories,
    load_model,
)

from data import load_puzzle_dataset


def main():
    device = torch.device("cuda")

    # Load x07 model @ step 12500
    ckpt_path = str(Path(__file__).parent / "ckpts" / "x07" / "step12500.pt")
    print(f"Loading x07 model from {ckpt_path}")
    exp = load_model("x07", ckpt_path, device)

    # Load augmented train data (same data x07b trains on)
    data_dir = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"
    data = load_puzzle_dataset(data_dir, "train")
    n_puzzles = 2000  # 250 instances × 8 augs
    inputs = data["inputs"][:n_puzzles]
    labels = data["labels"][:n_puzzles]
    print(f"Data: {len(inputs)} puzzles from {data_dir}/train")

    # Collect trajectories
    traj = collect_trajectories(exp, inputs, labels, device)

    # Compute maze structure for base instances (every 8th)
    structures = []
    for i in range(250):
        structures.append(analyze_maze_structure(
            traj["inputs"][i * 8], traj["labels"][i * 8]
        ))
    traj["maze_structures"] = structures
    traj["group_indices"] = data["group_indices"][:n_puzzles + 1].numpy()

    del exp
    torch.cuda.empty_cache()

    # --- Analysis ---
    logits = traj["logits"]  # [2000, 16, 900, vocab]
    labs = traj["labels"]     # [2000, 900]
    inps = traj["inputs"]     # [2000, 900]
    q_halt = traj["q_halt"]   # [2000, 16]
    N = len(logits)

    # Step-by-step accuracy
    print(f"\n{'='*70}")
    print(f"  x07 model on augmented train data: {N} puzzles (250 instances × 8 augs)")
    print(f"{'='*70}")
    print("\n  Step-by-step accuracy:")
    for step in range(16):
        preds = logits[:, step].argmax(axis=-1)
        cell_acc = (preds == labs).mean() * 100
        puzzle_acc = (preds == labs).all(axis=-1).mean() * 100
        avg_q = q_halt[:, step].mean()
        print(f"    H{step+1:2d}: cell={cell_acc:6.2f}%  puzzle={puzzle_acc:5.1f}%  "
              f"q_halt={avg_q:+.3f}")

    # Final step
    final_preds = logits[:, -1].argmax(axis=-1)
    wrong_mask = final_preds != labs

    # Per-augmentation breakdown
    print("\n--- Per-Augmentation Accuracy ---")
    for aug in range(8):
        idx = np.arange(aug, N, 8)
        preds_aug = final_preds[idx]
        labs_aug = labs[idx]
        cell_acc = (preds_aug == labs_aug).mean() * 100
        puzzle_acc = (preds_aug == labs_aug).all(axis=-1).mean() * 100
        n_perfect = (preds_aug == labs_aug).all(axis=-1).sum()
        print(f"  Aug {aug}: cell={cell_acc:6.2f}%  puzzle={puzzle_acc:5.1f}%  "
              f"({n_perfect}/250 perfect)")

    # Dihedral analysis: how many of 8 augs solved per instance?
    print("\n--- Dihedral Consistency ---")
    correct_per_puzzle = (final_preds == labs).all(axis=-1)  # [2000]
    correct_by_instance = correct_per_puzzle.reshape(250, 8)
    solved_counts = correct_by_instance.sum(axis=1)
    for k in range(9):
        n = (solved_counts == k).sum()
        if n > 0:
            print(f"  {k}/8 augs solved: {n} instances ({100*n/250:.1f}%)")

    # Error analysis
    print("\n--- Error Analysis ---")
    solution_mask = (inps == 2) & (labs == 5)
    sol_wrong = (wrong_mask & solution_mask).sum()
    nonsol_wrong = (wrong_mask & ~solution_mask).sum()
    print(f"  Solution cells wrong: {sol_wrong}/{solution_mask.sum()} "
          f"({100*sol_wrong/solution_mask.sum():.2f}%)")
    print(f"  Non-solution cells wrong: {nonsol_wrong}/{(~solution_mask).sum()} "
          f"({100*nonsol_wrong/(~solution_mask).sum():.2f}%)")

    # FP/FN
    fp = ((final_preds == 5) & (labs != 5)).sum()
    fn = ((labs == 5) & (final_preds != 5)).sum()
    print(f"  False positives (pred 5, not solution): {fp}")
    print(f"  False negatives (missed solution): {fn}")

    # Error count distribution
    errors_per_puzzle = wrong_mask.sum(axis=1)
    print(f"\n  Errors per puzzle: mean={errors_per_puzzle.mean():.1f}, "
          f"std={errors_per_puzzle.std():.1f}, "
          f"min={errors_per_puzzle.min()}, max={errors_per_puzzle.max()}")
    perfect = (errors_per_puzzle == 0).sum()
    print(f"  Perfect: {perfect}/{N} ({100*perfect/N:.1f}%)")

    # Wrong-turn proximity
    print("\n--- Wrong-Turn Proximity ---")
    near_branch = 0
    total_errors = 0
    for i in range(0, N, 8):  # aug 0 only for structure
        si = i // 8
        s = structures[si]
        inp = inps[i].reshape(30, 30)
        lab = labs[i].reshape(30, 30)
        pred = final_preds[i].reshape(30, 30)

        solution_cells = set(map(tuple, np.argwhere(lab == 5)))
        solution_cells.add(s["start"])
        solution_cells.add(s["end"])

        branch_points = set()
        for r, c in solution_cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<30 and 0<=nc<30 and inp[nr,nc] >= 2 and (nr,nc) not in solution_cells:
                    branch_points.add((r,c))
                    break

        for r in range(30):
            for c in range(30):
                if lab[r,c] == 5 and pred[r,c] != lab[r,c]:
                    total_errors += 1
                    if branch_points:
                        min_dist = min(abs(r-br)+abs(c-bc) for br,bc in branch_points)
                        if min_dist <= 3:
                            near_branch += 1

    if total_errors > 0:
        print(f"  Near branch (≤3): {near_branch}/{total_errors} "
              f"({100*near_branch/total_errors:.1f}%)")

    # Convergence H8→H16
    print("\n--- Convergence H8→H16 (solution cells) ---")
    preds_8 = logits[:, 7].argmax(axis=-1)
    preds_16 = final_preds
    correct_8 = preds_8 == labs
    correct_16 = preds_16 == labs
    stayed = (correct_8 & correct_16 & solution_mask).sum()
    improved = (~correct_8 & correct_16 & solution_mask).sum()
    regressed = (correct_8 & ~correct_16 & solution_mask).sum()
    stuck = (~correct_8 & ~correct_16 & solution_mask).sum()
    total = solution_mask.sum()
    print(f"  Stayed correct: {stayed} ({100*stayed/total:.2f}%)")
    print(f"  Improved:       {improved} ({100*improved/total:.2f}%)")
    print(f"  Regressed:      {regressed} ({100*regressed/total:.2f}%)")
    print(f"  Stuck wrong:    {stuck} ({100*stuck/total:.2f}%)")

    # Compare with x07b results if available
    pkl_path = Path(__file__).parent / "trajectory_data.pkl"
    if pkl_path.exists():
        print("\n--- Comparison: x07 vs x07b on augmented train data ---")
        with open(pkl_path, "rb") as f:
            prev = pickle.load(f)

        if "x07b_train_9500" in prev:
            x07b = prev["x07b_train_9500"]
            x07b_preds = x07b["logits"][:, -1].argmax(axis=-1)
            x07b_correct = (x07b_preds == x07b["labels"]).all(axis=-1)
            x07_correct = correct_per_puzzle

            both = (x07_correct & x07b_correct).sum()
            only_x07 = (x07_correct & ~x07b_correct).sum()
            only_x07b = (~x07_correct & x07b_correct).sum()
            neither = (~x07_correct & ~x07b_correct).sum()
            print(f"  Both correct:   {both} ({100*both/N:.1f}%)")
            print(f"  Only x07:       {only_x07} ({100*only_x07/N:.1f}%)")
            print(f"  Only x07b:      {only_x07b} ({100*only_x07b/N:.1f}%)")
            print(f"  Neither:        {neither} ({100*neither/N:.1f}%)")

            # Per-aug comparison
            print("\n  Per-aug comparison:")
            for aug in range(8):
                idx = np.arange(aug, N, 8)
                x07_acc = (x07_correct[idx]).mean() * 100
                x07b_acc = (x07b_correct[idx]).mean() * 100
                print(f"    Aug {aug}: x07={x07_acc:5.1f}%  x07b={x07b_acc:5.1f}%  "
                      f"Δ={x07_acc - x07b_acc:+.1f}pp")

    # Save crossover data
    out_path = Path(__file__).parent / "crossover_data.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(traj, f)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
