"""Analyze maze trajectory data collected by collect_trajectories.py."""

from collections import defaultdict
from pathlib import Path

import pickle

import numpy as np


def load_data():
    path = Path(__file__).parent / "trajectory_data.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def basic_stats(data: dict, label: str):
    """Per-step accuracy trajectory and q_halt stats."""
    logits = data["logits"]  # [N, 16, 900, vocab]
    labels = data["labels"]  # [N, 900]
    q_halt = data["q_halt"]  # [N, 16]
    inputs = data["inputs"]  # [N, 900]
    N = len(logits)

    print(f"\n{'='*70}")
    print(f"  {label}: {N} puzzles")
    print(f"{'='*70}")

    # Per-step accuracy
    print("\n  Step-by-step accuracy:")
    for step in range(16):
        preds = logits[:, step].argmax(axis=-1)  # [N, 900]
        cell_correct = (preds == labels)
        cell_acc = cell_correct.mean() * 100
        puzzle_acc = cell_correct.all(axis=-1).mean() * 100
        avg_q = q_halt[:, step].mean()
        halt_rate = (q_halt[:, step] > 0).mean() * 100
        print(f"    H{step+1:2d}: cell={cell_acc:6.2f}%  puzzle={puzzle_acc:5.1f}%  "
              f"q_halt={avg_q:+.3f}  halt={halt_rate:5.1f}%")

    return logits, labels, inputs, q_halt


def error_analysis(data: dict, label: str):
    """Deep dive into what cells are wrong and why."""
    logits = data["logits"]  # [N, 16, 900, vocab]
    labels = data["labels"]  # [N, 900]
    inputs = data["inputs"]  # [N, 900]
    N = len(logits)

    # Final step predictions
    final_preds = logits[:, -1].argmax(axis=-1)  # [N, 900]
    wrong_mask = final_preds != labels  # [N, 900]

    # Solution cells: where input=2 and label=5 (path to find)
    solution_mask = (inputs == 2) & (labels == 5)
    # Non-solution cells: everything else
    nonsolution_mask = ~solution_mask

    print(f"\n--- Error Analysis: {label} ---")

    # 1. Errors on solution vs non-solution cells
    sol_wrong = wrong_mask & solution_mask
    nonsol_wrong = wrong_mask & nonsolution_mask
    n_sol = solution_mask.sum()
    n_nonsol = nonsolution_mask.sum()
    print(f"\n  Solution cells: {sol_wrong.sum()}/{n_sol} wrong "
          f"({100*sol_wrong.sum()/max(n_sol,1):.2f}%)")
    print(f"  Non-solution cells: {nonsol_wrong.sum()}/{n_nonsol} wrong "
          f"({100*nonsol_wrong.sum()/max(n_nonsol,1):.2f}%)")

    # 2. Error types on solution cells
    if sol_wrong.sum() > 0:
        wrong_preds_on_sol = final_preds[sol_wrong]
        wrong_labels_on_sol = labels[sol_wrong]
        print("\n  What model predicts on WRONG solution cells:")
        for v in range(6):
            count = (wrong_preds_on_sol == v).sum()
            if count > 0:
                print(f"    Predicted {v}: {count} ({100*count/len(wrong_preds_on_sol):.1f}%)")
        print("    (True label should be 5=solution path)")

    # 3. Error types on non-solution cells
    if nonsol_wrong.sum() > 0:
        wrong_preds_nonsol = final_preds[nonsol_wrong]
        wrong_labels_nonsol = labels[nonsol_wrong]
        print("\n  What model predicts on WRONG non-solution cells:")
        for v in range(6):
            pred_count = (wrong_preds_nonsol == v).sum()
            true_count = (wrong_labels_nonsol == v).sum()
            if pred_count > 0 or true_count > 0:
                print(f"    Predicted {v}: {pred_count}, True {v}: {true_count}")

    # 4. Logit entropy on wrong vs correct cells
    final_logits = logits[:, -1]  # [N, 900, vocab]
    probs = np.exp(final_logits - final_logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=-1)  # [N, 900]

    correct_mask = ~wrong_mask
    if wrong_mask.sum() > 0 and correct_mask.sum() > 0:
        print(f"\n  Entropy: wrong={entropy[wrong_mask].mean():.3f}, "
              f"correct={entropy[correct_mask].mean():.3f}")

    # 5. Per-puzzle error count distribution
    errors_per_puzzle = wrong_mask.sum(axis=1)
    print(f"\n  Errors per puzzle: mean={errors_per_puzzle.mean():.1f}, "
          f"std={errors_per_puzzle.std():.1f}, "
          f"min={errors_per_puzzle.min()}, max={errors_per_puzzle.max()}")
    perfect = (errors_per_puzzle == 0).sum()
    print(f"  Perfect puzzles: {perfect}/{N} ({100*perfect/N:.1f}%)")

    # Distribution buckets
    for threshold in [0, 1, 2, 5, 10, 20, 50, 100]:
        count = (errors_per_puzzle <= threshold).sum()
        print(f"    ≤{threshold:3d} errors: {count}/{N} ({100*count/N:.1f}%)")

    return wrong_mask, solution_mask, errors_per_puzzle


def spatial_analysis(data: dict, label: str):
    """Where on the 30x30 grid do errors cluster?"""
    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    wrong_mask = (final_preds != labels).reshape(N, 30, 30)
    solution_mask = ((inputs == 2) & (labels == 5)).reshape(N, 30, 30)

    # Error heatmap (average across puzzles)
    error_rate = wrong_mask.mean(axis=0)  # [30, 30]
    sol_rate = solution_mask.mean(axis=0)  # [30, 30]

    print(f"\n--- Spatial Analysis: {label} ---")

    # Errors by position (border vs interior)
    border_mask = np.zeros((30, 30), dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = border_mask[:, -1] = True
    interior_mask = ~border_mask

    border_err = wrong_mask[:, border_mask].mean()
    interior_err = wrong_mask[:, interior_mask].mean()
    print(f"  Border error rate: {border_err*100:.2f}%")
    print(f"  Interior error rate: {interior_err*100:.2f}%")

    # Error rate by quadrant
    for qr, qc, name in [(slice(0,15), slice(0,15), "top-left"),
                          (slice(0,15), slice(15,30), "top-right"),
                          (slice(15,30), slice(0,15), "bottom-left"),
                          (slice(15,30), slice(15,30), "bottom-right")]:
        err = wrong_mask[:, qr, qc].mean()
        print(f"  {name}: {err*100:.2f}%")

    # Print compact error heatmap (5x5 blocks)
    print("\n  Error heatmap (6x6 blocks, % wrong):")
    for br in range(6):
        row_str = "    "
        for bc in range(6):
            block_err = error_rate[br*5:(br+1)*5, bc*5:(bc+1)*5].mean() * 100
            row_str += f"{block_err:5.1f} "
        print(row_str)


def dihedral_analysis(data: dict, label: str, n_augs: int = 8):
    """Are errors consistent across dihedral augmentations?"""
    logits = data["logits"]
    labels = data["labels"]
    N = len(logits)

    if N % n_augs != 0:
        print(f"\n--- Dihedral Analysis: {label} SKIPPED (N={N} not divisible by {n_augs}) ---")
        return

    n_instances = N // n_augs
    final_preds = logits[:, -1].argmax(axis=-1)  # [N, 900]
    correct = (final_preds == labels).all(axis=-1)  # [N]
    correct_by_instance = correct.reshape(n_instances, n_augs)

    print(f"\n--- Dihedral Analysis: {label} ({n_instances} instances × {n_augs} augs) ---")

    # Per-instance: how many of 8 augs are solved?
    solved_counts = correct_by_instance.sum(axis=1)  # [n_instances]
    for k in range(n_augs + 1):
        n = (solved_counts == k).sum()
        if n > 0:
            print(f"  {k}/{n_augs} augs solved: {n} instances ({100*n/n_instances:.1f}%)")

    # Are errors correlated across augs?
    all_solved = (solved_counts == n_augs).sum()
    none_solved = (solved_counts == 0).sum()
    partial = n_instances - all_solved - none_solved
    print(f"\n  All 8 solved: {all_solved}, None solved: {none_solved}, Partial: {partial}")

    # Per-cell analysis for partially-solved instances
    if partial > 0:
        partial_idx = np.where((solved_counts > 0) & (solved_counts < n_augs))[0]
        # For these instances, count how many cells are wrong across augs
        cell_wrong_counts = []
        for idx in partial_idx[:20]:  # first 20
            base = idx * n_augs
            wrong_per_aug = []
            for aug in range(n_augs):
                preds = final_preds[base + aug]
                labs = labels[base + aug]
                wrong_per_aug.append((preds != labs).sum())
            cell_wrong_counts.append(wrong_per_aug)
        cell_wrong_counts = np.array(cell_wrong_counts)
        print(f"\n  Wrong cells per aug for {len(cell_wrong_counts)} partial instances:")
        print(f"    Mean errors per aug: {cell_wrong_counts.mean(axis=1)}")
        print(f"    Std across augs: {cell_wrong_counts.std(axis=1).mean():.1f}")


def backtracking_analysis(data: dict, label: str):
    """Correlate errors with maze backtracking difficulty."""
    structures = data.get("maze_structures", [])
    if not structures:
        print(f"\n--- Backtracking Analysis: {label} SKIPPED (no structure data) ---")
        return

    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    errors_per_puzzle = (final_preds != labels).sum(axis=1)

    # For augmented datasets, map puzzle index to structure index
    n_structs = len(structures)
    if n_structs < N:
        # Augmented: 8 puzzles per instance
        n_augs = N // n_structs
        struct_idx = np.arange(N) // n_augs
    else:
        struct_idx = np.arange(N)

    print(f"\n--- Backtracking Analysis: {label} ---")

    # Group by error count, show structure stats
    perfect = errors_per_puzzle == 0
    imperfect = errors_per_puzzle > 0

    if imperfect.sum() > 0 and perfect.sum() > 0:
        perf_structs = [structures[struct_idx[i]] for i in np.where(perfect)[0] if struct_idx[i] < len(structures)]
        imp_structs = [structures[struct_idx[i]] for i in np.where(imperfect)[0] if struct_idx[i] < len(structures)]

        for metric in ["path_len", "wrong_turns", "max_wrong_depth", "mean_wrong_depth", "total_wrong_cells"]:
            perf_vals = [s[metric] for s in perf_structs]
            imp_vals = [s[metric] for s in imp_structs]
            if perf_vals and imp_vals:
                print(f"  {metric}:")
                print(f"    Perfect: mean={np.mean(perf_vals):.1f} ± {np.std(perf_vals):.1f}")
                print(f"    Imperfect: mean={np.mean(imp_vals):.1f} ± {np.std(imp_vals):.1f}")

    # Correlation between error count and structure metrics
    all_vals = {}
    for metric in ["path_len", "wrong_turns", "max_wrong_depth", "total_wrong_cells"]:
        vals = np.array([structures[min(struct_idx[i], len(structures)-1)][metric] for i in range(N)])
        corr = np.corrcoef(errors_per_puzzle, vals)[0, 1]
        all_vals[metric] = vals
        print(f"  Correlation(errors, {metric}): {corr:.3f}")


def convergence_analysis(data: dict, label: str):
    """How do errors evolve over the 16 steps?"""
    logits = data["logits"]  # [N, 16, 900, vocab]
    labels = data["labels"]  # [N, 900]
    inputs = data["inputs"]
    N = len(logits)

    solution_mask = (inputs == 2) & (labels == 5)  # [N, 900]

    print(f"\n--- Convergence Analysis: {label} ---")

    # Track cells that are correct at step 8 but wrong at step 16 (regression)
    preds_8 = logits[:, 7].argmax(axis=-1)
    preds_16 = logits[:, -1].argmax(axis=-1)

    correct_at_8 = preds_8 == labels
    correct_at_16 = preds_16 == labels

    regressed = correct_at_8 & ~correct_at_16  # correct->wrong
    improved = ~correct_at_8 & correct_at_16   # wrong->correct
    stuck_wrong = ~correct_at_8 & ~correct_at_16  # wrong->wrong
    stayed_right = correct_at_8 & correct_at_16  # correct->correct

    print("  H8→H16 transitions (solution cells only):")
    sol_regressed = (regressed & solution_mask).sum()
    sol_improved = (improved & solution_mask).sum()
    sol_stuck = (stuck_wrong & solution_mask).sum()
    sol_stayed = (stayed_right & solution_mask).sum()
    sol_total = solution_mask.sum()
    print(f"    Stayed correct: {sol_stayed} ({100*sol_stayed/sol_total:.2f}%)")
    print(f"    Improved:       {sol_improved} ({100*sol_improved/sol_total:.2f}%)")
    print(f"    Regressed:      {sol_regressed} ({100*sol_regressed/sol_total:.2f}%)")
    print(f"    Stuck wrong:    {sol_stuck} ({100*sol_stuck/sol_total:.2f}%)")

    # At which step do stuck-wrong cells first go wrong?
    if stuck_wrong.sum() > 0:
        stuck_positions = np.argwhere(stuck_wrong & solution_mask)
        # For first 100 stuck cells, track when they first predicted correctly
        sample = stuck_positions[:200]
        first_correct_step = []
        never_correct = 0
        for puzzle_idx, cell_idx in sample:
            ever_correct = False
            for step in range(16):
                pred = logits[puzzle_idx, step, cell_idx].argmax()
                if pred == labels[puzzle_idx, cell_idx]:
                    first_correct_step.append(step)
                    ever_correct = True
                    break
            if not ever_correct:
                never_correct += 1
        if first_correct_step:
            print(f"\n  Stuck-wrong solution cells (sample of {len(sample)}):")
            print(f"    Never correct across all 16 steps: {never_correct}/{len(sample)}")
            if first_correct_step:
                print(f"    First correct at step: mean={np.mean(first_correct_step):.1f}")


def cross_checkpoint_analysis(data_early: dict, data_late: dict, label: str):
    """Compare errors between early and late checkpoint."""
    logits_e = data_early["logits"]
    logits_l = data_late["logits"]
    labels = data_early["labels"]  # same puzzles

    preds_e = logits_e[:, -1].argmax(axis=-1)
    preds_l = logits_l[:, -1].argmax(axis=-1)

    correct_e = preds_e == labels
    correct_l = preds_l == labels

    puzzle_correct_e = correct_e.all(axis=-1)
    puzzle_correct_l = correct_l.all(axis=-1)

    print(f"\n--- Cross-Checkpoint: {label} ---")
    both = (puzzle_correct_e & puzzle_correct_l).sum()
    only_early = (puzzle_correct_e & ~puzzle_correct_l).sum()
    only_late = (~puzzle_correct_e & puzzle_correct_l).sum()
    neither = (~puzzle_correct_e & ~puzzle_correct_l).sum()
    N = len(labels)
    print(f"  Both correct: {both} ({100*both/N:.1f}%)")
    print(f"  Only early:   {only_early} ({100*only_early/N:.1f}%)")
    print(f"  Only late:    {only_late} ({100*only_late/N:.1f}%)")
    print(f"  Neither:      {neither} ({100*neither/N:.1f}%)")

    # Cell-level
    cell_both = (correct_e & correct_l).sum()
    cell_only_e = (correct_e & ~correct_l).sum()
    cell_only_l = (~correct_e & correct_l).sum()
    cell_neither = (~correct_e & ~correct_l).sum()
    total = labels.size
    print("\n  Cell-level:")
    print(f"    Both correct: {cell_both} ({100*cell_both/total:.2f}%)")
    print(f"    Only early:   {cell_only_e} ({100*cell_only_e/total:.2f}%)")
    print(f"    Only late:    {cell_only_l} ({100*cell_only_l/total:.2f}%)")
    print(f"    Neither:      {cell_neither} ({100*cell_neither/total:.2f}%)")


def solution_path_position_analysis(data: dict, label: str):
    """Are errors correlated with position along the solution path?"""
    structures = data.get("maze_structures", [])
    if not structures:
        return

    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    wrong_mask = final_preds != labels

    n_structs = len(structures)
    n_augs = max(1, N // n_structs)

    print(f"\n--- Solution Path Position Analysis: {label} ---")

    # For each puzzle, categorize errors by distance from start along solution
    dist_bins = [0, 10, 20, 30, 50, 70, 100, 150, 200]
    bin_wrong = defaultdict(int)
    bin_total = defaultdict(int)

    for i in range(min(N, n_structs * n_augs)):
        si = min(i // n_augs, len(structures) - 1)
        s = structures[si]
        path_dist = s.get("path_dist", {})
        if not path_dist:
            continue

        inp = inputs[i].reshape(30, 30)
        lab = labels[i].reshape(30, 30)
        pred = final_preds[i].reshape(30, 30)

        # For augmented puzzles, we can only use structure from aug 0
        # For aug 0, the coordinates match directly
        if n_augs > 1 and (i % n_augs) != 0:
            continue

        for (r, c), dist in path_dist.items():
            if lab[r, c] == 5:  # solution cell
                # Find which distance bin
                b = 0
                for j, threshold in enumerate(dist_bins):
                    if dist >= threshold:
                        b = j
                bin_total[b] += 1
                if pred[r, c] != lab[r, c]:
                    bin_wrong[b] += 1

    print("  Error rate by distance from start along solution path:")
    for b, threshold in enumerate(dist_bins):
        t = bin_total.get(b, 0)
        w = bin_wrong.get(b, 0)
        if t > 0:
            next_t = dist_bins[b+1] if b+1 < len(dist_bins) else "+"
            print(f"    dist {threshold:3d}-{next_t}: {w}/{t} wrong ({100*w/max(t,1):.1f}%)")


def wrong_turn_proximity_analysis(data: dict, label: str):
    """Are errors near wrong-turn branch points?"""
    structures = data.get("maze_structures", [])
    if not structures:
        return

    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    n_structs = len(structures)
    n_augs = max(1, N // n_structs)

    print(f"\n--- Wrong-Turn Proximity Analysis: {label} ---")

    # For each solution cell error, measure distance to nearest wrong-turn branch point
    near_branch = 0
    far_branch = 0
    total_errors = 0

    for i in range(0, min(N, n_structs * n_augs), n_augs):  # aug 0 only
        si = i // n_augs
        if si >= len(structures):
            break
        s = structures[si]
        path_dist = s.get("path_dist", {})
        if not path_dist:
            continue

        inp = inputs[i].reshape(30, 30)
        lab = labels[i].reshape(30, 30)
        pred = final_preds[i].reshape(30, 30)

        # Find branch points: solution cells adjacent to non-solution open cells
        solution_cells = set(map(tuple, np.argwhere(lab == 5)))
        start = s.get("start", (-1,-1))
        end_pos = s.get("end", (-1,-1))
        solution_cells.add(start)
        solution_cells.add(end_pos)

        branch_points = set()
        for r, c in solution_cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<30 and 0<=nc<30 and inp[nr,nc] >= 2 and (nr,nc) not in solution_cells:
                    branch_points.add((r,c))
                    break

        # For each wrong solution cell, check if it's near a branch point
        for r in range(30):
            for c in range(30):
                if lab[r,c] == 5 and pred[r,c] != lab[r,c]:
                    total_errors += 1
                    # Min distance to any branch point (Manhattan)
                    if branch_points:
                        min_dist = min(abs(r-br)+abs(c-bc) for br,bc in branch_points)
                        if min_dist <= 3:
                            near_branch += 1
                        else:
                            far_branch += 1

    if total_errors > 0:
        print(f"  Solution cell errors near branch point (≤3 cells): "
              f"{near_branch}/{total_errors} ({100*near_branch/total_errors:.1f}%)")
        print(f"  Far from branch point (>3 cells): "
              f"{far_branch}/{total_errors} ({100*far_branch/total_errors:.1f}%)")


def false_positive_analysis(data: dict, label: str):
    """Cells predicted as solution (5) that shouldn't be — are they on wrong turns?"""
    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    structures = data.get("maze_structures", [])
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)

    # False positives: predicted 5 but label != 5
    # False negatives: label == 5 but predicted != 5
    fp = (final_preds == 5) & (labels != 5)
    fn = (labels == 5) & (final_preds != 5)

    print(f"\n--- False Positive/Negative Analysis: {label} ---")
    print(f"  False positives (predicted solution, actually not): {fp.sum()}")
    print(f"  False negatives (missed solution cell): {fn.sum()}")

    if fp.sum() > 0:
        # What are the true labels of false positive cells?
        fp_true = labels[fp]
        print("  FP true labels: ", end="")
        for v in range(6):
            count = (fp_true == v).sum()
            if count > 0:
                print(f"{v}={count} ", end="")
        print()

        # Are FPs on non-solution open paths (value 2 in both input and label)?
        fp_on_open = ((inputs == 2) & (labels == 2) & (final_preds == 5))
        print(f"  FPs that are open-path cells (input=2, label=2): {fp_on_open.sum()}")
        print("  = model marking wrong-turn paths as solution")


def main():
    results = load_data()

    # Run all analyses
    for label in sorted(results.keys()):
        basic_stats(results[label], label)
        error_analysis(results[label], label)
        spatial_analysis(results[label], label)
        convergence_analysis(results[label], label)
        backtracking_analysis(results[label], label)
        solution_path_position_analysis(results[label], label)
        wrong_turn_proximity_analysis(results[label], label)
        false_positive_analysis(results[label], label)

    # Dihedral analysis for augmented data
    for label in sorted(results.keys()):
        if "x07b_train" in label:
            dihedral_analysis(results[label], label)

    # Cross-checkpoint comparisons
    cross_checkpoint_analysis(results["x07b_train_7500"], results["x07b_train_9500"], "x07b train 7500→9500")
    cross_checkpoint_analysis(results["x07_train_6500"], results["x07_train_12500"], "x07 train 6500→12500")
    cross_checkpoint_analysis(results["x07b_test_7500"], results["x07b_test_9500"], "x07b test 7500→9500")
    cross_checkpoint_analysis(results["x07_test_6500"], results["x07_test_12500"], "x07 test 6500→12500")


if __name__ == "__main__":
    main()
