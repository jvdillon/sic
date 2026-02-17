"""Analyze trajectory data collected by collect_trajectories.py.

Task-agnostic: uses _meta from the pickle to determine puzzle type and
run appropriate analyses. Maze-specific analyses (spatial, backtracking,
wrong-turn proximity, etc.) only run for type=maze.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import argparse
import pickle

import numpy as np


def _task(meta: dict[str, Any]) -> str:
    """Base task type (strip -aug suffix)."""
    return meta.get("task", meta.get("type", "maze").removesuffix("-aug"))


def load_data(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def basic_stats(data: dict[str, Any], label: str) -> None:
    """Per-step accuracy trajectory and q_halt stats."""
    logits = data["logits"]
    labels = data["labels"]
    q_halt = data["q_halt"]
    N, n_steps = logits.shape[0], logits.shape[1]

    print(f"\n{'=' * 70}")
    print(f"  {label}: {N} puzzles, {n_steps} steps")
    print(f"{'=' * 70}")

    print("\n  Step-by-step accuracy:")
    for step in range(n_steps):
        preds = logits[:, step].argmax(axis=-1)
        cell_correct = preds == labels
        cell_acc = cell_correct.mean() * 100
        puzzle_acc = cell_correct.all(axis=-1).mean() * 100
        avg_q = q_halt[:, step].mean()
        halt_rate = (q_halt[:, step] > 0).mean() * 100
        print(
            f"    H{step + 1:2d}: cell={cell_acc:6.2f}%  puzzle={puzzle_acc:5.1f}%  "
            f"q_halt={avg_q:+.3f}  halt={halt_rate:5.1f}%"
        )


def error_analysis(data: dict[str, Any], label: str, meta: dict[str, Any]) -> None:
    """Deep dive into what cells are wrong and why."""
    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    N = len(logits)
    vocab_size = logits.shape[-1]

    final_preds = logits[:, -1].argmax(axis=-1)
    wrong_mask = final_preds != labels

    print(f"\n--- Error Analysis: {label} ---")

    # For maze: solution cells are input=2 & label=5
    if _task(meta) == "maze":
        solution_mask = (inputs == 2) & (labels == 5)
        nonsolution_mask = ~solution_mask

        sol_wrong = wrong_mask & solution_mask
        nonsol_wrong = wrong_mask & nonsolution_mask
        n_sol = solution_mask.sum()
        n_nonsol = nonsolution_mask.sum()
        print(
            f"\n  Solution cells: {sol_wrong.sum()}/{n_sol} wrong "
            f"({100 * sol_wrong.sum() / max(n_sol, 1):.2f}%)"
        )
        print(
            f"  Non-solution cells: {nonsol_wrong.sum()}/{n_nonsol} wrong "
            f"({100 * nonsol_wrong.sum() / max(n_nonsol, 1):.2f}%)"
        )

        if sol_wrong.sum() > 0:
            wrong_preds_on_sol = final_preds[sol_wrong]
            print("\n  What model predicts on WRONG solution cells:")
            for v in range(vocab_size):
                count = (wrong_preds_on_sol == v).sum()
                if count > 0:
                    print(
                        f"    Predicted {v}: {count}"
                        f" ({100 * count / len(wrong_preds_on_sol):.1f}%)"
                    )

    # Logit entropy on wrong vs correct cells
    final_logits = logits[:, -1]
    probs = np.exp(final_logits - final_logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=-1)

    correct_mask = ~wrong_mask
    if wrong_mask.sum() > 0 and correct_mask.sum() > 0:
        print(
            f"\n  Entropy: wrong={entropy[wrong_mask].mean():.3f}, "
            f"correct={entropy[correct_mask].mean():.3f}"
        )

    # Per-puzzle error count distribution
    errors_per_puzzle = wrong_mask.sum(axis=1)
    print(
        f"\n  Errors per puzzle: mean={errors_per_puzzle.mean():.1f}, "
        f"std={errors_per_puzzle.std():.1f}, "
        f"min={errors_per_puzzle.min()}, max={errors_per_puzzle.max()}"
    )
    perfect = (errors_per_puzzle == 0).sum()
    print(f"  Perfect puzzles: {perfect}/{N} ({100 * perfect / N:.1f}%)")

    for threshold in [0, 1, 2, 5, 10, 20, 50, 100]:
        count = (errors_per_puzzle <= threshold).sum()
        print(f"    <={threshold:3d} errors: {count}/{N} ({100 * count / N:.1f}%)")


def spatial_analysis(data: dict[str, Any], label: str) -> None:
    """Where on the 30x30 grid do errors cluster? (maze only)"""
    logits = data["logits"]
    labels = data["labels"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    wrong_mask = (final_preds != labels).reshape(N, 30, 30)
    error_rate = wrong_mask.mean(axis=0)

    print(f"\n--- Spatial Analysis: {label} ---")

    border_mask = np.zeros((30, 30), dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = border_mask[:, -1] = True
    interior_mask = ~border_mask

    print(f"  Border error rate: {wrong_mask[:, border_mask].mean() * 100:.2f}%")
    print(f"  Interior error rate: {wrong_mask[:, interior_mask].mean() * 100:.2f}%")

    for qr, qc, name in [
        (slice(0, 15), slice(0, 15), "top-left"),
        (slice(0, 15), slice(15, 30), "top-right"),
        (slice(15, 30), slice(0, 15), "bottom-left"),
        (slice(15, 30), slice(15, 30), "bottom-right"),
    ]:
        print(f"  {name}: {wrong_mask[:, qr, qc].mean() * 100:.2f}%")

    print("\n  Error heatmap (6x6 blocks, % wrong):")
    for br in range(6):
        row_str = "    "
        for bc in range(6):
            block_err = (
                error_rate[br * 5 : (br + 1) * 5, bc * 5 : (bc + 1) * 5].mean() * 100
            )
            row_str += f"{block_err:5.1f} "
        print(row_str)


def dihedral_analysis(
    data: dict[str, Any],
    label: str,
    n_augs: int,
) -> None:
    """Are errors consistent across dihedral augmentations?"""
    logits = data["logits"]
    labels = data["labels"]
    N = len(logits)

    if N % n_augs != 0:
        print(
            f"\n--- Dihedral Analysis: {label} SKIPPED"
            f" (N={N} not divisible by {n_augs}) ---"
        )
        return

    n_instances = N // n_augs
    final_preds = logits[:, -1].argmax(axis=-1)
    correct = (final_preds == labels).all(axis=-1)
    correct_by_instance = correct.reshape(n_instances, n_augs)

    print(
        f"\n--- Dihedral Analysis: {label}"
        f" ({n_instances} instances x {n_augs} augs) ---"
    )

    solved_counts = correct_by_instance.sum(axis=1)
    for k in range(n_augs + 1):
        n = (solved_counts == k).sum()
        if n > 0:
            print(
                f"  {k}/{n_augs} augs solved: {n} instances"
                f" ({100 * n / n_instances:.1f}%)"
            )

    all_solved = (solved_counts == n_augs).sum()
    none_solved = (solved_counts == 0).sum()
    partial = n_instances - all_solved - none_solved
    print(
        f"\n  All {n_augs} solved: {all_solved},"
        f" None solved: {none_solved}, Partial: {partial}"
    )

    if partial > 0:
        partial_idx = np.where((solved_counts > 0) & (solved_counts < n_augs))[0]
        cell_wrong_counts = []
        for idx in partial_idx[:20]:
            base = idx * n_augs
            wrong_per_aug = [
                (final_preds[base + aug] != labels[base + aug]).sum()
                for aug in range(n_augs)
            ]
            cell_wrong_counts.append(wrong_per_aug)
        cell_wrong_counts_arr = np.array(cell_wrong_counts)
        print(
            f"\n  Wrong cells per aug for"
            f" {len(cell_wrong_counts_arr)} partial instances:"
        )
        print(f"    Mean errors per aug: {cell_wrong_counts_arr.mean(axis=1)}")
        print(f"    Std across augs: {cell_wrong_counts_arr.std(axis=1).mean():.1f}")


def backtracking_analysis(data: dict[str, Any], label: str) -> None:
    """Correlate errors with maze backtracking difficulty. (maze only)"""
    structures = data.get("maze_structures", [])
    if not structures:
        return

    logits = data["logits"]
    labels = data["labels"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    errors_per_puzzle = (final_preds != labels).sum(axis=1)

    n_structs = len(structures)
    n_augs = N // n_structs if n_structs < N else 1
    struct_idx = np.arange(N) // n_augs if n_structs < N else np.arange(N)

    print(f"\n--- Backtracking Analysis: {label} ---")

    perfect = errors_per_puzzle == 0
    imperfect = errors_per_puzzle > 0

    if imperfect.sum() > 0 and perfect.sum() > 0:
        perf_structs = [
            structures[struct_idx[i]]
            for i in np.where(perfect)[0]
            if struct_idx[i] < len(structures)
        ]
        imp_structs = [
            structures[struct_idx[i]]
            for i in np.where(imperfect)[0]
            if struct_idx[i] < len(structures)
        ]

        for metric in [
            "path_len",
            "wrong_turns",
            "max_wrong_depth",
            "mean_wrong_depth",
            "total_wrong_cells",
        ]:
            perf_vals = [s[metric] for s in perf_structs]
            imp_vals = [s[metric] for s in imp_structs]
            if perf_vals and imp_vals:
                print(f"  {metric}:")
                print(
                    f"    Perfect: mean={np.mean(perf_vals):.1f}"
                    f" +/- {np.std(perf_vals):.1f}"
                )
                print(
                    f"    Imperfect: mean={np.mean(imp_vals):.1f}"
                    f" +/- {np.std(imp_vals):.1f}"
                )

    for metric in ["path_len", "wrong_turns", "max_wrong_depth", "total_wrong_cells"]:
        vals = np.array(
            [
                structures[min(struct_idx[i], len(structures) - 1)][metric]
                for i in range(N)
            ]
        )
        corr = np.corrcoef(errors_per_puzzle, vals)[0, 1]
        print(f"  Correlation(errors, {metric}): {corr:.3f}")


def convergence_analysis(
    data: dict[str, Any], label: str, meta: dict[str, Any]
) -> None:
    """How do errors evolve over steps?"""
    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]
    n_steps = logits.shape[1]
    mid = n_steps // 2

    print(f"\n--- Convergence Analysis: {label} ---")

    preds_mid = logits[:, mid - 1].argmax(axis=-1)
    preds_final = logits[:, -1].argmax(axis=-1)

    correct_mid = preds_mid == labels
    correct_final = preds_final == labels

    regressed = correct_mid & ~correct_final
    improved = ~correct_mid & correct_final
    stuck_wrong = ~correct_mid & ~correct_final

    # For maze: restrict to solution cells
    if _task(meta) == "maze":
        mask = (inputs == 2) & (labels == 5)
        mask_label = "solution cells"
    else:
        mask = np.ones_like(labels, dtype=bool)
        mask_label = "all cells"

    total = mask.sum()
    print(f"  H{mid}->H{n_steps} transitions ({mask_label}):")
    print(
        f"    Stayed correct: {(correct_mid & correct_final & mask).sum()}"
        f" ({100 * (correct_mid & correct_final & mask).sum() / total:.2f}%)"
    )
    print(
        f"    Improved:       {(improved & mask).sum()}"
        f" ({100 * (improved & mask).sum() / total:.2f}%)"
    )
    print(
        f"    Regressed:      {(regressed & mask).sum()}"
        f" ({100 * (regressed & mask).sum() / total:.2f}%)"
    )
    print(
        f"    Stuck wrong:    {(stuck_wrong & mask).sum()}"
        f" ({100 * (stuck_wrong & mask).sum() / total:.2f}%)"
    )

    stuck_positions = np.argwhere(stuck_wrong & mask)
    if len(stuck_positions) > 0:
        sample = stuck_positions[:200]
        first_correct_step = []
        never_correct = 0
        for puzzle_idx, cell_idx in sample:
            ever_correct = False
            for step in range(n_steps):
                if (
                    logits[puzzle_idx, step, cell_idx].argmax()
                    == labels[puzzle_idx, cell_idx]
                ):
                    first_correct_step.append(step)
                    ever_correct = True
                    break
            if not ever_correct:
                never_correct += 1
        if first_correct_step:
            print(f"\n  Stuck-wrong cells (sample of {len(sample)}):")
            print(
                f"    Never correct across all {n_steps} steps:"
                f" {never_correct}/{len(sample)}"
            )
            print(f"    First correct at step: mean={np.mean(first_correct_step):.1f}")


def cross_checkpoint_analysis(
    data_early: dict[str, Any],
    data_late: dict[str, Any],
    label: str,
) -> None:
    """Compare errors between early and late checkpoint."""
    logits_e = data_early["logits"]
    logits_l = data_late["logits"]
    labels = data_early["labels"]

    preds_e = logits_e[:, -1].argmax(axis=-1)
    preds_l = logits_l[:, -1].argmax(axis=-1)

    correct_e = preds_e == labels
    correct_l = preds_l == labels

    puzzle_correct_e = correct_e.all(axis=-1)
    puzzle_correct_l = correct_l.all(axis=-1)

    print(f"\n--- Cross-Checkpoint: {label} ---")
    N = len(labels)
    both = (puzzle_correct_e & puzzle_correct_l).sum()
    only_early = (puzzle_correct_e & ~puzzle_correct_l).sum()
    only_late = (~puzzle_correct_e & puzzle_correct_l).sum()
    neither = (~puzzle_correct_e & ~puzzle_correct_l).sum()
    print(f"  Both correct: {both} ({100 * both / N:.1f}%)")
    print(f"  Only early:   {only_early} ({100 * only_early / N:.1f}%)")
    print(f"  Only late:    {only_late} ({100 * only_late / N:.1f}%)")
    print(f"  Neither:      {neither} ({100 * neither / N:.1f}%)")

    total = labels.size
    cell_both = (correct_e & correct_l).sum()
    cell_only_e = (correct_e & ~correct_l).sum()
    cell_only_l = (~correct_e & correct_l).sum()
    cell_neither = (~correct_e & ~correct_l).sum()
    print("\n  Cell-level:")
    print(f"    Both correct: {cell_both} ({100 * cell_both / total:.2f}%)")
    print(f"    Only early:   {cell_only_e} ({100 * cell_only_e / total:.2f}%)")
    print(f"    Only late:    {cell_only_l} ({100 * cell_only_l / total:.2f}%)")
    print(f"    Neither:      {cell_neither} ({100 * cell_neither / total:.2f}%)")


def solution_path_position_analysis(data: dict[str, Any], label: str) -> None:
    """Are errors correlated with position along the solution path? (maze only)"""
    structures = data.get("maze_structures", [])
    if not structures:
        return

    logits = data["logits"]
    labels = data["labels"]
    N = len(logits)

    final_preds = logits[:, -1].argmax(axis=-1)
    n_structs = len(structures)
    n_augs = max(1, N // n_structs)

    print(f"\n--- Solution Path Position Analysis: {label} ---")

    dist_bins = [0, 10, 20, 30, 50, 70, 100, 150, 200]
    bin_wrong: dict[int, int] = defaultdict(int)
    bin_total: dict[int, int] = defaultdict(int)

    for i in range(0, min(N, n_structs * n_augs), n_augs):
        si = i // n_augs
        if si >= len(structures):
            break
        s = structures[si]
        path_dist = s.get("path_dist", {})
        if not path_dist:
            continue

        lab = labels[i].reshape(30, 30)
        pred = final_preds[i].reshape(30, 30)

        for (r, c), dist in path_dist.items():
            if lab[r, c] == 5:
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
            next_t = dist_bins[b + 1] if b + 1 < len(dist_bins) else "+"
            print(
                f"    dist {threshold:3d}-{next_t}:"
                f" {w}/{t} wrong ({100 * w / max(t, 1):.1f}%)"
            )


def wrong_turn_proximity_analysis(data: dict[str, Any], label: str) -> None:
    """Are errors near wrong-turn branch points? (maze only)"""
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

    near_branch = 0
    far_branch = 0
    total_errors = 0

    for i in range(0, min(N, n_structs * n_augs), n_augs):
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

        solution_cells = set(map(tuple, np.argwhere(lab == 5)))
        start = s.get("start", (-1, -1))
        end_pos = s.get("end", (-1, -1))
        solution_cells.add(start)
        solution_cells.add(end_pos)

        branch_points: set[tuple[int, int]] = set()
        for r, c in solution_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < 30
                    and 0 <= nc < 30
                    and inp[nr, nc] >= 2
                    and (nr, nc) not in solution_cells
                ):
                    branch_points.add((r, c))
                    break

        for r in range(30):
            for c in range(30):
                if lab[r, c] == 5 and pred[r, c] != lab[r, c]:
                    total_errors += 1
                    if branch_points:
                        min_dist = min(
                            abs(r - br) + abs(c - bc) for br, bc in branch_points
                        )
                        if min_dist <= 3:
                            near_branch += 1
                        else:
                            far_branch += 1

    if total_errors > 0:
        print(
            f"  Near branch point (<=3 cells):"
            f" {near_branch}/{total_errors}"
            f" ({100 * near_branch / total_errors:.1f}%)"
        )
        print(
            f"  Far from branch point (>3 cells):"
            f" {far_branch}/{total_errors}"
            f" ({100 * far_branch / total_errors:.1f}%)"
        )


def false_positive_analysis(
    data: dict[str, Any],
    label: str,
    meta: dict[str, Any],
) -> None:
    """Cells predicted as solution that shouldn't be."""
    if meta["type"] != "maze":
        return

    logits = data["logits"]
    labels = data["labels"]
    inputs = data["inputs"]

    final_preds = logits[:, -1].argmax(axis=-1)

    fp = (final_preds == 5) & (labels != 5)
    fn = (labels == 5) & (final_preds != 5)

    print(f"\n--- False Positive/Negative Analysis: {label} ---")
    print(f"  False positives (predicted solution, actually not): {fp.sum()}")
    print(f"  False negatives (missed solution cell): {fn.sum()}")

    if fp.sum() > 0:
        fp_true = labels[fp]
        print("  FP true labels: ", end="")
        for v in range(6):
            count = (fp_true == v).sum()
            if count > 0:
                print(f"{v}={count} ", end="")
        print()

        fp_on_open = (inputs == 2) & (labels == 2) & (final_preds == 5)
        print(f"  FPs on open-path cells (input=2, label=2): {fp_on_open.sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=None,
        help="Input pickle path (default: trajectory_data.pkl)",
    )
    args = parser.parse_args()

    path = (
        Path(args.input)
        if args.input
        else Path(__file__).parent / "trajectory_data.pkl"
    )
    results = load_data(path)
    meta: dict[str, dict[str, Any]] = results.pop("_meta", {})

    # Run analyses per label
    for label in sorted(results.keys()):
        m = meta.get(label, {"type": "maze", "split": "test", "n_augs": 1})
        basic_stats(results[label], label)
        error_analysis(results[label], label, m)
        convergence_analysis(results[label], label, m)
        false_positive_analysis(results[label], label, m)

        # Maze-specific analyses
        if _task(m) == "maze":
            spatial_analysis(results[label], label)
            backtracking_analysis(results[label], label)
            solution_path_position_analysis(results[label], label)
            wrong_turn_proximity_analysis(results[label], label)

        # Dihedral analysis for augmented data
        if m.get("n_augs", 1) > 1:
            dihedral_analysis(results[label], label, m["n_augs"])

    # Cross-checkpoint: for each (exp, split) pair with 2+ steps, compare consecutive
    from itertools import groupby  # noqa: PLC0415

    def _key(label: str) -> tuple[str, str]:
        m = meta.get(label, {})
        return (m.get("exp", ""), m.get("split", ""))

    sorted_labels = sorted(
        (k for k in results if k in meta),
        key=lambda k: (_key(k), meta.get(k, {}).get("step", 0)),
    )
    for (exp, split), group in groupby(sorted_labels, key=_key):
        labels_in_group = list(group)
        for i in range(len(labels_in_group) - 1):
            l_early = labels_in_group[i]
            l_late = labels_in_group[i + 1]
            step_e = meta[l_early]["step"]
            step_l = meta[l_late]["step"]
            cross_checkpoint_analysis(
                results[l_early],
                results[l_late],
                f"{exp} {split} {step_e}->{step_l}",
            )


if __name__ == "__main__":
    main()
