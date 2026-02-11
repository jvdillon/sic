"""Probe: anatomy of the loss for solved vs unsolved puzzles.

Break down per-cell cross-entropy by:
- Label type (wall=1, open=2, start=3, end=4, solution=5)
- Correct vs incorrect prediction
- Solved vs unsolved puzzle
"""

from __future__ import annotations

import pathlib
import sys

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import resume_from_checkpoint
from maze.x000l import Experiment


def run():
    exp = Experiment()
    exp.setup_optimizers()
    resume_from_checkpoint(exp, "maze/ckpts/x000l/step15000.pt")
    model = exp.model
    model.config.compile_core = False
    device = exp.device
    assert device is not None
    dtype = exp.dtype

    loader = exp.make_test_loader()
    all_inputs, all_labels = [], []
    for batch in loader:
        inputs, labels, _pids, vc = batch
        for i in range(vc):
            all_inputs.append(inputs[i])
            all_labels.append(labels[i])

    n_puzzles = len(all_inputs)
    H_init = model.H_init[0]
    L_init = model.L_init[0]
    batch_size = min(256, n_puzzles)

    # Collect per-puzzle stats
    solved_loss_by_label = {t: [] for t in range(1, 6)}
    unsolved_loss_by_label = {t: [] for t in range(1, 6)}
    unsolved_wrong_loss_by_label = {t: [] for t in range(1, 6)}
    unsolved_correct_loss_by_label = {t: [] for t in range(1, 6)}
    n_solved = 0
    n_unsolved = 0

    # For unsolved: what does the model predict at error sites?
    error_pred_dist = dict.fromkeys(range(6), 0)
    error_true_dist = dict.fromkeys(range(6), 0)
    # What's the model predicting at FP sites (pred=5 but label!=5)?
    fp_label_dist = dict.fromkeys(range(6), 0)
    # What's the model predicting at FN sites (label=5 but pred!=5)?
    fn_pred_dist = dict.fromkeys(range(6), 0)

    with torch.no_grad():
        for start in range(0, n_puzzles, batch_size):
            end = min(start + batch_size, n_puzzles)
            B = end - start
            inp = torch.stack(all_inputs[start:end])
            lab = torch.stack(all_labels[start:end])

            z_H = H_init.unsqueeze(0).expand(B, -1, -1).clone()
            z_L = L_init.unsqueeze(0).expand(B, -1, -1).clone()
            pids = torch.zeros(B, device=device, dtype=torch.int32)

            with torch.autocast(device_type=device.type, dtype=dtype):
                out = model(inp, z_H, z_L, pids)
            logits = out["logits"].float()  # [B, 900, 6]
            preds = logits.argmax(dim=-1)  # [B, 900]

            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                lab.reshape(-1).long(),
                reduction="none",
            ).reshape(B, -1)  # [B, 900]

            for i in range(B):
                solved = (preds[i] == lab[i]).all().item()
                loss_map = solved_loss_by_label if solved else unsolved_loss_by_label

                for t in range(1, 6):
                    mask = lab[i] == t
                    if mask.any():
                        loss_map[t].extend(ce[i][mask].tolist())

                if solved:
                    n_solved += 1
                else:
                    n_unsolved += 1
                    wrong = preds[i] != lab[i]
                    correct = ~wrong

                    for t in range(1, 6):
                        mask_t = lab[i] == t
                        if mask_t.any():
                            wrong_t = wrong & mask_t
                            correct_t = correct & mask_t
                            if wrong_t.any():
                                unsolved_wrong_loss_by_label[t].extend(
                                    ce[i][wrong_t].tolist()
                                )
                            if correct_t.any():
                                unsolved_correct_loss_by_label[t].extend(
                                    ce[i][correct_t].tolist()
                                )

                    # Error analysis
                    wrong_idx = wrong.nonzero(as_tuple=True)[0]
                    for j in wrong_idx:
                        pred_val = int(preds[i][j].item())
                        true_val = int(lab[i][j].item())
                        error_pred_dist[pred_val] += 1
                        error_true_dist[true_val] += 1

                    # FP: pred=5, label!=5
                    fp = (preds[i] == 5) & (lab[i] != 5)
                    for j in fp.nonzero(as_tuple=True)[0]:
                        fp_label_dist[int(lab[i][j].item())] += 1

                    # FN: label=5, pred!=5
                    fn = (lab[i] == 5) & (preds[i] != 5)
                    for j in fn.nonzero(as_tuple=True)[0]:
                        fn_pred_dist[int(preds[i][j].item())] += 1

            if start % 512 == 0:
                print(f"  processed {start}/{n_puzzles}", flush=True)

    label_names = {1: "wall", 2: "open", 3: "start", 4: "end", 5: "solution"}

    print(f"\n{'=' * 60}")
    print("LOSS ANATOMY: x000l @ step 15000")
    print(f"  {n_solved} solved, {n_unsolved} unsolved")
    print(f"{'=' * 60}")

    print("\n--- SOLVED puzzles: mean CE by label type ---")
    for t in range(1, 6):
        vals = solved_loss_by_label[t]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {label_names[t]:8s} (n={len(vals):7d}): {m:.6f}")

    print("\n--- UNSOLVED puzzles: mean CE by label type ---")
    for t in range(1, 6):
        vals = unsolved_loss_by_label[t]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {label_names[t]:8s} (n={len(vals):7d}): {m:.6f}")

    print("\n--- UNSOLVED: CE at CORRECT cells by label type ---")
    for t in range(1, 6):
        vals = unsolved_correct_loss_by_label[t]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {label_names[t]:8s} (n={len(vals):7d}): {m:.6f}")

    print("\n--- UNSOLVED: CE at WRONG cells by label type ---")
    for t in range(1, 6):
        vals = unsolved_wrong_loss_by_label[t]
        if vals:
            m = sum(vals) / len(vals)
            print(f"  {label_names[t]:8s} (n={len(vals):7d}): {m:.6f}")

    print("\n--- Error patterns at wrong cells ---")
    total_errors = sum(error_pred_dist.values())
    print(f"Total error cells: {total_errors}")
    print("\nTrue labels at error sites:")
    for t, c in sorted(error_true_dist.items()):
        if c > 0:
            print(
                f"  {t} ({label_names.get(t, '?'):8s}): {c:5d} ({100 * c / total_errors:.1f}%)"
            )
    print("\nPredicted labels at error sites:")
    for t, c in sorted(error_pred_dist.items()):
        if c > 0:
            print(
                f"  {t} ({label_names.get(t, '?'):8s}): {c:5d} ({100 * c / total_errors:.1f}%)"
            )

    print("\nFalse Positives (pred=solution, label=?):")
    for t, c in sorted(fp_label_dist.items()):
        if c > 0:
            print(f"  true={t} ({label_names.get(t, '?'):8s}): {c}")
    print("\nFalse Negatives (label=solution, pred=?):")
    for t, c in sorted(fn_pred_dist.items()):
        if c > 0:
            print(f"  pred={t} ({label_names.get(t, '?'):8s}): {c}")

    # Gradient competition: what fraction of total loss comes from errors?
    all_unsolved_loss = []
    all_unsolved_wrong_loss = []
    for t in range(1, 6):
        all_unsolved_loss.extend(unsolved_loss_by_label[t])
        all_unsolved_wrong_loss.extend(unsolved_wrong_loss_by_label[t])

    if all_unsolved_loss and all_unsolved_wrong_loss:
        total = sum(all_unsolved_loss)
        wrong_total = sum(all_unsolved_wrong_loss)
        correct_total = total - wrong_total
        n_wrong = len(all_unsolved_wrong_loss)
        n_correct = len(all_unsolved_loss) - n_wrong
        print("\n--- Gradient competition (unsolved puzzles) ---")
        print(
            f"  {n_wrong} wrong cells: total CE = {wrong_total:.1f} ({100 * wrong_total / total:.1f}% of loss)"
        )
        print(
            f"  {n_correct} correct cells: total CE = {correct_total:.1f} ({100 * correct_total / total:.1f}% of loss)"
        )
        print(f"  Per-cell CE wrong:   {wrong_total / n_wrong:.4f}")
        print(f"  Per-cell CE correct: {correct_total / n_correct:.4f}")


if __name__ == "__main__":
    run()
