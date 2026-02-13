"""Diagnose per-H-cycle behavior across checkpoints."""

from pathlib import Path

import sys

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    exp_name = sys.argv[1]
    steps = [int(s) for s in sys.argv[2:]]

    mod = __import__(f"maze.{exp_name}", fromlist=["Experiment"])
    exp = mod.Experiment()
    exp.model.to(exp.device)

    # Load test data
    loader = exp.make_test_loader()
    batch = next(iter(loader))
    inputs, labels = batch[0].to(exp.device), batch[1].to(exp.device)
    B = inputs.shape[0]
    grid_len = exp.config.num_puzzle_grid_tokens
    puzzle_ids = torch.zeros(B, device=exp.device, dtype=torch.int32)

    for step in steps:
        ckpt_path = Path(__file__).parent / "ckpts" / exp_name / f"step{step:05d}.pt"
        ckpt = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
        exp.model.load_state_dict(ckpt["model"])
        exp.model.eval()

        z_H, z_L = exp._init_z(B)

        # Run multiple ACT steps, collect per-H-cycle accuracy at each
        max_steps = 16
        for act_step in range(max_steps):
            with torch.no_grad(), torch.autocast(
                device_type=exp.device.type, dtype=exp.dtype
            ):
                out = exp.model(inputs, z_H, z_L, puzzle_ids)

            z_H = out["z_H"]
            z_L = out["z_L"]
            all_logits = out["all_logits"]  # list of [B, grid_len, V]
            q_halt = out["q_halt"]

            # Per-H-cycle accuracy
            accs = []
            for h_idx, lg in enumerate(all_logits):
                preds = lg.argmax(dim=-1)
                cell_correct = (preds == labels).float().mean().item() * 100
                puzzle_correct = (
                    (preds == labels).all(dim=-1).float().mean().item() * 100
                )
                accs.append((cell_correct, puzzle_correct))

            # Per-H-cycle logit stats
            logit_norms = []
            for lg in all_logits:
                logit_norms.append(lg.float().norm(dim=-1).mean().item())

            # z_H drift between H-cycles
            z_H_list = out["all_z_H"]
            z_drifts = []
            for i in range(1, len(z_H_list)):
                drift = (z_H_list[i] - z_H_list[i - 1]).float().norm(dim=-1).mean().item()
                z_drifts.append(drift)

            halt_frac = (q_halt > 0).float().mean().item() * 100

            acc_str = "  ".join(
                f"H{i}:cell={ca:.1f}%,puz={pa:.1f}%"
                for i, (ca, pa) in enumerate(accs)
            )
            norm_str = "  ".join(
                f"H{i}:lnorm={n:.1f}" for i, n in enumerate(logit_norms)
            )
            drift_str = "  ".join(
                f"H{i}→H{i+1}:Δ={d:.2f}" for i, d in enumerate(z_drifts)
            )

            print(
                f"step={step:5d} act={act_step:2d} | {acc_str} | "
                f"{norm_str} | {drift_str} | halt={halt_frac:.0f}%"
            )

        print()


if __name__ == "__main__":
    main()
