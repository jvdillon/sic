"""Visualize model predictions with ACT step diffs overlaid on a single grid.

Each puzzle is one grid. Cells that changed are colored by which ACT step last
changed them. Dark = correct, light/pastel = wrong. Walls/start/goal stay in
their base colors (they're almost never wrong).

Usage:
    python maze/viz_eval.py EXP STEP [options]

Examples:
    python maze/viz_eval.py x07 3500 --split test --num_samples 64 --rows_per_png 8
    python maze/viz_eval.py x07b 2500 --split train --num_samples 16 --rows_per_png 8 --aug

"""

import argparse
import importlib
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment import Experiment as ExperimentBase

from data import PuzzleDataset


# Step 0: natural maze colors (token-based)
BASE_COLORS = np.array(
    [
        [230, 230, 230],  # 0: pad - light gray
        [40, 40, 40],  # 1: wall - near black
        [255, 255, 255],  # 2: empty - white
        [255, 230, 0],  # 3: start - yellow
        [255, 60, 60],  # 4: goal - red
        [60, 190, 60],  # 5: solution - green
    ],
    dtype=np.uint8,
)

# Steps 1-15: distinct colors that contrast against the base maze palette.
# Avoid green (solution), blue (start), red (goal), black (wall), white (empty).
# Use: orange, magenta, cyan, yellow, purple, teal, coral, lime, pink, gold, ...
DIFF_COLORS = (
    np.array(
        [
            [255, 140, 0],  # 1: orange
            [220, 0, 220],  # 2: magenta
            [0, 210, 210],  # 3: cyan
            [100, 100, 255],  # 4: blue
            [160, 0, 255],  # 5: purple
            [0, 180, 120],  # 6: teal
            [255, 100, 100],  # 7: coral
            [140, 220, 0],  # 8: lime
            [255, 80, 180],  # 9: pink
            [200, 170, 0],  # 10: gold
            [230, 230, 0],  # 11: yellow
            [180, 120, 60],  # 12: brown
            [0, 255, 160],  # 13: mint
            [255, 160, 200],  # 14: light pink
            [120, 200, 255],  # 15: sky blue
        ],
        dtype=np.float32,
    )
    / 255.0
)  # normalize to [0,1]


def run_model_steps(
    exp: ExperimentBase,
    inputs: torch.Tensor,
    puzzle_ids: torch.Tensor,
    max_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run model for max_steps, collecting predictions and q_halt at each step."""
    B = inputs.shape[0]
    z_H, z_L = exp._init_z(B)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    preds_per_step = []
    qhalt_per_step = []

    for _ in range(max_steps):
        with torch.autocast(device_type=exp.device.type, dtype=exp.dtype):  # pyright: ignore[reportOptionalMemberAccess]
            out = exp._eval_forward(inputs, z_H, z_L, puzzle_ids)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        z_H = out["z_H"]
        z_L = out["z_L"]
        preds_per_step.append(out["logits"].argmax(dim=-1).cpu().numpy())
        qhalt_per_step.append(out["q_halt"].float().cpu().numpy())

    return preds_per_step, qhalt_per_step


def render_overlay(
    preds_per_step: list[np.ndarray],
    labels: np.ndarray,
    grid_shape: tuple[int, int],
    halt_step: int,
    cell_px: int,
) -> np.ndarray:
    """Render one puzzle: base maze + thick colored path per ACT step.

    Base: black=wall, white=floor, light green=ground truth solution,
    blue=start, red=goal.

    Each ACT step draws a thick line through its predicted solution cells,
    with sub-cell offsets so steps don't overlap each other.

    Returns: (H * cell_px, W * cell_px, 3) uint8 image.
    """
    H, W = grid_shape
    n_steps = len(preds_per_step)
    label_grid = labels.reshape(H, W)
    final_s = min(halt_step, n_steps - 1)

    GT_COLORS = np.array(
        [
            [230, 230, 230],  # 0: pad
            [30, 30, 30],  # 1: wall - black
            [255, 255, 255],  # 2: empty - white
            [80, 140, 255],  # 3: start - blue
            [255, 220, 40],  # 4: goal - yellow
            [200, 245, 200],  # 5: solution - lighter green
        ],
        dtype=np.uint8,
    )
    base_img = GT_COLORS[label_grid]

    img = np.repeat(np.repeat(base_img, cell_px, axis=0), cell_px, axis=1)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    preds = [p.reshape(H, W) for p in preds_per_step[: min(n_steps, final_s + 1)]]

    # Find stabilization point
    last_active = 0
    for s in range(len(preds) - 1):
        if (preds[s] != preds[s + 1]).any():
            last_active = s + 1
    n_draw = last_active + 1

    # Sub-cell offsets: spread along the diagonal so each step is
    # separated in *both* x and y. This ensures visibility whether
    # the path segment is horizontal or vertical.
    half = cell_px // 2
    line_w = max(2, cell_px // 6)
    max_radius = half - 2
    # Max diagonal distance from center: spread evenly
    if n_draw == 1:
        offsets = [(0, 0)]
    else:
        step = min(3.0, 2.0 * max_radius / (n_draw - 1))
        offsets = []
        for i in range(n_draw):
            t = i - (n_draw - 1) / 2.0  # centered: -k..0..+k
            dx = round(t * step)
            dy = round(t * step)
            offsets.append((dx, dy))

    # Step colors
    step_colors = [(60, 190, 60)]  # step 0 = green
    for s in range(1, n_draw):
        dc = DIFF_COLORS[min(s - 1, len(DIFF_COLORS) - 1)]
        rgb = tuple(int(v * 220 + 20) for v in dc)
        step_colors.append(rgb)  # pyright: ignore[reportArgumentType]

    # Start/goal positions from ground truth (fixed across steps)
    start_mask = label_grid == 3
    goal_mask = label_grid == 4

    # Error overlay on base (halt step only).
    # Missed solution cells (label=sol, pred!=sol) → light orange.
    # All other errors → light red.
    halt_pred = preds_per_step[final_s].reshape(H, W)
    wrong = halt_pred != label_grid
    missed_sol = wrong & (label_grid == 5)
    other_wrong = wrong & ~missed_sol

    result = np.array(pil_img).astype(np.float32)
    orange = np.array([255, 180, 80], dtype=np.float32)
    red = np.array([255, 80, 80], dtype=np.float32)

    missed_px = np.repeat(np.repeat(missed_sol, cell_px, axis=0), cell_px, axis=1)
    other_px = np.repeat(np.repeat(other_wrong, cell_px, axis=0), cell_px, axis=1)
    result[missed_px] = result[missed_px] * 0.45 + orange * 0.55
    result[other_px] = result[other_px] * 0.55 + red * 0.45

    pil_img = Image.fromarray(result.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    # Draw step lines (pure colors, on top of red-tinted base)
    for s in range(n_draw):
        path_mask = (preds[s] == 5) | start_mask | goal_mask
        dx, dy = offsets[s]
        color = step_colors[s]

        for r in range(H):
            for c in range(W):
                if not path_mask[r, c]:
                    continue
                cx = c * cell_px + half + dx
                cy = r * cell_px + half + dy

                d = line_w // 2
                draw.rectangle([cx - d, cy - d, cx + d, cy + d], fill=color)

                if c + 1 < W and path_mask[r, c + 1]:
                    nx = (c + 1) * cell_px + half + dx
                    draw.line([(cx, cy), (nx, cy)], fill=color, width=line_w)

                if r + 1 < H and path_mask[r + 1, c]:
                    ny = (r + 1) * cell_px + half + dy
                    draw.line([(cx, cy), (cx, ny)], fill=color, width=line_w)

    return np.array(pil_img)


def save_individual(
    preds_per_step: list[np.ndarray],
    qhalt_per_step: list[np.ndarray],
    labels: np.ndarray,
    instance_ids: list[int],
    aug_ids: list[int],
    grid_shape: tuple[int, int],
    max_steps: int,
    out_dir: pathlib.Path,
    split: str = "test",
) -> None:
    """Save each maze as its own PNG."""
    R = labels.shape[0]
    num_steps_run = len(preds_per_step)
    H, W = grid_shape
    cell_px = 32
    label_h = 24
    pad = 4

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10
        )
    except OSError:
        font = ImageFont.load_default()

    # Find halt step per sample
    halt_step = np.full(R, max_steps, dtype=int)
    for s in range(num_steps_run):
        for r in range(R):
            if qhalt_per_step[s][r] > 0 and halt_step[r] == max_steps:
                halt_step[r] = s

    final_idx = np.minimum(halt_step, num_steps_run - 1)
    final_preds = np.array([preds_per_step[final_idx[r]][r] for r in range(R)])
    num_incorrect = labels.shape[1] - (final_preds == labels).sum(axis=1)

    for r in range(R):
        sample_preds = [p[r] for p in preds_per_step]
        maze_img = render_overlay(
            sample_preds,
            labels[r],
            grid_shape,
            int(halt_step[r]),
            cell_px,
        )
        # Add label bar below
        img_h = H * cell_px + label_h + pad
        img_w = W * cell_px
        canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
        canvas[: H * cell_px, :] = maze_img

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        hs = str(halt_step[r] + 1) if halt_step[r] < max_steps else "--"
        ni = int(num_incorrect[r])
        draw.text(
            (2, H * cell_px + pad),
            f"{'e' if split == 'test' else 't'}{instance_ids[r]}/a{aug_ids[r]}  h={hs}  err={ni}",
            fill=(0, 0, 0),
            font=font,
        )

        prefix = "e" if split == "test" else "t"
        h = int(halt_step[r])
        h_str = f"{h + 1:02d}"
        err_str = "hit" if ni == 0 else f"err{ni:03d}"
        fname = f"{prefix}{instance_ids[r]:04d}_a{aug_ids[r]}_h{h_str}_{err_str}.png"
        pil.save(out_dir / fname)

    print(f"  saved {R} images to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ACT steps")
    parser.add_argument("exp", type=str, help="Experiment name (e.g. x07)")
    parser.add_argument("step", type=int, help="Checkpoint step")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument(
        "--aug", action="store_true", help="Group by instance (8 augs per instance)"
    )
    parser.add_argument("--max_steps", type=int, default=16)
    args = parser.parse_args()

    maze_dir = pathlib.Path(__file__).resolve().parent
    mod = importlib.import_module(args.exp)
    exp: ExperimentBase = mod.Experiment()
    exp.setup_model()

    ckpt_path = maze_dir / "ckpts" / args.exp / f"step{args.step:05d}.pt"
    ckpt = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
    exp.model.load_state_dict(ckpt["model"])
    exp.current_step = args.step
    exp.model.eval()
    print(f"Loaded {ckpt_path}")

    ds = PuzzleDataset(
        data_dir=exp.data_dir,
        device=torch.device("cpu"),
        batch_size=args.num_samples,
        train=(args.split == "train"),
        shuffle=False,
    )
    instance_bounds = ds.instance_bounds.numpy()

    all_inputs, all_labels, all_puzzle_ids = [], [], []
    collected = 0
    for batch_inputs, batch_labels, batch_pids, valid in ds:
        n = min(valid, args.num_samples - collected)
        all_inputs.append(batch_inputs[:n])
        all_labels.append(batch_labels[:n])
        all_puzzle_ids.append(batch_pids[:n])
        collected += n
        if collected >= args.num_samples:
            break

    inputs_t = torch.cat(all_inputs).to(exp.device)
    labels_t = torch.cat(all_labels)
    puzzle_ids_t = torch.cat(all_puzzle_ids).to(exp.device)
    N = inputs_t.shape[0]

    sample_instance_ids = []
    sample_aug_ids = []
    for i in range(N):
        inst = int(np.searchsorted(instance_bounds[1:], i, side="left"))
        aug = i - int(instance_bounds[inst])
        sample_instance_ids.append(inst)
        sample_aug_ids.append(aug)

    labels_np = labels_t.numpy()
    grid_shape = (30, 30)

    out_dir = maze_dir / "results" / args.exp / str(args.step)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process in chunks to avoid OOM
    chunk = 128
    print(f"Running {N} samples for {args.max_steps} ACT steps (chunk={chunk})...")
    for c_start in range(0, N, chunk):
        c_end = min(c_start + chunk, N)
        sl = slice(c_start, c_end)

        with torch.no_grad():
            preds_chunk, qhalt_chunk = run_model_steps(
                exp,
                inputs_t[sl],
                puzzle_ids_t[sl],
                args.max_steps,
            )

        save_individual(
            preds_per_step=preds_chunk,
            qhalt_per_step=qhalt_chunk,
            labels=labels_np[sl],
            instance_ids=sample_instance_ids[c_start:c_end],
            aug_ids=sample_aug_ids[c_start:c_end],
            grid_shape=grid_shape,
            max_steps=args.max_steps,
            out_dir=out_dir,
            split=args.split,
        )

    print(f"Done. {N} images saved to {out_dir}")


if __name__ == "__main__":
    main()
