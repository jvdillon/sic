"""Evaluate early checkpoints on TRAIN data for x07, x07b, x07h."""

import importlib
import pathlib
import sys

import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from experiment import Experiment as ExperimentBase


CONFIGS: list[tuple[str, list[int]]] = [
    ("x07", [500, 1000, 1500, 2000, 2500]),
    ("x07b", [500, 1000, 1500, 2000, 2500]),
    ("x07h", [500, 1000, 1500, 2000, 2500]),
]


def main():
    maze_dir = pathlib.Path(__file__).resolve().parent

    for module_name, steps in CONFIGS:
        mod = importlib.import_module(module_name)
        ckpt_dir = maze_dir / "ckpts" / module_name

        for step in steps:
            ckpt_path = ckpt_dir / f"step{step:05d}.pt"
            if not ckpt_path.exists():
                print(f"SKIP {module_name} step {step}: no checkpoint")
                continue

            experiment: ExperimentBase = mod.Experiment()
            experiment.setup_model()
            ckpt = torch.load(
                ckpt_path, map_location=experiment.device, weights_only=False
            )
            experiment.model.load_state_dict(ckpt["model"])
            experiment.current_step = step

            trainloader = experiment.make_train_loader()
            experiment.model.eval()
            with torch.no_grad():
                cell_acc, puzzle_acc = experiment.evaluate_act(iter(trainloader))
            print(
                f"{module_name} step={step:5d}  "
                f"train_cell={cell_acc:.2f}%  "
                f"train_puzzle={puzzle_acc:.2f}%"
            )
            print()


if __name__ == "__main__":
    main()
