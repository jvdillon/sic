"""Evaluate a single checkpoint on train and test sets."""

from pathlib import Path

import sys

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    exp_name = sys.argv[1]
    step = int(sys.argv[2])
    ckpt_dir = Path(__file__).parent / "ckpts" / exp_name
    path = ckpt_dir / f"step{step:05d}.pt"

    mod = __import__(f"maze.{exp_name}", fromlist=["Experiment"])
    exp = mod.Experiment()
    exp.model.to(exp.device)

    ckpt = torch.load(path, map_location=exp.device, weights_only=False)
    exp.model.load_state_dict(ckpt["model"])
    exp.model.eval()

    print(f"=== {exp_name} step {step} TEST ===")
    exp.evaluate_act(iter(exp.make_test_loader()))
    print(f"=== {exp_name} step {step} TRAIN ===")
    exp.evaluate_act(iter(exp.make_train_loader()))


if __name__ == "__main__":
    main()
