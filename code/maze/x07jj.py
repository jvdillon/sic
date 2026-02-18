"""x07j: x07a + aug data, bundle_size=8, anchor_seq_index=1, AC on H-cycles."""

from __future__ import annotations

from experiment import main
from maze.x07a import Experiment as Experiment07a


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    data_dir: str = "/opt/scratch/datasets/maze-30x30-hard-1k-aug"
    augmentation_random_bundle_max_size: int = 8


if __name__ == "__main__":
    main(Experiment())
