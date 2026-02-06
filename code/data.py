"""Puzzle data loading for TRM.

Glossary:
  task        : Problem type, e.g., "maze", "sudoku", "007bbfb7" (ARC task ID)
  instance    : One puzzle (one base maze, one sudoku grid, one ARC problem, etc)
  augmentation: One transformed version of an instance (rotation, digit perm, etc.)
  sample      : One row in the data arrays = one (instance, augmentation) pair

Data on disk is a flattened cross-product: {(instance, aug) for each task}.

Files:
  all__inputs.npy            : [n_samples, seq_len] - input sequences
  all__labels.npy            : [n_samples, seq_len] - target sequences
  all__group_indices.npy     : [n_instances+1] - instance boundaries (we prefer "instance" vs "group")
  all__puzzle_identifiers.npy: [n_samples] - task ID per sample (zeros for maze/sudoku)

Data structure for Maze (1 task, 1000 instances, 8 augs each):

    all__inputs.npy [8000 rows]
    ┌─────────────────────────────┐
    │ row 0: aug 0 of instance 0  │
    │ row 1: aug 1 of instance 0  │
    │ ...                         │
    │ row 7: aug 7 of instance 0  │
    ├─────────────────────────────┤
    │ row 8: aug 0 of instance 1  │
    │ ...                         │
    │ row 15: aug 7 of instance 1 │
    ├─────────────────────────────┤
    │ ...                         │
    ├─────────────────────────────┤
    │ row 7992: aug 0 of inst 999 │
    │ ...                         │
    │ row 7999: aug 7 of inst 999 │
    └─────────────────────────────┘

    all__group_indices.npy = [0, 8, 16, 24, ..., 7992, 8000]
                              │  │   │   │        │     │
                              │  │   │   │        │     └─ end of instance 999
                              │  │   │   │        └─ start of instance 999
                              │  │   │   └─ start of instance 3
                              │  │   └─ start of instance 2
                              │  └─ start of instance 1
                              └─ start of instance 0

Stratified sampling (one aug per instance per epoch):

    epoch 1:  [rand(0,8), rand(8,16), rand(16,24), ...]  → 1000 samples
               ~~~~~~~~   ~~~~~~~~~   ~~~~~~~~~~
               pick one   pick one    pick one
               from       from        from
               inst 0     inst 1      inst 2

    epoch 2:  [rand(0,8), rand(8,16), rand(16,24), ...]  → 1000 different samples

Non-stratified: all 8000 samples each epoch.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypedDict, get_args

import json
import math
import pathlib

from torch import Tensor

import numpy as np
import torch


PuzzleType = Literal["sudoku", "maze", "arc"]


class PuzzleData(TypedDict):
    inputs: Tensor
    labels: Tensor
    puzzle_identifiers: Tensor | None
    group_indices: Tensor
    vocab_size: int
    seq_len: int


@dataclass(kw_only=True, frozen=True, slots=True)
class PuzzleConfig:
    grid_shape: tuple[int, ...]  # (9, 9) for sudoku, (30, 30) for maze, (900,) for ARC
    vocab_size: int
    mask_token: int | None = None

    @property
    def grid_len(self) -> int:
        return math.prod(self.grid_shape)


_PUZZLE_CONFIGS: dict[PuzzleType, PuzzleConfig] = {
    "sudoku": PuzzleConfig(grid_shape=(9, 9), vocab_size=12, mask_token=10),
    "maze": PuzzleConfig(grid_shape=(30, 30), vocab_size=12),
    "arc": PuzzleConfig(grid_shape=(900,), vocab_size=12),
}


def get_puzzle_config(puzzle_type: PuzzleType) -> PuzzleConfig:
    if puzzle_type not in _PUZZLE_CONFIGS:
        valid = ", ".join(get_args(PuzzleType))
        raise ValueError(f"Unknown puzzle type: {puzzle_type}. Valid: {valid}")
    return _PUZZLE_CONFIGS[puzzle_type]


def _build_dihedral_indices(n: int, device: torch.device) -> Tensor:
    base = torch.arange(n * n, device=device).reshape(n, n)
    symmetries = []
    for k in range(4):
        rotated = torch.rot90(base, k)
        symmetries.append(rotated.reshape(-1))
        symmetries.append(rotated.flip(1).reshape(-1))
    return torch.stack(symmetries)


def augment_sudoku(inputs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Apply random dihedral symmetry and digit permutation to sudoku puzzles."""
    cfg = get_puzzle_config("sudoku")
    n = cfg.grid_shape[0]  # side length (9 for sudoku)
    B = inputs.shape[0]
    device = inputs.device

    perms = torch.zeros(B, cfg.vocab_size, device=device, dtype=torch.long)
    perms[:, 0] = 0
    assert cfg.mask_token is not None
    perms[:, cfg.mask_token] = cfg.mask_token
    rand_vals = torch.rand(B, n, device=device)
    perms[:, 1:10] = rand_vals.argsort(dim=1) + 1

    inputs_aug = torch.gather(perms, 1, inputs.long()).to(inputs.dtype)
    labels_aug = torch.gather(perms, 1, labels.long()).to(labels.dtype)

    dihedral_indices = _build_dihedral_indices(n, device)
    sym_choice = torch.randint(0, 8, (B,), device=device)
    selected_indices = dihedral_indices[sym_choice]

    inputs_aug = torch.gather(inputs_aug, 1, selected_indices)
    labels_aug = torch.gather(labels_aug, 1, selected_indices)

    return inputs_aug, labels_aug


def load_puzzle_dataset(data_dir: str, split: str) -> PuzzleData:
    """Load puzzle data from disk.

    Returns dict with: inputs, labels, puzzle_identifiers, group_indices, vocab_size, seq_len.
    """
    data_path = pathlib.Path(data_dir).expanduser() / split

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    metadata_path = data_path / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    with metadata_path.open() as f:
        metadata = json.load(f)

    inputs = torch.from_numpy(np.load(data_path / "all__inputs.npy")).to(torch.int32)
    labels = torch.from_numpy(np.load(data_path / "all__labels.npy")).to(torch.int32)

    puzzle_ids_path = data_path / "all__puzzle_identifiers.npy"
    puzzle_ids = (
        torch.from_numpy(np.load(puzzle_ids_path)).to(torch.int32)
        if puzzle_ids_path.exists()
        else None
    )

    group_path = data_path / "all__group_indices.npy"
    if not group_path.exists():
        raise FileNotFoundError(f"Required file not found: {group_path}")
    group_indices = torch.from_numpy(np.load(group_path))

    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_ids,
        "group_indices": group_indices,
        "vocab_size": metadata["vocab_size"],
        "seq_len": metadata["seq_len"],
    }


class PuzzleDataset:
    """Puzzle dataset iterator for Maze, Sudoku, and ARC.

    Args:
        data_dir: Path to dataset directory.
        device: Torch device for output tensors.
        batch_size: Batch size.
        train: If True, load train split; else test.
        shuffle: If True, shuffle each epoch.
        stratified: If True, sample one aug per instance per epoch.
        gpu_cache: If True, keep data on GPU. Set False for large datasets (ARC).
        num_instances: Limit to first N instances (for ablations).

    Yields:
        (inputs, labels, puzzle_ids, valid_count) tuples.

    """

    def __init__(
        self,
        data_dir: str,
        device: torch.device,
        batch_size: int,
        train: bool = True,
        shuffle: bool = True,
        stratified: bool = True,
        gpu_cache: bool = True,
        num_instances: int | None = None,
    ):
        split = "train" if train else "test"
        data = load_puzzle_dataset(data_dir, split)

        inputs = data["inputs"]
        labels = data["labels"]
        puzzle_ids = data["puzzle_identifiers"]
        instance_bounds = data["group_indices"]
        self.vocab_size = data["vocab_size"]
        self.seq_len = data["seq_len"]

        # Limit instances if requested
        if num_instances is not None and num_instances < len(instance_bounds) - 1:
            max_samples = int(instance_bounds[num_instances])
            instance_bounds = instance_bounds[: num_instances + 1]
        else:
            max_samples = len(inputs)

        # Storage device
        self.device = device
        self.gpu_cache = gpu_cache
        storage = device if gpu_cache else torch.device("cpu")

        self.inputs = inputs[:max_samples].to(storage)
        self.labels = labels[:max_samples].to(storage)
        self.puzzle_ids = (
            torch.zeros(max_samples, dtype=torch.int32, device=storage)
            if puzzle_ids is None
            else puzzle_ids[:max_samples].to(storage)
        )
        self.instance_bounds = instance_bounds.to(storage)

        self.n = len(self.inputs)
        self.n_instances = len(instance_bounds) - 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stratified = stratified and train

        # Check for uniform instance sizes.
        # We don't really need this code stanza except to preserve the RNG
        # access pattern of the existing implementation which was used for the
        # Sudoku results.
        if self.stratified:
            instance_sizes = instance_bounds[1:] - instance_bounds[:-1]
            if (instance_sizes == instance_sizes[0]).all():
                self._uniform_instance_size: int | None = int(instance_sizes[0])
            else:
                self._uniform_instance_size = None

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        device = self.instance_bounds.device

        if self.stratified:
            starts = self.instance_bounds[:-1]
            if self._uniform_instance_size is None:
                # Variable instances: use rand * sizes
                sizes = self.instance_bounds[1:] - starts
                offsets = (torch.rand(self.n_instances, device=device) * sizes).long()
            else:
                # Uniform num instances case.
                # We don't really need this case but it preserves the RNG
                # access pattern of the existing implementation which was used
                # for the Sudoku results.
                offsets = torch.randint(
                    0,
                    self._uniform_instance_size,
                    (self.n_instances,),
                    device=device,
                )
            indices = starts + offsets
            if self.shuffle:
                indices = indices[torch.randperm(self.n_instances, device=device)]
            n_samples = self.n_instances
        else:
            if self.shuffle:
                indices = torch.randperm(self.n, device=device)
            else:
                indices = torch.arange(self.n, device=device)
            n_samples = self.n

        for i in range(0, n_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            valid = len(batch_idx)

            inputs = self.inputs[batch_idx]
            labels = self.labels[batch_idx]
            puzzle_ids = self.puzzle_ids[batch_idx]

            if valid < self.batch_size:
                pad = self.batch_size - valid
                inputs = torch.cat([inputs, inputs.new_zeros(pad, inputs.shape[1])])
                labels = torch.cat([labels, labels.new_zeros(pad, labels.shape[1])])
                puzzle_ids = torch.cat([puzzle_ids, puzzle_ids.new_zeros(pad)])

            if not self.gpu_cache:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                puzzle_ids = puzzle_ids.to(self.device)

            yield inputs, labels, puzzle_ids, valid
