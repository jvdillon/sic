"""Sudoku data loading for TRM.

Provides GPUCachedSudoku for efficient GPU-cached training with stratified sampling.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypedDict

import json
import pathlib

from torch import Tensor

import numpy as np
import torch


class SudokuDataset(TypedDict):
    inputs: Tensor
    labels: Tensor
    vocab_size: int
    seq_len: int


def augment_sudoku(inputs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Apply all 8 dihedral symmetries + digit permutation."""
    B = inputs.shape[0]
    inputs_aug = inputs.clone()
    labels_aug = labels.clone()
    for i in range(B):
        perm = torch.zeros(11, device=inputs.device, dtype=torch.long)
        perm[1:10] = torch.randperm(9, device=inputs.device) + 1
        perm[10] = 10
        inputs_aug[i] = perm[inputs[i].long()].to(inputs_aug.dtype)
        valid = labels[i] > 0
        if valid.any():
            labels_aug[i, valid] = perm[labels[i, valid].long()].to(labels_aug.dtype)

        k = int(torch.randint(low=0, high=4, size=(1,)).item())
        flip = int(torch.randint(low=0, high=2, size=(1,)).item())
        grid_in = inputs_aug[i].reshape(9, 9)
        grid_lab = labels_aug[i].reshape(9, 9)
        if flip:
            grid_in = grid_in.flip(1)
            grid_lab = grid_lab.flip(1)
        if k > 0:
            grid_in = grid_in.rot90(k)
            grid_lab = grid_lab.rot90(k)
        inputs_aug[i] = grid_in.reshape(81)
        labels_aug[i] = grid_lab.reshape(81)
    return inputs_aug, labels_aug


def load_sudoku_dataset(data_dir: str, split: str = "train") -> SudokuDataset:
    """Load Sudoku dataset into memory.

    Returns dict with:
        inputs: (N, 81) int32 tensor, values 1-10 (1=blank, 2-10=digits 1-9)
        labels: (N, 81) int32 tensor, values 1-10
        vocab_size: int
        seq_len: int
    """
    data_path = pathlib.Path(data_dir).expanduser() / split

    with (data_path / "dataset.json").open() as f:
        metadata = json.load(f)

    inputs = np.load(data_path / "all__inputs.npy")
    labels = np.load(data_path / "all__labels.npy")

    return {
        "inputs": torch.from_numpy(inputs).to(torch.int32),
        "labels": torch.from_numpy(labels).to(torch.int32),
        "vocab_size": metadata["vocab_size"],
        "seq_len": metadata["seq_len"],
    }


class GPUCachedSudoku:
    """GPU-cached Sudoku dataset with stratified sampling.

    TRM uses stratified sampling: each epoch visits each base puzzle exactly once,
    randomly picking one augmentation per puzzle. This ensures batch diversity.
    """

    def __init__(
        self,
        data_dir: str,
        device: torch.device,
        dtype: torch.dtype | None,
        batch_size: int,
        train: bool = True,
        shuffle: bool = True,
        stratified: bool = True,
    ):
        split = "train" if train else "test"
        data = load_sudoku_dataset(data_dir, split)

        self.inputs = data["inputs"].to(device)
        self.labels = data["labels"].to(device)
        self.vocab_size = data["vocab_size"]
        self.seq_len = data["seq_len"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stratified = stratified and train
        self.device = device
        self.dtype = dtype

        self.n = len(self.inputs)

        if self.stratified:
            data_path = pathlib.Path(data_dir).expanduser() / split
            group_indices = np.load(data_path / "all__group_indices.npy")
            self.group_starts = torch.from_numpy(group_indices[:-1]).to(device)
            self.n_groups = len(self.group_starts)
            self.augs_per_group = int(group_indices[1] - group_indices[0])

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, int]]:
        if self.stratified:
            aug_offsets = torch.randint(
                low=0,
                high=self.augs_per_group,
                size=(self.n_groups,),
                device=self.device,
            )
            indices = self.group_starts + aug_offsets
            if self.shuffle:
                perm = torch.randperm(self.n_groups, device=self.device)
                indices = indices[perm]
            for i in range(0, self.n_groups, self.batch_size):
                idx = indices[i : i + self.batch_size]
                valid_count = len(idx)
                if valid_count < self.batch_size:
                    pad_size = self.batch_size - valid_count
                    inputs_pad = torch.zeros(
                        pad_size,
                        *self.inputs.shape[1:],
                        device=self.device,
                        dtype=self.inputs.dtype,
                    )
                    labels_pad = torch.zeros(
                        pad_size,
                        *self.labels.shape[1:],
                        device=self.device,
                        dtype=self.labels.dtype,
                    )
                    yield (
                        torch.cat([self.inputs[idx], inputs_pad]),
                        torch.cat([self.labels[idx], labels_pad]),
                        valid_count,
                    )
                else:
                    yield (self.inputs[idx], self.labels[idx], valid_count)
        else:
            if self.shuffle:
                indices = torch.randperm(self.n, device=self.device)
            else:
                indices = torch.arange(self.n, device=self.device)
            for i in range(0, self.n, self.batch_size):
                idx = indices[i : i + self.batch_size]
                valid_count = len(idx)
                if valid_count < self.batch_size:
                    pad_size = self.batch_size - valid_count
                    inputs_pad = torch.zeros(
                        pad_size,
                        *self.inputs.shape[1:],
                        device=self.device,
                        dtype=self.inputs.dtype,
                    )
                    labels_pad = torch.zeros(
                        pad_size,
                        *self.labels.shape[1:],
                        device=self.device,
                        dtype=self.labels.dtype,
                    )
                    yield (
                        torch.cat([self.inputs[idx], inputs_pad]),
                        torch.cat([self.labels[idx], labels_pad]),
                        valid_count,
                    )
                else:
                    yield (self.inputs[idx], self.labels[idx], valid_count)
