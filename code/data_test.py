"""Unit tests for data.py."""

from __future__ import annotations

import torch

from research.projects.trm3.data import augment_sudoku


def test_augment_sudoku_shapes():
    inputs = torch.randint(1, 10, (4, 81))
    labels = torch.randint(1, 10, (4, 81))
    inputs_aug, labels_aug = augment_sudoku(inputs, labels)
    assert inputs_aug.shape == (4, 81)
    assert labels_aug.shape == (4, 81)


def test_augment_sudoku_valid_range():
    inputs = torch.randint(1, 10, (4, 81))
    labels = torch.randint(1, 10, (4, 81))
    inputs_aug, labels_aug = augment_sudoku(inputs, labels)
    assert inputs_aug.min() >= 0
    assert inputs_aug.max() <= 10
    assert labels_aug.min() >= 0
    assert labels_aug.max() <= 10


def test_augment_sudoku_deterministic_with_seed():
    inputs = torch.randint(1, 10, (4, 81))
    labels = torch.randint(1, 10, (4, 81))

    torch.manual_seed(42)
    out1 = augment_sudoku(inputs.clone(), labels.clone())

    torch.manual_seed(42)
    out2 = augment_sudoku(inputs.clone(), labels.clone())

    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


if __name__ == "__main__":
    from research.projects.trm3.util import test_main

    test_main(__file__)
