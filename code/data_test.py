"""Unit tests for data.py."""

from __future__ import annotations

from unittest import mock

import torch

from data import augment_sudoku, GPUCachedSudoku


def test_augment_sudoku_shapes():
    inputs = torch.randint(low=1, high=10, size=(4, 81))
    labels = torch.randint(low=1, high=10, size=(4, 81))
    inputs_aug, labels_aug = augment_sudoku(inputs, labels)
    assert inputs_aug.shape == (4, 81)
    assert labels_aug.shape == (4, 81)


def test_augment_sudoku_valid_range():
    inputs = torch.randint(low=1, high=10, size=(4, 81))
    labels = torch.randint(low=1, high=10, size=(4, 81))
    inputs_aug, labels_aug = augment_sudoku(inputs, labels)
    assert inputs_aug.min() >= 0
    assert inputs_aug.max() <= 10
    assert labels_aug.min() >= 0
    assert labels_aug.max() <= 10


def test_augment_sudoku_deterministic_with_seed():
    inputs = torch.randint(low=1, high=10, size=(4, 81))
    labels = torch.randint(low=1, high=10, size=(4, 81))

    torch.manual_seed(42)
    out1 = augment_sudoku(inputs.clone(), labels.clone())

    torch.manual_seed(42)
    out2 = augment_sudoku(inputs.clone(), labels.clone())

    assert torch.equal(out1[0], out2[0])
    assert torch.equal(out1[1], out2[1])


def _make_mock_dataset(n_samples: int, seq_len: int = 81):
    """Create mock dataset dict for testing."""
    return {
        "inputs": torch.arange(n_samples).unsqueeze(1).expand(-1, seq_len),
        "labels": torch.arange(n_samples).unsqueeze(1).expand(-1, seq_len) + 100,
        "vocab_size": 11,
        "seq_len": seq_len,
    }


def _make_mock_group_indices(n_groups: int, augs_per_group: int):
    """Create group indices array for stratified sampling."""
    import numpy as np
    return np.arange(0, n_groups * augs_per_group + 1, augs_per_group)


class TestGPUCachedSudokuIter:
    """Tests for GPUCachedSudoku.__iter__."""

    def test_non_stratified_visits_all_samples(self):
        """Each sample visited exactly once per epoch."""
        n_samples, batch_size = 10, 3
        mock_data = _make_mock_dataset(n_samples)

        with mock.patch("data.load_sudoku_dataset", return_value=mock_data):
            loader = GPUCachedSudoku(
                data_dir="/fake",
                device=torch.device("cpu"),
                dtype=None,
                batch_size=batch_size,
                train=False,
                shuffle=False,
                stratified=False,
            )

        all_indices = []
        for inputs, _, valid_count in loader:
            all_indices.extend(inputs[:valid_count, 0].tolist())

        assert sorted(all_indices) == list(range(n_samples))

    def test_non_stratified_padding(self):
        """Final batch padded to batch_size with valid_count correct."""
        n_samples, batch_size = 10, 4
        mock_data = _make_mock_dataset(n_samples)

        with mock.patch("data.load_sudoku_dataset", return_value=mock_data):
            loader = GPUCachedSudoku(
                data_dir="/fake",
                device=torch.device("cpu"),
                dtype=None,
                batch_size=batch_size,
                train=False,
                shuffle=False,
                stratified=False,
            )

        batches = list(loader)
        assert len(batches) == 3  # ceil(10/4) = 3

        # Last batch should be padded
        inputs, _, valid_count = batches[-1]
        assert inputs.shape[0] == batch_size
        assert valid_count == 2  # 10 % 4 = 2

    def test_shuffle_changes_order(self):
        """Shuffle produces different orderings."""
        n_samples, batch_size = 20, 5
        mock_data = _make_mock_dataset(n_samples)

        with mock.patch("data.load_sudoku_dataset", return_value=mock_data):
            loader = GPUCachedSudoku(
                data_dir="/fake",
                device=torch.device("cpu"),
                dtype=None,
                batch_size=batch_size,
                train=False,
                shuffle=True,
                stratified=False,
            )

        torch.manual_seed(1)
        order1 = [inputs[:vc, 0].tolist() for inputs, _, vc in loader]

        torch.manual_seed(2)
        order2 = [inputs[:vc, 0].tolist() for inputs, _, vc in loader]

        assert order1 != order2  # Different seeds -> different order

    def test_stratified_picks_one_aug_per_group(self):
        """Stratified sampling visits each group exactly once."""
        n_groups, augs_per_group, batch_size = 5, 4, 2
        n_samples = n_groups * augs_per_group
        mock_data = _make_mock_dataset(n_samples)
        mock_group_idx = _make_mock_group_indices(n_groups, augs_per_group)

        with mock.patch("data.load_sudoku_dataset", return_value=mock_data), \
             mock.patch("numpy.load", return_value=mock_group_idx):
            loader = GPUCachedSudoku(
                data_dir="/fake",
                device=torch.device("cpu"),
                dtype=None,
                batch_size=batch_size,
                train=True,
                shuffle=False,
                stratified=True,
            )

        all_indices = []
        for inputs, _, valid_count in loader:
            all_indices.extend(inputs[:valid_count, 0].tolist())

        # Should have exactly n_groups samples (one per group)
        assert len(all_indices) == n_groups

        # Each index should be within its group's range
        groups_visited = [idx // augs_per_group for idx in all_indices]
        assert sorted(groups_visited) == list(range(n_groups))

    def test_stratified_different_augs_each_epoch(self):
        """Different epochs pick different augmentations."""
        n_groups, augs_per_group, batch_size = 5, 4, 10
        n_samples = n_groups * augs_per_group
        mock_data = _make_mock_dataset(n_samples)
        mock_group_idx = _make_mock_group_indices(n_groups, augs_per_group)

        with mock.patch("data.load_sudoku_dataset", return_value=mock_data), \
             mock.patch("numpy.load", return_value=mock_group_idx):
            loader = GPUCachedSudoku(
                data_dir="/fake",
                device=torch.device("cpu"),
                dtype=None,
                batch_size=batch_size,
                train=True,
                shuffle=False,
                stratified=True,
            )

        torch.manual_seed(1)
        indices1 = [inputs[:vc, 0].tolist() for inputs, _, vc in loader]

        torch.manual_seed(2)
        indices2 = [inputs[:vc, 0].tolist() for inputs, _, vc in loader]

        # Different seeds should pick different augmentations
        assert indices1 != indices2


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
