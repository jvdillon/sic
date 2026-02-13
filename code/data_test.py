"""Unit tests for data.py."""

from __future__ import annotations

from pathlib import Path

import json
import tempfile

import numpy as np
import torch

from data import PuzzleDataset, augment_sudoku, load_puzzle_dataset


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


class TestBundleSampling:
    """Tests for augmentation_random_bundle_max_size sampling."""

    def test_bundle_1_visits_all_samples(self):
        """bundle_max_size=1: all samples visited, fully shuffled."""
        n_puzzles, augs_per_puzzle, batch_size = 5, 4, 3

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)
            n_samples = n_puzzles * augs_per_puzzle

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=False,
                augmentation_random_bundle_max_size=1,
            )

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            assert sorted(all_indices) == list(range(n_samples))

    def test_bundle_max_groups_contiguous(self):
        """bundle_max_size=augs: all augs of an instance are contiguous."""
        n_puzzles, augs_per_puzzle, batch_size = 4, 8, 100

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=False,
                augmentation_random_bundle_max_size=augs_per_puzzle,
            )

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            # All samples visited
            assert sorted(all_indices) == list(range(n_puzzles * augs_per_puzzle))

            # Each group of augs_per_puzzle consecutive indices should
            # belong to the same instance
            for i in range(0, len(all_indices), augs_per_puzzle):
                group = all_indices[i : i + augs_per_puzzle]
                instances = {idx // augs_per_puzzle for idx in group}
                assert len(instances) == 1, (
                    f"Group at position {i} has mixed instances: {group}"
                )

    def test_bundle_max_shuffled_contiguous(self):
        """shuffle=True + bundle_max_size=augs: augs still contiguous."""
        n_puzzles, augs_per_puzzle, batch_size = 20, 8, 100

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=True,
                augmentation_random_bundle_max_size=augs_per_puzzle,
            )

            torch.manual_seed(42)
            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            assert sorted(all_indices) == list(range(n_puzzles * augs_per_puzzle))

            for i in range(0, len(all_indices), augs_per_puzzle):
                group = all_indices[i : i + augs_per_puzzle]
                instances = {idx // augs_per_puzzle for idx in group}
                assert len(instances) == 1, (
                    f"Group at position {i} has mixed instances: {group}"
                )

    def test_bundle_partial_visits_all(self):
        """bundle_max_size that doesn't evenly divide num_augs still visits all."""
        n_puzzles, augs_per_puzzle, batch_size = 3, 8, 100

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=False,
                augmentation_random_bundle_max_size=3,
            )

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            assert sorted(all_indices) == list(range(n_puzzles * augs_per_puzzle))

    def test_sequential_no_shuffle(self):
        """shuffle=False with train=False iterates all in order."""
        n_puzzles, augs_per_puzzle, batch_size = 5, 4, 3

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)
            n_samples = n_puzzles * augs_per_puzzle

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=False,
                shuffle=False,
            )

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            assert all_indices == list(range(n_samples))


class TestSudokuDataset:
    """Tests for Sudoku dataset format."""

    def test_loads_correct_shapes(self):
        n_puzzles, augs_per_puzzle, seq_len = 5, 10, 81

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(
                Path(tmp),
                n_puzzles,
                augs_per_puzzle,
                seq_len,
            )

            data = load_puzzle_dataset(str(data_dir), "train")

            assert data["inputs"].shape == (n_puzzles * augs_per_puzzle, seq_len)
            assert data["labels"].shape == (n_puzzles * augs_per_puzzle, seq_len)
            assert data["vocab_size"] == 11
            assert data["seq_len"] == seq_len


class TestMazeDataset:
    """Tests for Maze dataset format."""

    def test_loads_correct_shapes(self):
        n_puzzles, augs_per_puzzle, seq_len = 5, 10, 225

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_maze_dataset(
                Path(tmp),
                n_puzzles,
                augs_per_puzzle,
                seq_len,
            )

            data = load_puzzle_dataset(str(data_dir), "train")

            assert data["inputs"].shape == (n_puzzles * augs_per_puzzle, seq_len)
            assert data["seq_len"] == seq_len
            assert data["vocab_size"] == 5


class TestARCDataset:
    """Tests for ARC dataset format (multi-example)."""

    def test_loads_correct_shapes(self):
        n_groups, puzzles_per_group, examples_per_puzzle = 3, 4, 5
        seq_len = 900

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_arc_dataset(
                Path(tmp),
                n_groups,
                puzzles_per_group,
                examples_per_puzzle,
                seq_len,
            )

            data = load_puzzle_dataset(str(data_dir), "train")

            n_puzzles = n_groups * puzzles_per_group
            n_examples = n_puzzles * examples_per_puzzle

            assert data["inputs"].shape == (n_examples, seq_len)
            assert data["puzzle_identifiers"] is not None
            assert data["puzzle_identifiers"].shape == (n_examples,)

    def test_puzzle_ids_correct_for_examples(self):
        n_groups, puzzles_per_group, examples_per_puzzle = 2, 2, 3
        batch_size = 10

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_arc_dataset(
                Path(tmp),
                n_groups,
                puzzles_per_group,
                examples_per_puzzle,
            )

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=False,
            )

            for inputs, _, puzzle_ids, vc in loader:
                for i in range(vc):
                    example_idx = inputs[i, 0].item()
                    puzzle_id = puzzle_ids[i].item()
                    expected_puzzle = example_idx // examples_per_puzzle
                    assert puzzle_id == expected_puzzle


class TestPuzzleDataset:
    """General tests for PuzzleDataset."""

    def test_padding_last_batch(self):
        n_puzzles, augs_per_puzzle, batch_size = 3, 2, 4

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=False,
                shuffle=False,
            )

            batches = list(loader)
            assert len(batches) == 2  # ceil(6/4) = 2

            # First batch: 4 valid
            inputs, _, _, vc = batches[0]
            assert inputs.shape[0] == batch_size
            assert vc == 4

            # Second batch: 2 valid, 2 padded
            inputs, _, _, vc = batches[1]
            assert inputs.shape[0] == batch_size
            assert vc == 2

    def test_gpu_cache_false(self):
        n_puzzles, augs_per_puzzle, batch_size = 3, 2, 2

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=False,
                shuffle=False,
                gpu_cache=False,
            )

            assert loader.inputs.device == torch.device("cpu")

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            assert sorted(all_indices) == list(range(n_puzzles * augs_per_puzzle))


# --- Private test helpers ---


def _create_sudoku_dataset(
    tmp_path: Path,
    n_puzzles: int,
    augs_per_puzzle: int,
    seq_len: int = 81,
    vocab_size: int = 11,
) -> Path:
    n_samples = n_puzzles * augs_per_puzzle

    for split in ["train", "test"]:
        split_path = tmp_path / split
        split_path.mkdir()

        inputs = np.arange(n_samples, dtype=np.int32).reshape(-1, 1)
        inputs = np.broadcast_to(inputs, (n_samples, seq_len)).copy()
        labels = inputs + 100

        np.save(split_path / "all__inputs.npy", inputs)
        np.save(split_path / "all__labels.npy", labels)

        group_indices = np.arange(0, n_samples + 1, augs_per_puzzle)
        np.save(split_path / "all__group_indices.npy", group_indices)

        puzzle_indices = np.arange(n_samples + 1)
        np.save(split_path / "all__puzzle_indices.npy", puzzle_indices)

        with (split_path / "dataset.json").open("w") as f:
            json.dump({"vocab_size": vocab_size, "seq_len": seq_len}, f)

    return tmp_path


def _create_maze_dataset(
    tmp_path: Path,
    n_puzzles: int,
    augs_per_puzzle: int,
    seq_len: int = 225,
    vocab_size: int = 5,
) -> Path:
    return _create_sudoku_dataset(
        tmp_path,
        n_puzzles,
        augs_per_puzzle,
        seq_len,
        vocab_size,
    )


def _create_arc_dataset(
    tmp_path: Path,
    n_groups: int,
    puzzles_per_group: int,
    examples_per_puzzle: int,
    seq_len: int = 900,
    vocab_size: int = 11,
) -> Path:
    n_puzzles = n_groups * puzzles_per_group
    n_examples = n_puzzles * examples_per_puzzle

    for split in ["train", "test"]:
        split_path = tmp_path / split
        split_path.mkdir()

        inputs = np.arange(n_examples, dtype=np.int32).reshape(-1, 1)
        inputs = np.broadcast_to(inputs, (n_examples, seq_len)).copy()
        labels = inputs + 100

        np.save(split_path / "all__inputs.npy", inputs)
        np.save(split_path / "all__labels.npy", labels)

        puzzle_identifiers = np.repeat(
            np.arange(n_puzzles, dtype=np.int32), examples_per_puzzle
        )
        np.save(split_path / "all__puzzle_identifiers.npy", puzzle_identifiers)

        group_indices = np.arange(0, n_puzzles + 1, puzzles_per_group)
        np.save(split_path / "all__group_indices.npy", group_indices)

        puzzle_indices = np.arange(0, n_examples + 1, examples_per_puzzle)
        np.save(split_path / "all__puzzle_indices.npy", puzzle_indices)

        with (split_path / "dataset.json").open("w") as f:
            json.dump({"vocab_size": vocab_size, "seq_len": seq_len}, f)

    return tmp_path


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
