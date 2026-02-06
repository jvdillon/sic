"""Unit tests for data.py."""

from __future__ import annotations

from pathlib import Path

import json
import tempfile

from util import set_seed

import numpy as np
import torch

from data import PuzzleDataset, PuzzleDatasetLegacy, augment_sudoku, load_puzzle_dataset


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


class TestBackwardCompatibility:
    """Verify new loader matches old loader behavior exactly."""

    def test_stratified_sampling_matches_old_loader(self):
        """PuzzleDatasetLegacy produces exact same indices as old loader for Sudoku."""
        n_puzzles, augs_per_puzzle, batch_size = 10, 100, 4

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            # Run legacy loader (preserves b4b compat with randint)
            set_seed(42)
            loader = PuzzleDatasetLegacy(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                dtype=torch.float32,
                batch_size=batch_size,
                train=True,
                shuffle=False,  # No shuffle so we can compare indices directly
                stratified=True,
            )
            new_indices = []
            for inputs, _, vc in loader:
                new_indices.extend(inputs[:vc, 0].tolist())

            # Run old loader logic manually (from origin/main)
            set_seed(42)
            group_starts = torch.arange(0, n_puzzles * augs_per_puzzle, augs_per_puzzle)
            aug_offsets = torch.randint(0, augs_per_puzzle, (n_puzzles,))
            old_indices = (group_starts + aug_offsets).tolist()

            assert new_indices == old_indices, (
                f"Legacy loader indices don't match old loader!\n"
                f"New: {new_indices}\nOld: {old_indices}"
            )

    def test_non_stratified_matches_old_loader(self):
        """Non-stratified mode produces same sequential iteration."""
        n_puzzles, augs_per_puzzle, batch_size = 5, 4, 3

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)
            n_samples = n_puzzles * augs_per_puzzle

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=False,  # Non-stratified
                shuffle=False,
                stratified=False,
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

    def test_stratified_visits_each_group_once(self):
        n_puzzles, augs_per_puzzle, batch_size = 8, 50, 3

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=False,
                stratified=True,
            )

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            # Should have exactly n_puzzles samples
            assert len(all_indices) == n_puzzles

            # Each index should map to a different group
            groups = [idx // augs_per_puzzle for idx in all_indices]
            assert sorted(groups) == list(range(n_puzzles))


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

    def test_stratified_iteration(self):
        n_puzzles, augs_per_puzzle, batch_size = 6, 20, 4

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_maze_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=True,
                shuffle=False,
                stratified=True,
            )

            all_indices = []
            for inputs, _, _, vc in loader:
                all_indices.extend(inputs[:vc, 0].tolist())

            assert len(all_indices) == n_puzzles
            groups = [idx // augs_per_puzzle for idx in all_indices]
            assert sorted(groups) == list(range(n_puzzles))


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
        """Verify puzzle_ids correctly identifies which puzzle each example belongs to."""
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
                stratified=True,
            )

            for inputs, _, puzzle_ids, vc in loader:
                for i in range(vc):
                    example_idx = inputs[i, 0].item()
                    puzzle_id = puzzle_ids[i].item()
                    # Example index should be in the range for this puzzle
                    expected_puzzle = example_idx // examples_per_puzzle
                    assert puzzle_id == expected_puzzle, (
                        f"Example {example_idx} should have puzzle_id {expected_puzzle}, "
                        f"got {puzzle_id}"
                    )

    def test_non_stratified_iterates_all_examples_sequentially(self):
        """With stratified=False, iterates through all examples in order."""
        n_groups, puzzles_per_group, examples_per_puzzle = 3, 2, 4
        batch_size = 5

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_arc_dataset(
                Path(tmp),
                n_groups,
                puzzles_per_group,
                examples_per_puzzle,
            )

            n_examples = n_groups * puzzles_per_group * examples_per_puzzle

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=False,
                shuffle=False,
                stratified=False,
            )

            all_example_indices = []
            all_puzzle_ids = []
            for inputs, _, puzzle_ids, vc in loader:
                all_example_indices.extend(inputs[:vc, 0].tolist())
                all_puzzle_ids.extend(puzzle_ids[:vc].tolist())

            # Should iterate through all examples in order
            assert all_example_indices == list(range(n_examples))

            # Puzzle IDs should correctly map each example to its puzzle
            for example_idx, puzzle_id in zip(
                all_example_indices,
                all_puzzle_ids,
                strict=True,
            ):
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
                stratified=False,
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
        """Test that gpu_cache=False keeps data on CPU until batch time."""
        n_puzzles, augs_per_puzzle, batch_size = 3, 2, 2

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _create_sudoku_dataset(Path(tmp), n_puzzles, augs_per_puzzle)

            loader = PuzzleDataset(
                data_dir=str(data_dir),
                device=torch.device("cpu"),
                batch_size=batch_size,
                train=False,
                shuffle=False,
                stratified=False,
                gpu_cache=False,
            )

            # Data should be on CPU
            assert loader.inputs.device == torch.device("cpu")

            # Iteration should still work
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
    """Create a Sudoku-style dataset on disk (both train and test splits).

    File structure mirrors real datasets in /opt/scratch/datasets/:
      {split}/all__inputs.npy        - [n_samples, seq_len] int32
      {split}/all__labels.npy        - [n_samples, seq_len] int32
      {split}/all__group_indices.npy - [n_groups + 1] boundary array
      {split}/all__puzzle_indices.npy - [n_samples + 1] sequential (single-example)
      {split}/dataset.json           - {vocab_size, seq_len}
    """
    n_samples = n_puzzles * augs_per_puzzle

    for split in ["train", "test"]:
        split_path = tmp_path / split
        split_path.mkdir()

        # Create data where sample i has value i in first position (for easy tracking)
        inputs = np.arange(n_samples, dtype=np.int32).reshape(-1, 1)
        inputs = np.broadcast_to(inputs, (n_samples, seq_len)).copy()
        labels = inputs + 100

        np.save(split_path / "all__inputs.npy", inputs)
        np.save(split_path / "all__labels.npy", labels)

        # Group indices: [0, augs, 2*augs, ..., n_samples]
        group_indices = np.arange(0, n_samples + 1, augs_per_puzzle)
        np.save(split_path / "all__group_indices.npy", group_indices)

        # Puzzle indices: for single-example, just [0, 1, 2, ..., n_samples]
        puzzle_indices = np.arange(n_samples + 1)
        np.save(split_path / "all__puzzle_indices.npy", puzzle_indices)

        with (split_path / "dataset.json").open("w") as f:
            json.dump({"vocab_size": vocab_size, "seq_len": seq_len}, f)

    return tmp_path


def _create_maze_dataset(
    tmp_path: Path,
    n_puzzles: int,
    augs_per_puzzle: int,
    seq_len: int = 225,  # 15x15 maze
    vocab_size: int = 5,
) -> Path:
    """Create a Maze-style dataset on disk (both train and test splits).

    Maze format is identical to Sudoku, just different seq_len/vocab_size.
    """
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
    seq_len: int = 900,  # 30x30 grid
    vocab_size: int = 11,
) -> Path:
    """Create an ARC-style dataset on disk (multi-example, both train and test).

    ARC differs from Sudoku/Maze:
      - Multiple examples per puzzle (input-output pairs)
      - puzzle_indices has gaps (non-sequential)
      - puzzle_identifiers maps puzzle index to ID

    File structure:
      {split}/all__inputs.npy             - [n_examples, seq_len]
      {split}/all__labels.npy             - [n_examples, seq_len]
      {split}/all__puzzle_identifiers.npy - [n_examples] puzzle ID per sample
      {split}/all__group_indices.npy      - [n_groups + 1] group boundaries
      {split}/all__puzzle_indices.npy     - [n_puzzles + 1] example boundaries
      {split}/dataset.json                - {vocab_size, seq_len}
    """
    n_puzzles = n_groups * puzzles_per_group
    n_examples = n_puzzles * examples_per_puzzle

    for split in ["train", "test"]:
        split_path = tmp_path / split
        split_path.mkdir()

        # Create data where example i has value i in first position
        inputs = np.arange(n_examples, dtype=np.int32).reshape(-1, 1)
        inputs = np.broadcast_to(inputs, (n_examples, seq_len)).copy()
        labels = inputs + 100

        np.save(split_path / "all__inputs.npy", inputs)
        np.save(split_path / "all__labels.npy", labels)

        # Puzzle identifiers: one per sample, maps sample -> puzzle ID
        puzzle_identifiers = np.repeat(
            np.arange(n_puzzles, dtype=np.int32), examples_per_puzzle
        )
        np.save(split_path / "all__puzzle_identifiers.npy", puzzle_identifiers)

        # Group indices: maps group -> first puzzle index
        # [0, puzzles_per_group, 2*puzzles_per_group, ..., n_puzzles]
        group_indices = np.arange(0, n_puzzles + 1, puzzles_per_group)
        np.save(split_path / "all__group_indices.npy", group_indices)

        # Puzzle indices: maps puzzle -> first example index
        # [0, examples_per_puzzle, 2*examples_per_puzzle, ..., n_examples]
        puzzle_indices = np.arange(0, n_examples + 1, examples_per_puzzle)
        np.save(split_path / "all__puzzle_indices.npy", puzzle_indices)

        with (split_path / "dataset.json").open("w") as f:
            json.dump({"vocab_size": vocab_size, "seq_len": seq_len}, f)

    return tmp_path


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
