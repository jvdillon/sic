"""Puzzle data loading for TRM.

Provides PuzzleDatasetIterator for efficient GPU-cached training with stratified
sampling. Works with Sudoku, Maze, and ARC datasets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypedDict, get_args

import json
import pathlib

from torch import Tensor

import numpy as np
import torch


PuzzleType = Literal["sudoku", "maze", "arc"]


@dataclass(frozen=True, slots=True)
class PuzzleConfig:
    """Configuration for a puzzle type."""

    grid_size: int
    vocab_size: int
    mask_token: int | None = None  # None if puzzle type doesn't use masking

    @property
    def num_puzzle_grid_tokens(self) -> int:
        """Puzzle grid length (no HALT token)."""
        return self.grid_size * self.grid_size


_PUZZLE_CONFIGS: dict[PuzzleType, PuzzleConfig] = {
    # Sudoku: 0-9 digits (10) + mask (1) + halt (1) = 12 tokens
    "sudoku": PuzzleConfig(grid_size=9, vocab_size=12, mask_token=10),
    # Maze: tokens 0-5 (6 values) + reserved (5) + halt (1) = 12 tokens
    "maze": PuzzleConfig(grid_size=30, vocab_size=12),
    # ARC: PAD=0, EOS=1, colors 2-11 (10 colors for 0-9) = 12 tokens
    "arc": PuzzleConfig(grid_size=30, vocab_size=12),
}


def get_puzzle_config(puzzle_type: PuzzleType) -> PuzzleConfig:
    """Get configuration for a puzzle type.

    Args:
      puzzle_type: One of "sudoku", "maze", "arc".

    Returns:
      PuzzleConfig with grid_size, vocab_size, num_puzzle_grid_tokens, and mask_token.

    Raises:
      ValueError: If puzzle_type is not recognized.

    """
    if puzzle_type not in _PUZZLE_CONFIGS:
        valid = ", ".join(get_args(PuzzleType))
        raise ValueError(f"Unknown puzzle type: {puzzle_type}. Valid types: {valid}")
    return _PUZZLE_CONFIGS[puzzle_type]


class PuzzleDataset(TypedDict):
    """Dataset dict returned by load_puzzle_dataset.

    Fields:
      inputs: Input tensors [n_examples, seq_len], dtype int32.
      labels: Label tensors [n_examples, seq_len], dtype int32.
      vocab_size: Size of token vocabulary.
      seq_len: Sequence length.
      puzzle_identifiers: Optional puzzle IDs [n_puzzles], dtype int32.
        For single-example datasets, n_puzzles == n_examples.
        For multi-example datasets (ARC), n_puzzles < n_examples.
    """

    inputs: Tensor
    labels: Tensor
    vocab_size: int
    seq_len: int
    puzzle_identifiers: Tensor | None


def _build_dihedral_indices(n: int, device: torch.device) -> Tensor:
    """Build index mappings for 8 dihedral symmetries of an n x n grid.

    Returns:
      Tensor of shape [8, n*n] where each row is an index permutation.

    """
    base = torch.arange(n * n, device=device).reshape(n, n)
    symmetries = []
    for k in range(4):
        rotated = torch.rot90(base, k)
        symmetries.append(rotated.reshape(-1))
        symmetries.append(rotated.flip(1).reshape(-1))
    return torch.stack(symmetries)


def augment_sudoku(inputs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Apply random dihedral symmetry and digit permutation to each sudoku puzzle.

    Fully vectorized implementation with no per-sample loops or .item() calls.

    For each sample in the batch, independently applies:
    - A random permutation of digits 1-9 (0 and mask token 10 stay fixed)
    - A random element from the dihedral group D4 (4 rotations x 2 flips = 8 symmetries)

    Args:
      inputs: Input puzzles [batch_size, 81]. Values 0-10.
      labels: Label puzzles [batch_size, 81]. Values 0-10.

    Returns:
      Tuple of (augmented_inputs, augmented_labels) with same shapes.

    """
    cfg = get_puzzle_config("sudoku")
    B = inputs.shape[0]
    device = inputs.device

    # --- Vectorized digit permutation ---
    # Create batch of permutation tables [B, 11]. Token 0 and 10 stay fixed.
    # For tokens 1-9, apply independent random permutation per sample.
    perms = torch.zeros(B, cfg.vocab_size, device=device, dtype=torch.long)
    perms[:, 0] = 0
    assert cfg.mask_token is not None
    perms[:, cfg.mask_token] = cfg.mask_token
    # Generate B independent permutations of 1-9 using argsort of random values
    rand_vals = torch.rand(B, cfg.grid_size, device=device)
    perms[:, 1:10] = rand_vals.argsort(dim=1) + 1

    # Apply permutation: gather from perms using input values as indices.
    # perms[:, 0] = 0, so 0-valued inputs/labels stay 0 (no extra masking needed).
    inputs_aug = torch.gather(perms, 1, inputs.long()).to(inputs.dtype)
    labels_aug = torch.gather(perms, 1, labels.long()).to(labels.dtype)

    # --- Vectorized dihedral transformation ---
    # Sample one of 8 symmetries per sample
    dihedral_indices = _build_dihedral_indices(cfg.grid_size, device)  # [8, 81]
    sym_choice = torch.randint(0, 8, (B,), device=device)  # [B]
    selected_indices = dihedral_indices[sym_choice]  # [B, 81]

    # Apply spatial transformation via gather
    inputs_aug = torch.gather(inputs_aug, 1, selected_indices)
    labels_aug = torch.gather(labels_aug, 1, selected_indices)

    return inputs_aug, labels_aug


def load_puzzle_dataset(data_dir: str, split: str = "train") -> PuzzleDataset:
    """Load puzzle dataset into CPU memory.

    Args:
      data_dir: Path to dataset directory. Must contain a '{split}/' subdirectory
        with dataset.json, all__inputs.npy, and all__labels.npy files.
      split: "train" or "test".

    Returns:
      Dict with inputs, labels, vocab_size, seq_len, puzzle_identifiers.

    Raises:
      FileNotFoundError: If data_dir or required files don't exist.

    """
    data_path = pathlib.Path(data_dir).expanduser() / split

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    metadata_path = data_path / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    with metadata_path.open() as f:
        metadata = json.load(f)

    required_keys = {"seq_len", "vocab_size"}
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(f"dataset.json missing required fields: {missing_keys}")

    inputs_path = data_path / "all__inputs.npy"
    labels_path = data_path / "all__labels.npy"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found: {inputs_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    inputs = torch.from_numpy(np.load(inputs_path)).to(torch.int32)
    labels = torch.from_numpy(np.load(labels_path)).to(torch.int32)

    # Validate shapes
    if inputs.shape != labels.shape:
        raise ValueError(
            f"inputs and labels shape mismatch: {inputs.shape} vs {labels.shape}",
        )
    expected_seq_len = metadata["seq_len"]
    if inputs.shape[1] != expected_seq_len:
        raise ValueError(
            f"inputs seq_len mismatch: expected {expected_seq_len}, got {inputs.shape[1]}",
        )

    puzzle_ids_path = data_path / "all__puzzle_identifiers.npy"
    if puzzle_ids_path.exists():
        puzzle_identifiers = torch.from_numpy(np.load(puzzle_ids_path)).to(torch.int32)
        if puzzle_identifiers.ndim != 1:
            raise ValueError(
                f"puzzle_identifiers must be 1D, got shape {puzzle_identifiers.shape}",
            )
    else:
        puzzle_identifiers = None

    return {
        "inputs": inputs,
        "labels": labels,
        "vocab_size": metadata["vocab_size"],
        "seq_len": metadata["seq_len"],
        "puzzle_identifiers": puzzle_identifiers,
    }


class PuzzleDatasetIterator:
    """Puzzle dataset iterator with optional GPU caching and stratified sampling.

    Supports three dataset types:
    - Sudoku: single-example puzzles with augmentations
    - Maze: single-example puzzles with augmentations
    - ARC: multi-example puzzles (multiple input-output pairs per task)

    Two sampling strategies are available (only differ for multi-example datasets):
    - "pack": TRM/HRM-style. Packs multiple samples from same puzzle_id into batch.
      For ARC, a batch may contain several samples from one puzzle_id.
    - "single": One sample per puzzle_id_group.

    For Sudoku/Maze (single-example datasets), both modes are equivalent and
    produce identical results with the same RNG seed.

    Args:
      data_dir: Path to dataset directory.
      device: Torch device for output tensors.
      batch_size: Batch size. Must be positive.
      train: If True, load train split; else test split.
      shuffle: If True, shuffle indices each epoch.
      stratified: If True, use stratified sampling by puzzle_id_groups. Requires
        all__group_indices.npy file.
      gpu_cache: If True, cache full dataset on GPU. If False, keep on CPU
        and transfer batches on demand.
      sampling: "pack" for TRM/HRM-style (default), "single" for one sample
        per puzzle_id_group. Only differs for multi-example datasets like ARC.

    Yields:
      inputs: Batch of inputs [batch_size, seq_len].
      labels: Batch of labels [batch_size, seq_len].
      puzzle_ids: Batch of puzzle identifiers [batch_size]. All zeros if no
        puzzle_identifiers were available in the dataset.
      valid_count: Number of valid samples (last batch may be zero-padded).

    Raises:
      FileNotFoundError: If required files are missing (e.g., stratified=True but
        all__group_indices.npy doesn't exist).
      ValueError: If batch_size <= 0.

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
        sampling: Literal["pack", "single"] = "pack",
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        split = "train" if train else "test"
        data = load_puzzle_dataset(data_dir, split)

        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stratified = stratified and train
        self.gpu_cache = gpu_cache
        self.sampling = sampling
        self.vocab_size = data["vocab_size"]
        self.seq_len = data["seq_len"]

        # Storage device: GPU if caching, else CPU (transfer batches on demand)
        self._storage_device = device if gpu_cache else torch.device("cpu")

        self.inputs = data["inputs"].to(self._storage_device)
        self.labels = data["labels"].to(self._storage_device)
        self.puzzle_identifiers = (
            None
            if data["puzzle_identifiers"] is None
            else data["puzzle_identifiers"].to(self._storage_device)
        )
        self.n = len(self.inputs)

        # Load puzzle_indices for multi-example detection and puzzle_id lookup
        data_path = pathlib.Path(data_dir).expanduser() / split
        self._load_puzzle_indices(data_path)
        self._load_group_indices(data_path)

    def _load_puzzle_indices(self, data_path: pathlib.Path) -> None:
        """Load puzzle_indices and build example->puzzle mapping if needed."""
        puzzle_indices_path = data_path / "all__puzzle_indices.npy"
        if not puzzle_indices_path.exists():
            self._multi_example = False
            self._puzzle_indices = None
            self._example_to_puzzle = None
            return

        puzzle_indices = torch.from_numpy(np.load(puzzle_indices_path))
        if puzzle_indices.ndim != 1 or len(puzzle_indices) < 2:
            raise ValueError(
                f"puzzle_indices must be 1D with at least 2 elements, "
                f"got shape {puzzle_indices.shape}",
            )
        if not torch.all(puzzle_indices[1:] >= puzzle_indices[:-1]):
            raise ValueError("puzzle_indices must be monotonically increasing")

        # Detect multi-example datasets by checking if any puzzle has >1 example.
        # Single-example: puzzle_indices = [0, 1, 2, ..., n] (consecutive)
        # Multi-example: puzzle_indices has gaps, e.g., [0, 4, 8, ...] for 4 examples each
        # Note: compare BEFORE device transfer to avoid unnecessary GPU round-trip
        n_puzzles = len(puzzle_indices) - 1
        expected_single_example = torch.arange(n_puzzles + 1)
        self._multi_example = not torch.equal(puzzle_indices, expected_single_example)

        # Validate that puzzle_indices covers all examples.
        # Note: int(tensor) works for scalar tensors; no .item() needed.
        expected_n_examples = int(puzzle_indices[-1])
        if expected_n_examples != self.n:
            raise ValueError(
                f"puzzle_indices implies {expected_n_examples} examples, "
                f"but dataset has {self.n} examples",
            )

        # Now transfer to storage device
        self._puzzle_indices = puzzle_indices.to(self._storage_device)

        if self._multi_example:
            # Build example->puzzle mapping for correct puzzle_identifiers lookup.
            # For example index e, find puzzle p where puzzle_indices[p] <= e < puzzle_indices[p+1].
            # searchsorted with side="right" returns index where e would be inserted,
            # so we subtract 1 to get the puzzle index.
            example_indices = torch.arange(self.n, device=self._storage_device)
            self._example_to_puzzle = (
                torch.searchsorted(self._puzzle_indices, example_indices, side="right")
                - 1
            )
        else:
            self._example_to_puzzle = None

    def _load_group_indices(self, data_path: pathlib.Path) -> None:
        """Load group indices for stratified sampling."""
        if not self.stratified:
            self._puzzle_id_group_indices = None
            self._n_puzzle_id_groups = 0
            self._puzzle_ids_per_group = 0
            return

        group_indices_path = data_path / "all__group_indices.npy"
        if not group_indices_path.exists():
            raise FileNotFoundError(
                f"Stratified sampling requires {group_indices_path}, but file not found. "
                f"Set stratified=False or provide the file.",
            )

        group_indices = torch.from_numpy(np.load(group_indices_path))
        if group_indices.ndim != 1 or len(group_indices) < 2:
            raise ValueError(
                f"group_indices must be 1D with at least 2 elements, "
                f"got shape {group_indices.shape}",
            )
        if not torch.all(group_indices[1:] >= group_indices[:-1]):
            raise ValueError("group_indices must be monotonically increasing")
        self._n_puzzle_id_groups = len(group_indices) - 1

        # For single-example datasets, all groups have same size (precompute once).
        # Compute BEFORE device transfer to avoid GPU sync from .item() call.
        if not self._multi_example:
            self._puzzle_ids_per_group = int(group_indices[1] - group_indices[0])
        else:
            self._puzzle_ids_per_group = 0  # Variable sizes, computed per-group

        self._puzzle_id_group_indices = group_indices.to(self._storage_device)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        if self.sampling == "pack" and self.stratified:
            yield from self._iter_pack()
        else:
            yield from self._iter_single()

    def _iter_single(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        """Iterate yielding one sample per puzzle_id_group (or per example if not stratified)."""
        if self.stratified:
            indices, puzzle_ids_indices = self._sample_stratified_single()
            n_samples = self._n_puzzle_id_groups
        else:
            indices = torch.arange(self.n, device=self._storage_device)
            if self._multi_example and self._example_to_puzzle is not None:
                puzzle_ids_indices = self._example_to_puzzle
            else:
                puzzle_ids_indices = indices
            n_samples = self.n

        if self.shuffle:
            perm = torch.randperm(n_samples, device=self._storage_device)
            indices = indices[perm]
            puzzle_ids_indices = puzzle_ids_indices[perm]

        yield from self._yield_batches(indices, puzzle_ids_indices, n_samples)

    def _iter_pack(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        """TRM/HRM-style iteration: pack multiple samples from same puzzle_id.

        For single-example datasets (Sudoku/Maze), maintains backward compatibility
        with the original GPUCachedSudoku implementation by using the same RNG
        consumption order: (1) randint to sample all puzzle_ids at once, then
        (2) randperm to shuffle. This ensures bit-for-bit reproducibility with
        the same random seed.

        For multi-example datasets (ARC), uses TRM/HRM RNG order: (1) randperm to
        shuffle puzzle_id_groups first, then (2) rand to sample per puzzle_id_group.
        This different order is necessary because ARC has variable-size groups.
        """
        assert self._puzzle_id_group_indices is not None
        assert self._puzzle_indices is not None

        if not self._multi_example:
            yield from self._iter_pack_single_example()
        else:
            yield from self._iter_pack_multi_example()

    def _iter_pack_single_example(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        """Pack iteration for single-example datasets (Sudoku/Maze).

        Uses backward-compatible RNG order: (1) randint to sample all puzzle_ids
        at once, then (2) randperm to shuffle. This matches the original
        GPUCachedSudoku implementation for bit-for-bit reproducibility.
        """
        assert self._puzzle_id_group_indices is not None
        device = self._storage_device
        n_groups = self._n_puzzle_id_groups
        group_starts = self._puzzle_id_group_indices[:-1]

        # RNG call 1: sample one puzzle_id per group (vectorized, no .item() calls)
        puzzle_id_offsets = torch.randint(
            0,
            self._puzzle_ids_per_group,
            (n_groups,),
            device=device,
        )
        indices = group_starts + puzzle_id_offsets

        # RNG call 2: shuffle
        if self.shuffle:
            perm = torch.randperm(n_groups, device=device)
            indices = indices[perm]

        # Single-example: puzzle_id == sample index
        yield from self._yield_batches(indices, indices, n_groups)

    def _iter_pack_multi_example(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        """Pack iteration for multi-example datasets (ARC).

        Fully vectorized: samples one puzzle_id per group, expands all samples
        from each puzzle_id, then yields batches. Samples from the same puzzle_id
        are kept contiguous (packed) in the iteration order.
        """
        assert self._puzzle_id_group_indices is not None
        assert self._puzzle_indices is not None

        device = self._storage_device
        n_groups = self._n_puzzle_id_groups
        group_indices = self._puzzle_id_group_indices
        puzzle_indices = self._puzzle_indices

        # Step 1: Shuffle group order
        if self.shuffle:
            group_order = torch.randperm(n_groups, device=device)
        else:
            group_order = torch.arange(n_groups, device=device)

        # Step 2: Sample one puzzle_id per group (vectorized)
        group_starts = group_indices[group_order]
        group_sizes = group_indices[group_order + 1] - group_starts
        if (group_sizes <= 0).any():
            raise ValueError("All groups must have at least one puzzle_id")
        # Floating point safety: torch.rand() produces values in [0, 1), so
        # rand() * N produces values in [0, N). After .long() truncation,
        # results are in [0, N-1] - exactly what we need for valid indices.
        puzzle_id_offsets = (torch.rand(n_groups, device=device) * group_sizes).long()
        sampled_puzzle_ids = group_starts + puzzle_id_offsets

        # Step 3: Get sample counts for each puzzle_id
        sample_starts = puzzle_indices[sampled_puzzle_ids]
        sample_counts = puzzle_indices[sampled_puzzle_ids + 1] - sample_starts

        # Step 4: Expand puzzle_ids for each sample using repeat_interleave
        expanded_puzzle_ids = torch.repeat_interleave(sampled_puzzle_ids, sample_counts)
        expanded_starts = torch.repeat_interleave(sample_starts, sample_counts)

        # Step 5: Generate within-puzzle offsets using cumsum trick
        # Note: one int() conversion needed for torch.arange and batch iteration
        total_samples = int(sample_counts.sum())
        global_offsets = torch.arange(total_samples, device=device)
        cumulative_counts = torch.cat(
            [
                torch.zeros(1, device=device, dtype=sample_counts.dtype),
                sample_counts.cumsum(0)[:-1],
            ],
        )
        puzzle_start_offsets = torch.repeat_interleave(cumulative_counts, sample_counts)
        sample_indices = expanded_starts + (global_offsets - puzzle_start_offsets)

        # Step 6: Yield batches
        yield from self._yield_batches(
            sample_indices,
            expanded_puzzle_ids,
            total_samples,
        )

    def _yield_batches(
        self,
        indices: Tensor,
        puzzle_ids_indices: Tensor,
        n_samples: int,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        """Yield batches from pre-computed indices."""
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_puzzle_indices = puzzle_ids_indices[i : i + self.batch_size]
            valid_count = len(batch_indices)
            yield self._make_batch(batch_indices, batch_puzzle_indices, valid_count)

    def _make_batch(
        self,
        batch_indices: Tensor,
        batch_puzzle_indices: Tensor,
        valid_count: int,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Fetch data, transfer to device if needed, and pad last batch."""
        inputs = self.inputs[batch_indices]
        labels = self.labels[batch_indices]
        # Create zero puzzle_ids on demand if puzzle_identifiers not available
        puzzle_ids = (
            torch.zeros(valid_count, device=self._storage_device, dtype=torch.int32)
            if self.puzzle_identifiers is None
            else self.puzzle_identifiers[batch_puzzle_indices]
        )

        # Transfer to output device if not already cached there
        if not self.gpu_cache:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            puzzle_ids = puzzle_ids.to(self.device)

        # Pad last batch to batch_size if needed
        if valid_count < self.batch_size:
            pad_size = self.batch_size - valid_count
            inputs = torch.cat([inputs, inputs.new_zeros(pad_size, inputs.shape[1])])
            labels = torch.cat([labels, labels.new_zeros(pad_size, labels.shape[1])])
            puzzle_ids = torch.cat([puzzle_ids, puzzle_ids.new_zeros(pad_size)])

        return inputs, labels, puzzle_ids, valid_count

    def _sample_stratified_single(self) -> tuple[Tensor, Tensor]:
        """Sample one example per puzzle_id_group for stratified iteration.

        Returns:
          Tuple of (sample_indices, puzzle_indices) where:
          - sample_indices: Indices into inputs/labels arrays [n_groups].
          - puzzle_indices: Indices into puzzle_identifiers array [n_groups].
            For single-example datasets, puzzle_indices == sample_indices.

        """
        assert self._puzzle_id_group_indices is not None
        n_groups = self._n_puzzle_id_groups
        device = self._storage_device
        group_starts = self._puzzle_id_group_indices[:-1]

        if self._multi_example:
            # ARC: two-level sampling (group -> puzzle_id -> sample)
            # Groups and puzzle_ids may have variable sizes.
            assert self._puzzle_indices is not None

            group_sizes = self._puzzle_id_group_indices[1:] - group_starts
            if (group_sizes <= 0).any():
                raise ValueError("All groups must have at least one puzzle_id")
            puzzle_id_offsets = (
                torch.rand(n_groups, device=device) * group_sizes
            ).long()
            sampled_puzzle_ids = group_starts + puzzle_id_offsets

            sample_starts = self._puzzle_indices[sampled_puzzle_ids]
            sample_sizes = self._puzzle_indices[sampled_puzzle_ids + 1] - sample_starts
            sample_offsets = (torch.rand(n_groups, device=device) * sample_sizes).long()
            sample_indices = sample_starts + sample_offsets

            return sample_indices, sampled_puzzle_ids

        # Sudoku/Maze: single-level sampling (group -> sample)
        # All groups have same size, puzzle_id == sample index.
        puzzle_id_offsets = torch.randint(
            0,
            self._puzzle_ids_per_group,
            (n_groups,),
            device=device,
        )
        sample_indices = group_starts + puzzle_id_offsets
        return sample_indices, sample_indices
