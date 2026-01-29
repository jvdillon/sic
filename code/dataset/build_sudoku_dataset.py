from pathlib import Path

import csv
import json

from argdantic import ArgParser
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm import tqdm

import numpy as np

from .common import PuzzleDatasetMetadata


cli = ArgParser()
rng = np.random.default_rng()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: int | None = None
    min_difficulty: int | None = None
    num_aug: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = rng.random() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []

    with Path(
        hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"),
    ).open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for _source, q, a, rating in reader:
            if (config.min_difficulty is None) or (
                int(rating) >= config.min_difficulty
            ):
                assert len(q) == 81
                assert len(a) == 81

                inputs.append(
                    np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(
                        9, 9
                    )
                    - ord("0")
                )
                labels.append(
                    np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
                )

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = rng.choice(
                total_samples, size=config.subsample_size, replace=False
            )
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs), labels, strict=False):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy
    def _seq_to_numpy(seq: list[np.ndarray]) -> np.ndarray:
        arr = np.concatenate(seq).reshape(len(seq), -1)

        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    # Save metadata as JSON.
    save_dir = Path(config.output_dir) / set_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "dataset.json").open("w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(save_dir / f"all__{k}.npy", v)

    # Save IDs mapping (for visualization only)
    with (Path(config.output_dir) / "identifiers.json").open("w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def preprocess_data(config: DataProcessConfig) -> None:
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
