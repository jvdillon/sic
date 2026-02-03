from code.dataset.common import PuzzleDatasetMetadata, dihedral_transform
from pathlib import Path
from typing import Any

import csv
import json
import math

from argdantic import ArgParser
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm import tqdm

import numpy as np


CHARSET = "# SGo"


cli = ArgParser()
rng = np.random.default_rng()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    output_dir: str = "data/maze-30x30-hard-1k"

    subsample_size: int | None = None
    aug: bool = False


def convert_subset(set_name: str, config: DataProcessConfig) -> None:
    # Read CSV
    all_chars: set[str] = set()
    grid_size: tuple[int, int] | None = None
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with Path(
        hf_hub_download(
            config.source_repo,
            f"{set_name}.csv",
            repo_type="dataset",
        ),
    ).open(newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for _source, q, a, _rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)

            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = rng.choice(
                total_samples,
                size=config.subsample_size,
                replace=False,
            )
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    results: dict[str, list[Any]] = {
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

    for inp, out in zip(tqdm(inputs), labels, strict=True):
        # Dihedral transformations for augmentation
        for aug_idx in range(8 if (set_name == "train" and config.aug) else 1):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # Char mappings
    assert len(all_chars - set(CHARSET)) == 0

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq: list[np.ndarray]) -> np.ndarray:
        return np.vstack([char2id[s.reshape(-1)] for s in seq])

    results_np = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    assert grid_size is not None

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),
        vocab_size=len(CHARSET) + 1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results_np["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results_np["group_indices"]) - 1,
        sets=["all"],
    )

    # Save metadata as JSON.
    save_dir = Path(config.output_dir) / set_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "dataset.json").open("w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results_np.items():
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
