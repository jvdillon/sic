import numpy as np
import pydantic


# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: list[str]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:  # noqa: PLR0911
    """8 dihedral symmetries by rotate, flip and mirror."""
    if tid == 0:
        return arr  # identity
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)  # horizontal flip
    if tid == 5:
        return np.flipud(arr)  # vertical flip
    if tid == 6:
        return arr.T  # transpose (reflection along main diagonal)
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])
