This dir's code is modified from the
[TRM repo](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
which itself is modified from the
[HRM repo](https://github.com/sapientinc/HRM).

From the project root (`~/projects/sic`):

## Sudoku

- [TRM paper](https://arxiv.org/abs/2510.04871): "1000 shuffling augmentations per data example."
- [HRM paper](https://arxiv.org/abs/2506.21734): "We augment Sudoku puzzles by applying band and digit permutations."
- [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels#dataset-preparation): `python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`
- [HRM README](https://github.com/sapientinc/HRM#dataset-preparation): `python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`

```bash
uv run python -m code.dataset.build_sudoku_dataset \
  --output-dir /opt/scratch/datasets/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

## Maze

- [TRM paper](https://arxiv.org/abs/2510.04871): "Maze-Hard uses 8 dihedral transformations per data example."
- [HRM paper](https://arxiv.org/abs/2506.21734): "Data augmentation is disabled for Maze tasks."
- [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels#dataset-preparation): `python dataset/build_maze_dataset.py # 1000 examples, 8 augments` (but omits `--aug`; code defaults to `aug=False`)
- [HRM README](https://github.com/sapientinc/HRM#dataset-preparation): `python dataset/build_maze_dataset.py # 1000 examples`

To match TRM paper (with augmentation):

```bash
uv run python -m code.dataset.build_maze_dataset \
  --output-dir /opt/scratch/datasets/maze-30x30-hard-1k-aug \
  --aug
```

To match HRM paper (no augmentation):

```bash
uv run python -m code.dataset.build_maze_dataset \
  --output-dir /opt/scratch/datasets/maze-30x30-hard-1k
```

## ARC

- [TRM paper](https://arxiv.org/abs/2510.04871): "ARC-AGI uses 1000 data augmentations (color permutation, dihedral-group, and translations transformations) per data example."
- [HRM paper](https://arxiv.org/abs/2506.21734): "applying translations, rotations, flips, and color permutations to the puzzles" with "1000 augmented variants".
- [TRM README](https://github.com/SamsungSAILMontreal/TinyRecursiveModels#dataset-preparation): uses `--input-file-prefix` with combined Kaggle-style JSON files.
- [HRM README](https://github.com/sapientinc/HRM#dataset-preparation): uses `--dataset-dirs` with original ARC directory structure.

The TRM/HRM command difference is only cosmetic: both produce identical output format (same
numpy arrays, metadata, and encoding). They differ only in how input data is read.

Our code follows the TRM interface, which expects combined Kaggle-style JSON files.
These are available in the TRM repo:

```bash
# Download ARC data
mkdir -p /opt/scratch/datasets/arc-agi
git clone --depth 1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git /tmp/trm
cp /tmp/trm/kaggle/combined/arc-agi_*.json /opt/scratch/datasets/arc-agi/
rm -rf /tmp/trm

# ARC-AGI-1
uv run python -m code.dataset.build_arc_dataset \
  --input-file-prefix /opt/scratch/datasets/arc-agi/arc-agi \
  --output-dir /opt/scratch/datasets/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
uv run python -m code.dataset.build_arc_dataset \
  --input-file-prefix /opt/scratch/datasets/arc-agi/arc-agi \
  --output-dir /opt/scratch/datasets/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2
training data contains some ARC-AGI-1 eval data.
