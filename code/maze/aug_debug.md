# Aug Data Unlearnability Debug

## Problem

Augmented maze data (8 dihedral transforms per instance) is
much harder to learn than non-augmented data.

| Exp  | Data  | RoPE | Schedule   | cell_acc | puzzle_acc |
|------|-------|------|------------|----------|------------|
| x07  | noaug | 1D   | 16 steps   | 99%+     | (halting)  |
| x07b | aug   | 1D   | 16 steps   | 93-97%   | 0-32%*     |
| x07c | aug   | 1D   | 16 steps   | 94-97%   | 0-17%*     |
| x07e | noaug | 2D   | curriculum | 99.4%    | 81.6%      |
| x07f | noaug | 2D   | 16 steps   | 99.3%    | 76.2%      |
| x07g | aug   | 2D   | curriculum | 95.9%    | 0.0%       |
| x07h | aug   | 2D   | 16 steps   | 96.1%    | 1.9%       |

*x07b/c puzzle_acc is highly unstable, oscillating wildly.

Key observations:
- Aug is hard regardless of RoPE (x07b/c also struggle)
- Aug + 2D RoPE is worst (x07g/h stuck at 0% puzzle)
- Noaug + 2D RoPE is best (x07e/f reach 76-81% puzzle)
- Aug symptoms: head convergence (H_cos ~0.96 vs ~0.67),
  high broken cells (~4800 vs ~170), q_halt doesn't learn

## Hypotheses

### H1: 2D RoPE incompatible with dihedral augmentation

**Claim:** After `np.rot90` on a 30x30 grid and re-flatten,
2D RoPE assigns wrong positions because it computes
`(row, col) = (1 + i//W, 1 + i%W)` from sequence index,
which wouldn't match the rotated cell's original position.

**Verdict:** REJECTED.

**Evidence:** RoPE encodes *relative* positions. Dihedral
transforms (rotations, reflections) are isometries that
preserve adjacency. After rotating and re-flattening, cell
`[i]` in the new flat sequence is at grid position
`(i//30, i%30)` of the rotated grid. Two adjacent cells in
the rotated grid are still at relative distance (0,1) or
(1,0). The 2D RoPE correctly captures this. The rotated maze
is a valid maze with correct spatial structure. Verified that
`np.rot90(noaug[0].reshape(30,30)).reshape(-1) == aug[1]`
and same for labels.

### H2: Augmented data is corrupted / invalid

**Claim:** Dihedral transforms might produce invalid mazes
(broken solution paths, wrong labels).

**Verdict:** REJECTED.

**Evidence:** Verified programmatically:
- `np.rot90(noaug[0].reshape(30,30)).reshape(-1) == aug[1]`
  (and same for labels) — transforms applied correctly
- Solution path in rotated maze is connected (BFS check:
  113/113 cells reachable)
- Start (3) and end (4) markers move to correct positions
- Input diff positions: wall (2) → solution (5) in labels,
  same count (111) across augmentations
- Token value distributions preserved

### H3: Aug is simply a harder task (not a bug)

**Claim:** With 8 orientations per maze, the model must learn
orientation-invariant reasoning rather than memorizing
position-specific patterns. This requires more capacity or
training time.

**Verdict:** PLAUSIBLE but doesn't explain the severity.

**Evidence:**
- x07b/c (aug, 1D RoPE) do eventually reach 17-32% puzzle_acc
  but are very unstable
- x07g/h (aug, 2D RoPE) are stuck near 0% puzzle_acc
- Both use same model (6.4M params), same 1000 base instances
- Stratified sampling ensures 1000 unique instances/epoch
  regardless of aug
- The gap between noaug (99%+) and aug (~95%) cell_acc
  suggests something beyond just "harder task"
- Open question: why does 2D RoPE make aug *worse*?
  (x07b reaches 32% puzzle with 1D RoPE; x07g stuck at 0%
  with 2D RoPE)

### H4: 2D RoPE + aug creates absolute position conflicts

**Claim:** 2D RoPE encodes absolute (row, col) positions.
With aug, the same local maze pattern appears at different
absolute positions across orientations. The model learns
position-dependent features that conflict across
augmentations, preventing convergence. 1D RoPE is less
affected because its positional signal is weaker/less
spatially informative.

**Verdict:** UNDER INVESTIGATION.

**Evidence so far:**
- 2D RoPE helps noaug dramatically (x07f 76% vs x07 with
  1D RoPE)
- 2D RoPE hurts aug (x07g 0% vs x07b 32% with 1D RoPE)
- This is consistent with 2D RoPE providing strong absolute
  position signal that helps when positions are consistent
  (noaug) but hurts when positions vary (aug)
- Head convergence in aug (H_cos ~0.96) suggests the model
  is stuck in a symmetric solution that can't differentiate
  positions — possibly because conflicting position signals
  cause the heads to average out

**What would confirm/reject:**
- Run aug with NO RoPE at all — if puzzle_acc improves vs
  x07g, confirms RoPE absolute position is harmful with aug
- Run aug with 2D RoPE but with relative-only attention
  (if architecture supports it)

### H5: Bias-free model can't adapt to augmentation

**Claim:** RoPE is a fixed positional bias — it injects
position info into attention but can't adapt. The core model
(QKV, O, MLP) is entirely `bias=False`. The only biases are
on output heads. With noaug, fixed RoPE is sufficient because
orientation never changes. With aug, the model needs a
*learnable* positional mechanism to handle varying
orientations, but has none — RoPE is fixed and there are no
bias terms to compensate.

The key observation: 1D RoPE → 2D RoPE changes aug behavior
dramatically (32% → 0% puzzle_acc). Both are fixed biases but
with different strengths. This suggests the model is sensitive
to the *type* of positional bias, and a learnable bias could
adapt where fixed ones fail.

**Verdict:** UNTESTED.

**What would confirm/reject:**
- Add learnable bias to attention (e.g., attention bias
  matrix, or bias=True in QKV/O projections) and run with
  aug + 2D RoPE
- If puzzle_acc improves, confirms the model needs learnable
  position-dependent parameters to handle augmentation
- Alternatively: learnable absolute position embeddings
  (added to token embeddings) alongside RoPE could provide
  the needed flexibility

### H6: Stratified sampling discards 7/8 of aug data

**Claim:** The old `stratified=True` sampling picked one
random augmentation per instance per epoch, meaning only
1000/8000 samples were seen per epoch. The model was
effectively undertrained on augmented data — it rarely saw
the same instance under different orientations within the
same training window, so it could attempt to memorize
token values rather than learn positional reasoning.

**Verdict:** FIX IMPLEMENTED, AWAITING RERUN.

**Evidence:**
- Code inspection confirmed: `stratified=True` selected
  one random aug index per instance, yielding 1000 samples
  per epoch from an 8000-sample dataset
- The model's training loss *increased* in x07h, consistent
  with memorization attempts being disrupted by the rare
  appearance of alternate orientations
- Noaug data (1000 samples, no stratification) trained
  normally — the only difference was the sampling strategy

**Fix:** Replaced `stratified: bool` with
`augmentation_random_bundle_max_size: int` (default=1).
All samples are now visited every epoch. The parameter
controls how augmentations of the same instance are
grouped in batches:
- k=1: fully shuffled (all 8000 samples, random order)
- k=8: all 8 augs of each instance are contiguous in the
  batch stream (8000 samples, grouped by instance)

**What would confirm/reject:**
- Rerun x07g/h with `augmentation_random_bundle_max_size=1`
  (all samples, fully shuffled). If puzzle_acc improves
  substantially, confirms stratified sampling was the
  primary cause.
- Compare k=1 vs k=8 to test whether contiguous
  same-instance augs help or hurt learning.
