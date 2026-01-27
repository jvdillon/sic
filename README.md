# Speed is Confidence

This repository contains the code for the "Speed is Confidence" research paper.

## Abstract

Biological neural systems must be fast but are energy-constrained. Evolution's solution: act on the first signal. Winner-take-all circuits and time-to-first-spike coding implicitly treat when a neuron fires as an expression of confidence.

We apply this principle to ensembles of Tiny Recursive Models (TRM) [Jolicoeur-Martineau et al., 2025]. By basing the ensemble prediction solely on the first to halt rather than averaging predictions, we achieve 97.2% puzzle accuracy on Sudoku-Extreme while using 10x less compute than test-time augmentation (the baseline achieves 87% and 97.3% with TTA). Inference speed is an implicit indication of confidence.

But can this capability be manifested as a training-only cost? Evidently yes: by maintaining K=4 parallel latent states during training but backpropping only through the lowest-loss "winner," a single model achieves 96.9% +/- 0.6% puzzle accuracy—roughly matching TTA performance without any test-time augmentation.

As in nature, this work was also resource constrained: all experimentation used a single RTX 5090. This necessitated efficiency and compelled our invention of a modified SwiGLU [Shazeer, 2020] which made Muon [Jordan et al., 2024] viable. With these improvements and K=1 training, we match TRM baseline performance (∼87%) in just 48 min (8k steps, bs=384, ∼16GiB). Higher accuracy (∼97%) is achieved in 36k steps and K=4 (bs=192, ∼30GiB) and takes about 6 hours.

## Installation

```bash
uv sync
```

## License

Apache-2.0
