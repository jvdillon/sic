# Speed is Confidence

This repository contains the "Speed is Confidence" research paper and its associated experiments.

- [Paper (arxiv)](https://arxiv.org/abs/2601.19085)
- [Paper (latest)](https://raw.githubusercontent.com/jvdillon/sic/main/paper/arxiv.pdf)

## Abstract

Biological neural systems must be fast but are energy-constrained. Evolution's solution: act on the first signal. Winner-take-all circuits and time-to-first-spike coding implicitly treat when a neuron fires as an expression of confidence.

We apply this principle to ensembles of Tiny Recursive Models (TRM) [Jolicoeur-Martineau et al., 2025]. On Sudoku-Extreme, halt-first selection achieves 97% accuracy vs. 91% for probability averaging—while requiring 10× fewer reasoning steps (early halting). A single baseline model achieves 85.5% +/- 1.3%. Inference speed is an implicit indication of confidence.

But can this capability be manifested as a training-only cost? Evidently yes: by maintaining K=4 parallel latent states during training but backpropping only through the lowest-loss "winner," we achieve 96.9% +/- 0.6% puzzle accuracy--roughly matching ensemble performance but at exactly the same cost as a single model, with half the variance of the baseline. (Four independent trials spanned 96.16% to 97.64%.)

As in nature, this work was also resource constrained: all experimentation used a single RTX 5090. This necessity compelled a modified SwiGLU [Shazeer, 2020] which made Muon [Jordan et al., 2024] viable. With these improvements and K=1 training, we match TRM baseline performance (∼85.5%) in just 48 min (8k steps, batch size 384, ∼16GiB). Higher accuracy (∼96.9%) is achieved in 36k steps and K=4 (batch size 192, ∼30GiB) and takes about 6 hours.

## Installation

```bash
uv sync
uv run code/sudoku/x182.py  # Train K=4, for example.
```

Much the code here builds on the excellent work of [Alexia
Jolicoeur-Martineau](https://github.com/AlexiaJM) available from
[TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

## License

Apache-2.0
