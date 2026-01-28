# Speed is Confidence

This repository contains the "Speed is Confidence" research paper and its associated experiments.

- [Paper (arxiv)](https://arxiv.org/abs/2601.19085)
- [Paper (latest)](https://raw.githubusercontent.com/jvdillon/sic/main/paper/arxiv.pdf)

## Abstract


Biological neural systems must be fast but are
energy-constrained. Evolution’s solution: act on
the first signal. Winner-take-all circuits and time-
to-first-spike coding implicitly treat when a neu-
ron fires as an expression of confidence.



We apply this principle to ensembles of Tiny
Recursive Models (TRM) [Jolicoeur-Martineau
et al., 2025]. On Sudoku-Extreme, halt-first se-
lection achieves 97% accuracy vs. 91% for proba-
bility averaging—while requiring 10× fewer rea-
soning steps. A single baseline model achieves
85.5% ± 1.3%.


Can we internalize this as a training-only cost?
Yes: by maintaining K=4 parallel latent states
but backpropping only through the lowest-loss
“winner,” we achieve 96.9% ± 0.6% accuracy—
matching ensemble performance at 1× inference
cost, with less than half the variance of the base-
line. (Four trials spanned 96.2%–97.6%.) A key
diagnostic: 89% of baseline failures are selection
problems, revealing a 99% accuracy ceiling.


As in nature, this work was also resource con-
strained: all experiments used a single RTX
5090. A modified SwiGLU [Shazeer, 2020] made
Muon [Jordan et al., 2024] and high LR viable,
enabling baseline training in 48 minutes and full
WTA (K=4) in 6 hours on consumer hardware.


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
