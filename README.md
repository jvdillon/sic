# Speed is Confidence

This repository contains the "Speed is Confidence" research paper and its associated experiments.

- [Paper (arxiv)](https://arxiv.org/abs/2601.19085)
- [Paper (latest; Raw PDF)](https://github.com/jvdillon/sic/raw/main/paper/arxiv.pdf)
- [Paper (latest; Google Docs viewer)](https://docs.google.com/viewer?url=https://github.com/jvdillon/sic/raw/main/paper/arxiv.pdf?v=0)

## Abstract

Biological neural systems must be fast but are energy-constrained. Evolution's solution: act on the first signal. Winner-take-all circuits and time-to-first-spike coding implicitly treat when a neuron fires as an expression of confidence.

We apply this principle to ensembles of Tiny Recursive Models (TRM) [Jolicoeur-Martineau et al., 2025]. On Sudoku-Extreme, halt-first selection achieves 97% accuracy vs. 91% for probability averaging--while requiring 10x fewer reasoning steps. A single baseline model achieves 85.5% +/- 1.3%.

Can we internalize this as a training-only cost? Yes: by maintaining K=4 parallel latent states but backpropping only through the lowest-loss "winner," we achieve 96.9% +/- 0.6% accuracy--matching ensemble performance at 1x inference cost, with less than half the variance of the baseline. A key diagnostic: 89% of baseline failures are selection problems, revealing a 99% accuracy ceiling.

As in nature, this work was also resource constrained: all experiments used a single RTX 5090. A modified SwiGLU [Shazeer, 2020] made Muon [Jordan et al., 2024] and high LR viable, enabling baseline training in 48 minutes and full WTA (K=4) in 6 hours--compared to TRM's 20 hours on an L40S.

## Installation

```bash
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run code/sudoku/x182.py  # Train K=4, for example.
```

Much the code here builds on the excellent work of [Alexia
Jolicoeur-Martineau](https://github.com/AlexiaJM) available from
[TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{dillon2026speedisconfidence,
      title={Speed is Confidence},
      author={Joshua V. Dillon},
      year={2026},
      eprint={2601.19085},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.19085},
}
```

and the Tiny Recursive Models paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871},
}
```

## License

Apache-2.0
