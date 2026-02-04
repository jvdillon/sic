# Speed is Confidence

This repository contains the "Speed is Confidence" research paper and its associated experiments.

- [Paper (arxiv)](https://arxiv.org/abs/2601.19085)
- [Paper (latest; Raw PDF)](https://github.com/jvdillon/sic/raw/main/paper/arxiv.pdf)
- [Paper (latest; Google Docs viewer)](https://docs.google.com/viewer?url=https://github.com/jvdillon/sic/raw/main/paper/arxiv.pdf?v=0)

## Abstract

Biological neural systems must be fast but are energy-constrained. Evolution's solution: act on the first signal. Winner-take-all circuits and time-to-first-spike coding implicitly treat when a neuron fires as an expression of confidence.

We apply this principle to Tiny Recursive Models (TRM) [Jolicoeur-Martineau et al., 2025]. On Sudoku-Extreme, a baseline TRM achieves 85.5% +/- 1.3%. But a key diagnostic reveals untapped potential: 89% of failures are *selection* problems--the model can solve these puzzles with a different random initialization. The true capability ceiling is 99%, not 86%.

Halt-first ensembling unlocks this potential: by selecting the first model to halt, we achieve 97% accuracy vs. 91% for probability averaging--while requiring 10x fewer reasoning steps. But can we internalize this as a training-only cost? Yes: by maintaining K=4 parallel latent states and backpropping only through the lowest-loss "winner," we achieve 96.9% +/- 0.6% accuracy--matching ensemble performance at 1x inference cost.

As in nature, this work was also resource constrained: all experiments used a single RTX 5090. A modified SwiGLU [Shazeer, 2020] made Muon [Jordan et al., 2024] and high LR viable, enabling baseline training in 48 minutes and full WTA (K=4) in 6 hours--compared to TRM's 20 hours on an L40S.

## Installation

```bash
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
CUDA_VISIBLE_DEVICES=0 uv run python -m code.sudoku.x182  # Train K=4, for example.
```

Much of my code builds on the work of [Alexia
Jolicoeur-Martineau](https://github.com/AlexiaJM) in the
[TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
repo which in turn built on the work of the [Sapient Inc](https://sapient.inc/)
team in the [HRM](https://github.com/sapientinc/HRM) repo.

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

and the Tiny Recursive Models (TRM) paper:

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

and the Hierarchical Reasoning Model (HRM) paper:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}
```

## License

Apache-2.0
