"""x002a: x002 with embedding gradient preserved through feedback.

Same as x002 but computes _pred_feedback outside torch.no_grad(),
so the final H-cycle's input_emb retains its grad path to
embed_tokens.weight.
"""

from __future__ import annotations

from experiment import main
from maze.x002 import Experiment as Experiment002


class Experiment(Experiment002):
    feedback_preserves_emb_grad: bool = True


if __name__ == "__main__":
    main(Experiment())
