"""x07d: x07a + use_rope=False, activation checkpointing on H-cycles.

Outcome: Learning fails.
"""

from __future__ import annotations

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3AC, TRM3ConfigProtocol


class Experiment(Experiment07a):
    total_train_steps: int = 8_000
    config: TRM3ConfigProtocol = TRM3AC.Config().update(
        Experiment07a.config,
        use_rope=False,
    )


if __name__ == "__main__":
    main(Experiment())
