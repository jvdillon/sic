"""x07b: x07a + activation checkpointing on H-cycles."""

from __future__ import annotations

from experiment import main
from maze.x07a import Experiment as Experiment07a
from model import TRM3AC, TRM3ConfigProtocol


class Experiment(Experiment07a):
    config: TRM3ConfigProtocol = TRM3AC.Config().update(Experiment07a.config)


if __name__ == "__main__":
    main(Experiment())
