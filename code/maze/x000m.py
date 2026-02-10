"""x000f: x000 with attn_muon_modified=True."""

from experiment import main
from maze.x000l import Experiment as Experiment000l


class Experiment(Experiment000l):
    batch_size: int = 176


if __name__ == "__main__":
    main(Experiment())
