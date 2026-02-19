"""x000c: x000 + lr_warmup_steps=1000, lr_min_ratio=0.10."""

from experiment import main, setup_muon_optimizers
from maze.old.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    lr_warmup_steps: int = 1000
    lr_min_ratio: float = 0.10

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.005,
        )


if __name__ == "__main__":
    main(Experiment())
