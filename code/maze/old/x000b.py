"""x000b: x000 + muon_lr=0.0025."""

from experiment import main, setup_muon_optimizers
from maze.old.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.0025,
        )


if __name__ == "__main__":
    main(Experiment())
