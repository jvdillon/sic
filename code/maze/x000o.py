"""x000o: x000l with halved Muon LR (0.0025)."""

from experiment import main, setup_muon_optimizers
from maze.x000l import Experiment as Experiment000l


class Experiment(Experiment000l):
    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.0025,
        )


if __name__ == "__main__":
    main(Experiment())
