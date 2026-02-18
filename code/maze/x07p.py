"""x07p: x07a + muon_lr=0.02."""

from experiment import main, setup_muon_optimizers
from maze.x07a import Experiment as Experiment07a


class Experiment(Experiment07a):
    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.02,
        )


if __name__ == "__main__":
    main(Experiment())
