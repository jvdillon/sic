"""x08a: x08 + muon_lr=0.02."""

from experiment import main, setup_muon_optimizers
from maze.x08 import Experiment as Experiment08


class Experiment(Experiment08):
    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.02,
        )


if __name__ == "__main__":
    main(Experiment())
