"""x000a: x000 except larger lr."""

from experiment import main, setup_muon_optimizers
from maze.x000 import Experiment as Experiment000


class Experiment(Experiment000):
    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.010,
        )


if __name__ == "__main__":
    main(Experiment())
