"""x000m: x000l + batch_size=176, muon_lr=0.010."""

from experiment import main, setup_muon_optimizers
from maze.old.x000l import Experiment as Experiment000l


class Experiment(Experiment000l):
    batch_size: int = 176

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(
            self.model,
            muon_lr=0.010,
            # muon_wd=1e-4/0.01=0.01
        )


if __name__ == "__main__":
    main(Experiment())
