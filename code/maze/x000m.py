"""x000f: x000 with attn_muon_modified=True."""

from experiment import main, setup_muon_optimizers
from maze.x000l import Experiment as Experiment000l


class Experiment(Experiment000l):
    batch_size: int = 176

    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.010,
        )


if __name__ == "__main__":
    main(Experiment())
