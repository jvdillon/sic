"""x08: x07 + attn_qk_norm=True, muon_lr=0.01."""

from experiment import main, setup_muon_optimizers
from maze.x07 import Experiment as Experiment07


class Experiment(Experiment07):
    def setup_optimizers(self) -> None:
        self.optimizer1, self.optimizer2 = setup_muon_optimizers(  # pyright: ignore[reportAttributeAccessIssue]
            self.model,
            muon_lr=0.01,
        )


if __name__ == "__main__":
    main(Experiment())
