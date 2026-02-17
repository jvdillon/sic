"""x07e: x07 + max_steps_schedule (progressive H-steps)."""

from __future__ import annotations

from typing import ClassVar

from experiment import main
from maze.x07 import Experiment as Experiment07


class Experiment(Experiment07):
    max_steps_schedule: ClassVar[dict[int, tuple[int, bool]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        0: (1, False),
        1000: (2, False),
        2000: (4, False),
        3000: (6, True),
    }


#     config: TRM3ConfigProtocol = dataclasses.replace(
#         cast(TRM3.Config, Experiment07.config),
#         rope_kwargs={"base": (512, 512)},
#     )

#     def _run_h_cycles(
#         self,
#         core: Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]],
#         embeddings: Tensor,
#         z_H: Tensor,
#         z_L: Tensor,
#         cos_sin: tuple[Tensor, Tensor] | None,
#         cos_sin_detach: tuple[Tensor, Tensor] | None,
#     ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#         del cos_sin_detach
#         for _ in range(self.model.config.H_cycles - 1):
#             result = checkpoint(
#                 core,
#                 embeddings,
#                 z_H,
#                 z_L,
#                 cos_sin,
#                 use_reentrant=False,
#             )
#             assert result is not None
#             _logits, _q_halt, z_H, z_L = result
#         return core(embeddings, z_H, z_L, cos_sin)


if __name__ == "__main__":
    main(Experiment())
