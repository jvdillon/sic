from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from types import TracebackType
from typing import Protocol, Self, TextIO

import hashlib
import math
import operator
import os
import pathlib
import random
import sys
import traceback


if True:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="^Please use the new API settings to control TF32 behavior",
    )


from torch import Tensor
from typing_extensions import override

import numpy as np
import pytest
import torch


class SupportsStr(Protocol):
    def __str__(self) -> str: ...


class SupportsArgSort(Protocol):
    def __eq__(self, other: object) -> bool: ...
    def __getitem__(self, key: object) -> object: ...
    def __hash__(self) -> int: ...
    def __lt__(self, other: object) -> bool: ...


# Module-level numpy Generator for modern random API.
# Use set_seed() to initialize, then use numpy_rng for random operations.
numpy_rng: np.random.Generator = np.random.default_rng()


def salt(*args: SupportsStr) -> int:
    # Intentionally using hashlib.md5 for reproducibility and portability.
    s = hashlib.md5(str(args).encode())  # noqa: S324
    return int(s.hexdigest(), 16) & ((1 << 31) - 1)


def set_seed(seed: int) -> None:
    """Set random seed locally (non-distributed)."""
    # os.environ["PYTHONHASHSEED"] = str(salt("PYTHONHASHSEED", i, seed))
    random.seed(salt("python", seed))
    # Reset the Generator's bit_generator state with the new seed
    numpy_rng.bit_generator.state = np.random.PCG64(salt("numpy", seed)).state
    torch.manual_seed(salt("torch", seed))
    if torch.cuda.is_available():
        # Initialize CUDA before seeding to ensure deterministic behavior
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        for i in range(torch.cuda.device_count()):
            torch.cuda.manual_seed(salt("cuda", i, seed))
        # Or torch.cuda.manual_seed_all(seed)


def enable_determinism(*, cudnn: bool = True, sdpa: bool = False) -> None:
    """Enable all determinism settings for reproducible results.

    https://docs.pytorch.org/docs/stable/notes/randomness.html

    Call once at program start, before any CUDA operations.

    To get bit-for-bit you may need to: rm -rf /tmp/torchinductor_${USER}/
    """
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


class Tee:
    """Write to both stdout/stderr and a file simultaneously."""

    stream: dict[str, TextIO]

    def __init__(
        self,
        log: str | pathlib.Path | TextIO,
        stream: tuple[TextIO, ...] = (sys.stdout, sys.stderr),
        flush_after_write: bool = True,
    ) -> None:
        log_file: TextIO = (
            pathlib.Path(log).open("w") if isinstance(log, (str, pathlib.Path)) else log  # noqa: SIM115
        )
        self.stream = {
            **{s.name.strip("<>"): s for s in stream},
            "log": log_file,
        }
        self.flush_after_write = flush_after_write

    def write(self, message: str) -> None:
        for s in self.stream.values():
            s.write(message)
        if self.flush_after_write:
            self.flush()

    def flush(self) -> None:
        for s in self.stream.values():
            s.flush()

    def __enter__(self) -> Self:
        log = self.stream["log"]
        for k, stream_obj in self.stream.items():
            if k == "log":
                continue
            tee_obj = Tee(log, (stream_obj,), self.flush_after_write)
            setattr(sys, k, tee_obj)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            self.stream["log"].write(
                "".join(
                    traceback.format_exception(
                        exc_type,
                        exc_val,
                        exc_tb,
                    ),
                ),
            )
        for k, v in self.stream.items():
            if k == "log":
                continue
            setattr(sys, k, v)
        self.stream["log"].close()
        self.stream = {}
        return False


class AutocastWrapper:
    """Wraps a model to apply autocast during forward pass.

    Transparently forwards all attribute access to the underlying model
    except for forward(), which applies autocast.
    """

    def __init__(self, model: object, device_type: str, dtype: torch.dtype) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_device_type", device_type)
        object.__setattr__(self, "_dtype", dtype)

    def __call__(self, x: Tensor) -> Tensor:
        with torch.autocast(
            device_type=object.__getattribute__(self, "_device_type"),
            dtype=object.__getattribute__(self, "_dtype"),
            cache_enabled=False,
        ):
            model = object.__getattribute__(self, "_model")
            return model(x)  # type: ignore[operator]

    @override
    def __getattribute__(self, name: str) -> object:
        if name in ("_model", "_device_type", "_dtype", "__call__", "__class__"):
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_model"), name)

    @override
    def __setattr__(self, name: str, value: object) -> None:
        if name in ("_model", "_device_type", "_dtype"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_model"), name, value)


def test_main(test_file: str) -> None:
    """Run pytest on a test file with standard flags.

    Usage:
        if __name__ == "__main__":
            from util import test_main
            test_main(__file__)

    Args:
        test_file: The test file path (usually __file__)

    """
    sys.exit(
        pytest.main(
            [
                test_file,  # The test file to run
                "-v",  # Verbose output
                "-s",  # Don't capture output (show print statements)
                "-W",  # Warning filter (overrides -Werror for specific warning)
                "ignore::pytest.PytestAssertRewriteWarning",  # Ignore assertion rewrite warnings (happens during direct execution)
                *sys.argv[
                    1:
                ],  # Pass through command-line arguments (e.g., -k for test filtering)
            ],
        ),
    )


def log1mexp(x: Tensor) -> Tensor:
    """log(1 - exp(x)) for x<0

    This function does NOT compute log(1 - exp(-|a|)) as it is often defined.
    The R version--see link below--is defined as,
        log1p(-exp(x)),   x < -log(2)
        log(-expm1(x)),   ow.
    where x = -|a| whereas this version simply assumes a<0.

    Note: log1mexp(0) = -inf.

    Reference:
    - https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    return torch.where(
        x < -math.log(2),
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def log_softmax(
    x: Tensor,
    *,
    dim: int | Sequence[int] = -1,
    nansafe: bool = True,
) -> Tensor:
    dim = _dims(dim)
    if not dim:
        raise ValueError("Must specify at least one dim.")
    lsm = _log_softmax_nansafe if nansafe else torch.log_softmax
    if len(dim) == 1:
        return lsm(x, dim=dim[0])
    dim = tuple({d % x.ndim for d in dim})  # dedupe + normalize
    dest = tuple(range(-len(dim), 0))
    x = torch.movedim(x, dim, dest)
    shape = x.shape
    x = x.flatten(-len(dim))
    x = lsm(x, dim=-1)
    x = x.unflatten(-1, shape[-len(dim) :])
    x = torch.movedim(x, dest, dim)
    return x


def softmax(
    x: Tensor,
    *,
    dim: int | Sequence[int] = -1,
    nansafe: bool = True,
) -> Tensor:
    return log_softmax(x, dim=dim, nansafe=nansafe).exp()


def _log_softmax_nansafe(x: Tensor, dim: int = -1) -> Tensor:
    kw = {"dim": dim, "keepdim": True}

    # Benchmarked alternatives for _log_softmax_nansafe (eager / compiled on CUDA):
    #
    # 1. nested (WINNER - 1.96ms / 700μs):
    #    x = torch.where(has_posinf, torch.where(is_posinf, 0, -inf), x)
    #
    # 2. unnested (2.49ms / 706μs):
    #    x = torch.where(is_posinf, 0., x)
    #    x = torch.where(has_posinf & ~is_posinf, -inf, x)
    #
    # 3. combined (2.40ms / 708μs):
    #    x = torch.where(has_posinf & ~is_posinf, -inf, x)
    #    x = torch.where(is_all_neginf | is_posinf, 0., x)
    #
    # 4. hybrid (2.52ms / 709μs):
    #    y = torch.where(is_posinf, 0., -inf)
    #    x = torch.where(has_posinf, y, x)
    #    x = torch.where(is_all_neginf | is_posinf, 0., x)
    #
    # 5. hybrid2 (2.26ms / 708μs):
    #    y = torch.where(is_posinf, 0., -inf)
    #    x = torch.where(has_posinf, y, x)
    #    x = torch.where(is_all_neginf, 0., x)  # no | is_posinf needed
    #
    # 6. nested_explicit (2.17ms / 707μs):
    #    zero = torch.tensor(0., device=x.device)
    #    neginf = torch.tensor(-inf, device=x.device)
    #    x = torch.where(has_posinf, torch.where(is_posinf, zero, neginf), x)
    #
    # Version              Eager Fwd    Eager Bwd    Comp Fwd     Comp Bwd
    # ----------------------------------------------------------------------
    # pure                 0.52         1.14         0.52         1.15
    # nested               2.08         3.43         0.69         1.38
    # unnested             2.55         4.35         0.70         1.42
    # combined             2.37         3.91         0.71         1.41
    # hybrid2              2.19         3.59         0.71         1.67
    # nested_explicit      2.26         3.60         0.71         1.42
    #
    # Key findings:
    # - Nested where(a, where(b, x, y), z) is fastest in eager mode
    # - Boolean broadcast ops (has_posinf & ~is_posinf) are expensive
    # - Intermediate tensor assignment (y = where(...)) slower than nested
    # - Python scalars faster than explicit torch.tensor() in where()
    # - torch.compile equalizes all versions (~700μs)

    # Use "double where trick" to avoid nan.
    # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf

    # Corner Case 1: any +inf.
    is_posinf = x.isposinf()
    has_posinf = is_posinf.any(**kw)
    x = torch.where(has_posinf, torch.where(is_posinf, 0, -math.inf), x)

    # Corner Case 2: all -inf.
    is_all_neginf = x.isneginf().all(**kw)
    x = torch.where(is_all_neginf, 0, x)
    x = torch.log_softmax(x, dim=dim)
    x = torch.where(is_all_neginf, -math.inf, x)

    return x


def logsumexp(
    x: Tensor,
    *,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
    nansafe: bool = True,
) -> Tensor:
    dim = dim if isinstance(dim, int) else tuple(dim)
    if not nansafe:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    kw: dict[str, int | tuple[int, ...] | bool] = {"dim": dim, "keepdim": True}

    # Notice that both corner cases are identical to log_softmax but we have additional
    # cleanup. (Evidently since `nabla_x lse = softmax(x)`.)

    # Corner Case 1: any +inf.
    is_posinf = x.isposinf()
    has_posinf = is_posinf.any(**kw)
    x = torch.where(has_posinf, torch.where(is_posinf, 0, -math.inf), x)

    # Corner Case 2: all -inf.
    is_all_neginf = x.isneginf().all(**kw)
    x = torch.where(is_all_neginf, 0, x)
    x = torch.logsumexp(x, dim=dim, keepdim=True)
    x = torch.where(is_all_neginf, -math.inf, x)

    # Cleanup
    x = torch.where(has_posinf, math.inf, x)
    if not keepdim:
        x = x.squeeze(dim)

    return x


def logmeanexp(
    x: Tensor,
    *,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
    nansafe: bool = True,
) -> Tensor:
    """Warning: Has NaN gradient when all elements are non-finite in the dim axis. This is
    because its gradient is softmax but softmax of [-inf, -inf, ..., -inf] is
    [0/0, 0/0, ..., 0/0].
    """
    dim = _dims(dim)
    logn = math.log(max(1, math.prod(x.shape[d] for d in dim)))
    return logsumexp(x, dim=dim, keepdim=keepdim, nansafe=nansafe) - logn


def entropy(
    logp: Tensor,
    *,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
) -> Tensor:
    """H(p) = -sum(p * log(p)) in log-domain for numerical stability."""
    # Note: entropy is nansafe=True by definition/convention hence lacks this argument.
    dim = dim if isinstance(dim, int) else tuple(dim)
    safe_logp = torch.where(torch.isneginf(logp), 0, logp)
    return -torch.sum(logp.exp() * safe_logp, dim=dim, keepdim=keepdim)


def jsd(
    logp: Tensor,
    *,
    ensemble_dim: int | Sequence[int] = 0,
    event_dim: int | Sequence[int] = -1,
) -> Tensor:
    """JSD over K distributions in log-domain.

    Args:
        logp: [K, N, num_classes] log probabilities
        ensemble_dim: dimension(s) over ensemble (K dimension)
        event_dim: dimension(s) over events (num_classes dimension)

    Returns:
        jsd: JSD values per cell

    """
    ensemble_dim = _dims(ensemble_dim)
    event_dim = _dims(event_dim)
    log_avg_p = logmeanexp(logp, dim=ensemble_dim, keepdim=True, nansafe=True)
    H_avg = entropy(log_avg_p, dim=event_dim, keepdim=True)
    H_each = entropy(logp, dim=event_dim, keepdim=True)
    avg_H = torch.mean(H_each, dim=ensemble_dim, keepdim=True)
    return (H_avg - avg_H).squeeze(dim=ensemble_dim + event_dim)


def _dims(dim: int | Integral | Sequence[int | Integral]) -> tuple[int, ...]:
    """Sanitize helper for `dims`-like or `shape`-like values."""
    if isinstance(dim, (int, Integral)):
        return (operator.index(dim),)
    return tuple(operator.index(d) for d in dim)


def argsort(x: Sequence[SupportsArgSort], descending: bool = False) -> list[int]:
    return sorted(range(len(x)), key=x.__getitem__, reverse=descending)
