"""Model building blocks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, Protocol, Self, TypedDict

import dataclasses
import functools
import math
import traceback

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import torch


InitFn = Callable[[Tensor], Tensor]


#############################################
#               Protocols                   #
#############################################


class ModuleProtocol(Protocol):
    """Protocol for nn.Module-like objects."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def eval(self) -> Self: ...
    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> object: ...
    def named_parameters(self) -> Iterator[tuple[str, Tensor]]: ...
    def parameters(self) -> Iterator[Tensor]: ...
    def register_parameter(self, name: str, param: nn.Parameter | None) -> None: ...
    def state_dict(self) -> dict[str, Tensor]: ...
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self: ...
    def train(self, mode: bool = True) -> Self: ...
    def zero_grad(self, set_to_none: bool = False) -> None: ...


class TRM1ConfigProtocol(Protocol):
    """Config protocol for TRM1 models."""

    seq_len: int
    hidden_size: int
    vocab_size: int
    dtype: torch.dtype | None
    num_heads: int
    num_layers: int
    H_cycles: int
    L_cycles: int
    state_noise: float
    head_bias: bool
    act: bool
    act_q_head_bias_init: float
    block_fn: Callable[[int], nn.Module]
    use_rope: bool
    rope_kwargs: dict[str, Any]
    causal: bool
    no_grad_inner: bool
    head_init_weight_fn: InitFn
    compile_core: bool
    compile_reasoning: bool
    max_num_compile_core: int

    @property
    def n_iters(self) -> int: ...

    def setup(self, *args: Any, **kwargs: Any) -> nn.Module: ...


class TRM3ConfigProtocol(Protocol):
    """Minimal config protocol for TRM3 models (duck typing for callers)."""

    puzzle_grid_shape: tuple[int, ...]
    hidden_size: int
    vocab_size: int
    dtype: torch.dtype | None
    num_layers: int
    H_cycles: int
    L_cycles: int
    head_bias: bool
    block_fn: Callable[[int], nn.Module]
    block_kwargs_by_layer: dict[int, dict[str, Any]]
    compile_core: bool
    compile_reasoning: bool
    max_num_compile_core: int
    K_H: int
    K_L: int
    num_puzzle_id_tokens: int
    num_register_tokens: int
    num_puzzle_ids: int
    q_halt_seq_index: int
    use_rope: bool
    rope_kwargs: dict[str, Any]
    num_heads: int
    carry_H: Literal[
        "top1",
        "top2",
        "copy_top1",
        "copy_top2",
        "all",
        "none",
    ]
    carry_L: Literal[
        "top1",
        "top2",
        "copy_top1",
        "copy_top2",
        "all",
        "none",
    ]
    core_damping: float
    anchor_seq_index: int | None
    label_smoothing_includes_pad_token: bool

    @property
    def num_puzzle_grid_tokens(self) -> int: ...

    @property
    def total_seq_len(self) -> int: ...

    @property
    def num_effective_heads(self) -> int: ...

    def setup(self, *args: Any, **kwargs: Any) -> nn.Module: ...


class CarryState(TypedDict):
    """Model carry state for WTA forward.

    Each chain is an (z_H, z_L) pair. At runtime we have N = K_H * K_L chains.
    Carry policies (top1, top2, copy_top1, copy_top2) operate on chains.
    """

    z_H: Tensor  # [B, N, S, C] - each chain's z_H state
    z_L: Tensor  # [B, N, S, C] - each chain's z_L state
    carry_count: Tensor  # [B, N] - per-chain count of actual carries (updates)


class WTAForwardOutput(TypedDict):
    """Output of wta_forward."""

    logits: Tensor  # [B, N, num_puzzle_grid_tokens, V] per-head logits
    q_halt: Tensor  # [B, N] per-head halt logits
    losses: Tensor  # [B, N] per-head losses
    winner_idx: Tensor  # [B] index of winning head
    carry: CarryState  # Updated carry state
    h_indices: Tensor  # [N] H indices used
    l_indices: Tensor  # [N] L indices used


class TRM1Protocol(ModuleProtocol, Protocol):
    """Protocol for TRM1 models (used by ExperimentBase)."""

    config: TRM1ConfigProtocol
    H_init: Tensor  # nn.Buffer extends Tensor
    L_init: Tensor  # nn.Buffer extends Tensor

    def forward(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Tensor | list[Tensor]]: ...

    def step(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Tensor]: ...


class TRM3Protocol(ModuleProtocol, Protocol):
    """Protocol for TRM3 models with carry state and WTA."""

    config: TRM3ConfigProtocol
    H_init: Tensor
    L_init: Tensor

    def init_carry(self, batch_size: int) -> CarryState: ...

    def reset_carry_at_indices(
        self,
        carry: CarryState,
        indices: Tensor,
        n_reset: int,
    ) -> CarryState: ...

    def wta_forward(
        self,
        input_ids: Tensor,
        carry: CarryState,
        labels: Tensor,
        *,
        puzzle_ids: Tensor | None = None,
        label_smoothing: float = 0.0,
        z_L_noise: float = 0.0,
    ) -> WTAForwardOutput: ...


# Backwards compatibility aliases
ModelConfigProtocol = TRM3ConfigProtocol
ModelProtocol = TRM3Protocol


# Enable recompilation logging via environment variable:
# $ TORCH_LOGS="recompiles_verbose" python script.py
# or set os.environ["TORCH_LOGS"] = "recompiles_verbose" before importing torch

_compile_traces: dict[str, list[str]] = {}


@torch.compiler.assume_constant_result
def trace_compile(
    key: str,
    *,
    max_compiles: int = -1,
    always_print: bool = False,
) -> int:
    """Print trace when compiling. Safe to call from compiled code."""
    trace = "".join(traceback.format_stack()[:-1])
    traces = _compile_traces.setdefault(key, [])
    traces.append(trace)
    if always_print:
        print(trace)
    if max_compiles > -1 and len(traces) > max_compiles:
        traces_str = "" if always_print else ("\n" + "\n--------\n".join(traces))
        raise RuntimeError(f"Too many compiles ({len(traces)}) for {key}.{traces_str}")
    return len(traces)


def normal_init_(tensor: Tensor, std: float | None = None) -> Tensor:
    """Default init: normal."""
    if std is None:
        c_in = tensor.shape[-1]
        std = c_in ** (-0.5)
    elif std == 0:
        nn.init.zeros_(tensor)
        return tensor
    assert std is not None
    nn.init.normal_(tensor, std=std)
    return tensor


def trunc_normal_init_(
    tensor: Tensor,
    *,
    std: float | None = None,  # Aka LeCun init.
    lower: float = -2.0,
    upper: float = 2.0,
) -> Tensor:
    """Truncated normal initialization."""
    with torch.no_grad():
        if std is None:
            c_in = tensor.shape[-1]
            std = c_in ** (-0.5)
        elif std == 0:
            tensor.zero_()
            return tensor
        sqrt2 = 2**0.5
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * upper**2)
        pdf_l = c * math.exp(-0.5 * lower**2)
        comp_std = std * (
            1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
        ) ** (-0.5)
        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
        return tensor


def kaiming_uniform_init_(
    tensor: Tensor,
    *,
    c_in: int | None = None,
    bound: float | None = None,
) -> Tensor:
    if bound is None:
        if c_in is None:
            raise ValueError("kaiming_uniform_init_ requires bound or c_in")
        bound = c_in**-0.5
    assert bound is not None
    nn.init.uniform_(tensor, -bound, bound)
    return tensor


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleave: bool = False,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors.

    cos/sin are half-dim (D//2). Splits q/k into pairs, applies a 2D
    rotation, and recombines. No replication needed — half the memory
    vs full-dim cos/sin approaches.

    The ``interleave`` flag selects the dimension pairing convention:

    - ``False`` (default): GPT-NeoX / HuggingFace half-split. Pairs
      dim i with dim i+D/2. This is the convention in all HuggingFace
      Transformers models (LLaMA, Mistral, Gemma, Qwen, etc.). Default
      because most pretrained checkpoints use HF.
    - ``True``: RoFormer / Meta LLaMA interleave. Pairs dim 2i with
      dim 2i+1 (consecutive). This matches the original paper's math
      and Meta's official LLaMA code.

    Both conventions are mathematically equivalent up to a permutation
    of embedding dimensions. HuggingFace applies a weight permutation
    during checkpoint conversion (``convert_llama_weights_to_hf.py``)
    to reconcile the two. For bit-for-bit reproduction, match the
    convention used during training.

    When q/k have more sequence positions than cos/sin (e.g. from
    padding), cos/sin are right-padded with identity values (cos=1,
    sin=0) so extra positions are unrotated.

    References:
      - Su et al., RoFormer (arXiv:2104.09864), Eq. 34.
      - HuggingFace transformers ``rotate_half``:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
      - Meta LLaMA ``apply_rotary_emb``:
        https://github.com/meta-llama/llama/blob/main/llama/model.py

    Args:
      q: [..., S, H, D].
      k: [..., S, H, D].
      cos: [..., S', H, D//2] where H=1 (fixed) or num_heads
          (learnable). S' <= S; right-padded to S if smaller.
      sin: Same shape as cos.
      interleave: Pairing convention (see above).

    Returns:
      q_embed: Rotated queries, same shape as q.
      k_embed: Rotated keys, same shape as k.

    """
    # Pad cos/sin to match q/k sequence length if needed.
    seq_dim = -3
    seq_len = q.shape[seq_dim]
    rope_len = cos.shape[seq_dim]
    if rope_len < seq_len:
        pad_shape = list(cos.shape)
        pad_shape[seq_dim] = seq_len - rope_len
        cos = torch.cat(
            [cos, torch.ones(pad_shape, dtype=cos.dtype, device=cos.device)],
            dim=seq_dim,
        )
        sin = torch.cat(
            [sin, torch.zeros(pad_shape, dtype=sin.dtype, device=sin.device)],
            dim=seq_dim,
        )
    return (
        _rope_rotate(q, cos, sin, interleave),
        _rope_rotate(k, cos, sin, interleave),
    )


def _rope_rotate(x: Tensor, cos: Tensor, sin: Tensor, interleave: bool) -> Tensor:
    """Apply 2D rotation to a single tensor using half-dim cos/sin."""
    dtype = x.dtype
    shape = x.shape
    # float32 rotation math → input dtype.
    if interleave:
        split_dim = -1
        x = x.float().reshape(*shape[:-1], -1, 2)
    else:
        split_dim = -2
        x = x.float().reshape(*shape[:-1], 2, -1)
    x0, x1 = x.moveaxis(split_dim, 0)
    return (
        torch.stack(
            [x0 * cos - x1 * sin, x1 * cos + x0 * sin],
            dim=split_dim,
        )
        .reshape(shape)
        .to(dtype)
    )


class RoPE(nn.Module):
    """N-dimensional Rotary Position Embedding with fixed frequencies.

    Number of axes inferred from ``dim``: scalar for 1D, iterable for ND.
    Per-axis channel counts can be unequal (e.g. ``dim=[44, 42, 42]``
    allocates 44 frequencies to the first axis and 42 each to the other
    two). Each nonzero count must be even and >= 4; use 0 to skip an
    axis (e.g. ``dim=[128, 0]`` encodes only the first axis).

    Returns half-dim cos/sin (shape ``[..., S, 1, dim//2]``). Pair
    with ``apply_rope`` which splits q/k, applies the 2D rotation, and
    recombines — no replication needed. The ``interleave`` flag on
    ``apply_rope`` selects the dimension pairing convention:
    GPT-NeoX / HuggingFace half-split (LLaMA, Mistral, Gemma, Qwen)
    vs RoFormer / Meta LLaMA interleave. See ``apply_rope`` docstring
    for details.

    ``smallest_recommended_base(dim, max_positions)`` reference:

        len      c=16    c=32    c=64   c=128   c=256
        ----------------------------------------------------
           16       3       3       2       2       2
           32       6       5       5       5       5
           64      14      12      11      10      10
          128      31      25      22      21      21
          256      69      52      46      43      42
          512     152     109      94      87      84
        1,024     337     229     192     177     169
        2,048     745     479     393     357     341
        4,096   1,645   1,004     803     722     686
        8,192   3,632   2,103   1,643   1,461   1,379
       16,384   8,021   4,405   3,361   2,954   2,774
       32,768  17,713   9,227   6,873   5,974   5,579
       65,536  39,114  19,328  14,058  12,080  11,218
      131,072  86,372  40,484  28,751  24,428  22,560

    Args:
      dim: Channel count per axis. Scalar for 1D (e.g. 128), iterable
          for ND. Each nonzero value must be even and >= 4. Use 0 to
          skip an axis.
      base: Frequency base(s). Scalar (shared across axes) or iterable
          (one per axis). Controls the longest wavelength: the lowest
          frequency has period ``2*pi*base^((c-2)/c)`` where c is the
          per-axis channel count. Use ``smallest_recommended_base`` to
          compute from max position counts.

    """

    def __init__(
        self,
        dim: int | Iterable[int],
        *,
        base: float | Iterable[float] = 10e3,
        legacy: bool = False,
    ):
        super().__init__()
        dims, bases = self._broadcast_args(dim, base)
        self.dims = dims
        self._legacy = legacy
        self._active_axes = [i for i, c in enumerate(dims) if c > 0]
        if not self._active_axes:
            raise ValueError(f"At least one dim must be nonzero, got {dims}.")
        self._bases, self._dims = zip(
            *((b, c) for b, c in zip(bases, dims, strict=True) if c > 0),
            strict=False,
        )
        for c in self._dims:
            self._validate_c(c)
        # Block-diagonal [N_axes, 1, total_half_dim]: each axis's
        # freqs occupy separate channels (zeros elsewhere).
        total = sum(c // 2 for c in self._dims)
        rows = []
        offset = 0
        for b, c in zip(self._bases, self._dims, strict=False):
            row = torch.zeros(total)
            row[offset : offset + c // 2] = self._make_inv_freqs(b, c)
            offset += c // 2
            rows.append(row)
        self._inv_freqs = torch.stack(rows).unsqueeze(-2)
        self._dtype = nn.Buffer(torch.empty(0), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype.dtype

    @property
    def device(self) -> torch.device:
        return self._dtype.device

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Encode positions to half-dim (cos, sin) embeddings.

        Args:
          positions: [..., S, num_axes] or [..., S] when num_axes=1.

        Returns:
          cos: [..., S, H, dim // 2] where H=1 for fixed, num_heads
              for per-head variants.
          sin: Same shape as cos.

        """
        if positions.ndim == 1:
            positions = positions.unsqueeze(-1)

        if self._legacy:
            # Old RotaryEmbedding precomputed cos/sin on CPU at init.
            # CPU and GPU trig differ by 1 ULP; replicate CPU path.
            device = "cpu"
        else:
            device = self.device

        positions = positions.to(dtype=torch.float32, device=device)
        inv_freqs = self._inv_freqs.to(dtype=torch.float32, device=device)

        # [..., S, N] @ [N, H, D] -> [..., S, H, D].
        emb = torch.einsum(
            "...n,nhd->...hd",
            positions[..., self._active_axes],
            inv_freqs,
        )
        return (
            emb.cos().to(dtype=self.dtype, device=self.device),
            emb.sin().to(dtype=self.dtype, device=self.device),
        )

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> Self:
        super()._apply(fn, recurse)
        self._inv_freqs = self._inv_freqs.to(device=self._dtype.device)
        return self

    @classmethod
    def smallest_recommended_base(
        cls,
        dim: int | Iterable[int],
        max_positions: int | Iterable[int],
    ) -> float | tuple[float, ...]:
        """Return the smallest reasonable base for given position range(s).

        The lowest frequency has period 2pi*base^((c-2)/c). Setting this
        >= max_positions gives::

          base = ((max_pos - 1) / (2pi))^(c / (c - 2))

        To dump the reference table:

        >>> for c in [64, 128, 256]:
        ...     for length in [16, 256, 1024, 16384, 131072]:
        ...         b = RoPE.smallest_recommended_base(c, length)
        ...         print(f"  c={c} len={length}: {b:.1f}")

        Args:
          dim: Per-axis channel count(s) (same as __init__).
          max_positions: Max position count per axis. Scalar or iterable.
              Broadcast against dim.

        Returns:
          base: Float (1D) or tuple of floats (ND).

        """
        dims, maxpos = cls._broadcast_args(dim, max_positions)
        bases = []
        for c, m in zip(dims, maxpos, strict=True):
            if c == 0:
                bases.append(0.0)
                continue
            cls._validate_c(c)
            bases.append(((m - 1) / (2 * math.pi)) ** (c / (c - 2)))
        return bases[0] if len(bases) == 1 else tuple(bases)

    @classmethod
    def _make_inv_freqs(cls, base: float, dim: int) -> Tensor:
        """Inverse frequencies: base^linspace(0, -1+2/c, c//2).

        Often this code is written like:

            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        """
        return (
            torch.linspace(
                0,
                math.log(base) * (-1 + 2 / dim),
                dim // 2,
                dtype=torch.float64,
            )
            .exp()
            .float()
        )

    @classmethod
    def _validate_c(cls, c: int) -> None:
        """Validate a single per-axis channel count."""
        if c < 4 or c % 2 != 0:
            raise ValueError(f"Per-axis dim {c} must be even and >= 4.")

    @classmethod
    def _broadcast_args(
        cls,
        dim: int | Iterable[int],
        other: float | Iterable[float] | Iterable[int],
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Broadcast dim and other to equal-length tuples."""
        dims = (int(dim),) if isinstance(dim, int) else tuple(int(d) for d in dim)
        others = (
            (float(other),)
            if isinstance(other, (int, float))
            else tuple(float(o) for o in other)
        )
        if len(dims) == 1 and len(others) > 1:
            dims = dims * len(others)
        if len(others) == 1:
            others = others * len(dims)
        if len(dims) != len(others):
            raise ValueError(
                f"len(dim)={len(dims)} must equal len(other)={len(others)}."
            )
        return dims, others


class RoPEMixed(RoPE):
    """N-dimensional RoPE with per-head frequencies.

    Extends ``RoPE`` with per-head frequency scaling via random
    directions on the N-sphere.

    When ``axial=True`` (default), per-axis channels stay separate
    (learned axial RoPE). When ``axial=False``, all axes share
    channels (summed) — this is RoPE-Mixed from
    [Heo et al., ECCV 2024](https://arxiv.org/abs/2403.13298).

    Returns half-dim cos/sin (shape ``[..., S, H, dim//2]``). Pair
    with ``apply_rope``.

    Args:
      dim: Channel count per axis (same semantics as ``RoPE``).
      num_heads: Number of attention heads.
      base: Frequency base(s) (same semantics as ``RoPE``).
      axial: If True, axial (cat). If False (default), sum.
      learnable: If True, frequencies are learnable. Default False.

    """

    def __init__(
        self,
        dim: int | Iterable[int],
        *,
        num_heads: int = 1,
        base: float | Iterable[float] = 10e3,
        axial: bool = False,
        learnable: bool = False,
        legacy: bool = False,
    ):
        super().__init__(dim, base=base, legacy=legacy)
        if axial:
            inv_freqs = self._inv_freqs
        else:
            if len(set(self._dims)) > 1:
                raise ValueError(f"Sum mode requires uniform dims, got {self._dims}.")
            inv_freqs = torch.stack(
                [
                    self._make_inv_freqs(b, c)
                    for b, c in zip(self._bases, self._dims, strict=False)
                ]
            ).unsqueeze(-2)
        if num_heads > 1:
            directions = nn.functional.normalize(
                trunc_normal_init_(
                    torch.empty(len(self._active_axes), num_heads, 1),
                    std=1.0,
                ),
                dim=0,
            )
            inv_freqs = inv_freqs * directions
        self._inv_freqs = nn.Parameter(inv_freqs, requires_grad=learnable)

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> Self:
        inv_freqs = self._inv_freqs.data.clone()
        nn.Module._apply(self, fn, recurse)  # noqa: SLF001
        self._inv_freqs.data = inv_freqs.to(device=self._dtype.device)
        return self


class Linear(nn.Module):
    """Linear with configurable init."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        bias: bool = True,
        init_weight_fn: InitFn = normal_init_,
        init_bias_fn: InitFn = kaiming_uniform_init_,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.weight = nn.Parameter(
            init_weight_fn(
                torch.empty(
                    [c_out, c_in],
                    dtype=dtype,
                ),
            ),
        )
        if bias:
            # TODO(josh): Consider adding to this "if": "or has kwarg 'c_in'".
            if init_bias_fn is kaiming_uniform_init_:
                init_bias_fn = functools.partial(kaiming_uniform_init_, c_in=c_in)
            self.bias = nn.Parameter(
                init_bias_fn(
                    torch.empty(
                        c_out,
                        dtype=dtype,
                    ),
                ),
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return nn.functional.linear(x, w, b)


default_norm_fn = functools.partial(
    nn.RMSNorm,
    eps=1e-5,
    elementwise_affine=False,
)
default_act_fn = nn.functional.silu


class EnsembleLinear(nn.Module):
    """Linear with ensemble dim for per-head Muon orthogonalization.

    Weight shape: [num_ensemble, c_out, c_in]
    Output shape: [..., num_ensemble, c_out]
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        num_ensemble: int,
        bias: bool = False,
        init_weight_fn: InitFn = normal_init_,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            init_weight_fn(torch.empty([num_ensemble, c_out, c_in])),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    num_ensemble,
                    c_out,
                ),
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., c_in] -> [..., num_ensemble, c_out]
        w = self.weight.to(x.dtype)
        out = torch.einsum("...c,edc->...ed", x, w)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


class Embedding(nn.Module):
    """Embedding with truncated normal init."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_weight_fn: InitFn = trunc_normal_init_,  # std = rsqrt(c_in)
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            init_weight_fn(
                torch.empty(
                    [num_embeddings, embedding_dim],
                    dtype=dtype,
                ),
            ),
        )

    def forward(self, x: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        return nn.functional.embedding(x, self.weight.to(dtype))


def make_grid_positions(
    grid_shape: tuple[int, ...],
    num_prefix_tokens: int,
    device: torch.device | str | None = "cpu",
    legacy: bool = False,
) -> Tensor:
    """Build [S, N] position tensor for an N-D grid with prefix tokens.

    Prefix tokens get sequential positions along the last axis (other axes 0).
    Grid tokens get 1-offset on axis 0 only (so prefix and grid never collide).
    For N=1 this is equivalent to arange(num_prefix + prod(grid_shape)).

    When legacy=True, uses the old convention (1-offset on ALL axes).
    """
    n = len(grid_shape)
    grid = torch.meshgrid(
        *(torch.arange(s, device=device) for s in grid_shape),
        indexing="ij",
    )
    grid_pos = torch.stack([g.reshape(-1) for g in grid], dim=-1)
    if legacy:
        grid_pos = 1 + grid_pos
    else:
        grid_pos[:, 0] += num_prefix_tokens
    prefix = torch.zeros(
        num_prefix_tokens,
        n,
        dtype=torch.long,
        device=device,
    )
    prefix[:, -1] = torch.arange(num_prefix_tokens, device=device)
    return torch.cat([prefix, grid_pos])


class Attention(nn.Module):
    """Multi-head attention with fused QKV EnsembleLinear for Muon orthogonalization."""

    def __init__(
        self,
        c_in: int,
        *,
        num_heads: int,
        num_key_value_heads: int | None = None,
        causal: bool = False,
        muon_modified: bool = False,
        checkpoint_muon_norm: bool = False,
        qk_norm: bool = False,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        init_weight_fn: InitFn = functools.partial(normal_init_, std=0.02),
    ):
        assert num_heads > 0
        super().__init__()
        self.head_dim = c_in // num_heads
        self.num_heads = num_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.checkpoint_muon_norm = checkpoint_muon_norm

        # Fused QKV: one kernel, each head orthogonalized independently by Muon
        # Ensemble dim = num_heads + 2 * num_key_value_heads (Q heads + K heads + V heads)
        self.qkv_proj = EnsembleLinear(
            c_in,
            self.head_dim,
            num_ensemble=num_heads + 2 * num_key_value_heads,
            bias=False,
            init_weight_fn=init_weight_fn,
        )
        self.o_proj = Linear(
            c_in,
            c_in,
            bias=False,
            init_weight_fn=init_weight_fn,
        )
        self.o_norm = norm_fn(c_in) if muon_modified else None
        self.qk_norm = norm_fn(self.head_dim) if qk_norm else None

    def forward(
        self,
        x: Tensor,  # S (H*D)
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        q, k, v = self.qkv_proj(x).split(
            [
                self.num_heads,
                self.num_key_value_heads,
                self.num_key_value_heads,
            ],
            dim=-2,
        )

        if self.qk_norm is not None:
            q = self.qk_norm(q)
            k = self.qk_norm(k)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rope(q, k, cos, sin)

        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))  # S H D -> H S D
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(-3, -2).flatten(-2)  # H S D -> S (H D)
        if self.o_norm is not None:
            if self.checkpoint_muon_norm:
                out = torch_checkpoint(self.o_norm, out, use_reentrant=False)
            else:
                out = self.o_norm(out)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU with optional muon_modified mode (norm before down_proj for Muon).

    muon_modified=True: sigmoid(gate) * norm(gate * x)
    muon_modified=False: silu(gate) * x
    """

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        gate: bool = True,
        multiple_of: int = 256,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        muon_modified: bool = True,
        init_weight_fn: InitFn = normal_init_,
    ):
        super().__init__()
        self.gate = gate
        if gate:
            expansion *= 2 / 3
        c_hidden = _find_multiple(round(expansion * c_in), multiple_of)
        self.up_proj = Linear(
            c_in,
            (c_hidden * 2) if gate else c_hidden,
            bias=False,
            init_weight_fn=init_weight_fn,
        )
        self.down_proj = Linear(
            c_hidden,
            c_in,
            bias=False,
            init_weight_fn=init_weight_fn,
        )
        if muon_modified:
            if act_fn is not nn.functional.silu:
                raise NotImplementedError(
                    "Muon modified SwiGLU is only defined whent "
                    f"act_fn is nn.functional.silu ({act_fn})."
                )
            self.norm = norm_fn(c_hidden)
        else:
            self.norm = None
        self.act = act_fn

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        if self.gate:
            gate, x = x.chunk(2, dim=-1)
            if self.norm is None:
                x = self.act(gate) * x
            else:
                # When SwiGLU is `silu(gate) * x` then this is the same as:
                # sigmoid(gate) * gate * x. But `gate * x` can have magnitude
                # information which is problematic for Muon. So instead we norm
                # just that part.
                x = torch.sigmoid(gate) * self.norm(gate * x)
        else:
            x = self.act(x)
        # TODO(josh): Muon may also benefit from placing the norm here instead of
        # only pre-gate.
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP. Uses post-norm."""

    def __init__(
        self,
        c_in: int,
        *,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        prenorm: bool = False,
        # MLP.
        expansion: float = 4.0,
        multiple_of: int = 256,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        gate: bool = True,
        mlp_muon_modified: bool = True,
        mlp_init_weight_fn: InitFn = normal_init_,
        # Attention specific kwargs.
        num_heads: int = 0,  # Required >0.
        num_key_value_heads: int | None = None,
        causal: bool = False,
        attn_muon_modified: bool = False,
        attn_checkpoint_muon_norm: bool = False,
        attn_qk_norm: bool = False,
        attn_init_weight_fn: InitFn = functools.partial(normal_init_, std=0.02),
        checkpoint: bool = False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.attn = Attention(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
            muon_modified=attn_muon_modified,
            checkpoint_muon_norm=attn_checkpoint_muon_norm,
            qk_norm=attn_qk_norm,
            norm_fn=norm_fn,
            init_weight_fn=attn_init_weight_fn,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            muon_modified=mlp_muon_modified,
            init_weight_fn=mlp_init_weight_fn,
        )
        self.prenorm = prenorm
        self.norm1 = norm_fn(c_in)
        self.norm2 = norm_fn(c_in)

    def _forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if self.prenorm:
            z = self.norm1(x)
            x = x + self.attn(z, cos_sin=cos_sin)
            z = self.norm2(x)
            x = x + self.mlp(z)
        else:
            x = x + self.attn(x, cos_sin=cos_sin)
            x = self.norm1(x)
            x = x + self.mlp(x)
            x = self.norm2(x)
        return x

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if self.checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(  # pyright: ignore[reportAttributeAccessIssue]
                self._forward,
                x,
                cos_sin,
                use_reentrant=False,
            )
        return self._forward(x, cos_sin)


class MLPMixerBlock(nn.Module):
    """MLP-Mixer block with token-mixing and channel-mixing. Uses post-norm."""

    def __init__(
        self,
        c_in: int,
        *,
        prenorm: bool = False,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        muon_modified: bool = True,
        gate: bool = True,
        # MLPMixer specific kwargs.
        seq_len: int = 0,
        token_multiple_of: int | None = None,
        init_weight_fn: InitFn = normal_init_,
    ):
        assert seq_len > 0
        super().__init__()
        if token_multiple_of is None:
            token_multiple_of = multiple_of
        self.attn = SwiGLU(
            seq_len,
            expansion=expansion,
            multiple_of=token_multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            muon_modified=muon_modified,
            init_weight_fn=init_weight_fn,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            muon_modified=muon_modified,
            init_weight_fn=init_weight_fn,
        )
        self.prenorm = prenorm
        self.norm1 = norm_fn(seq_len)
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if cos_sin is not None:
            raise NotImplementedError("MLPMixerBlock does not support RoPE")
        if self.prenorm:
            x = x.transpose(-2, -1)  # [*B, D, S]
            z = self.norm1(x)  # Specifically doing norm on S.
            x = x + self.attn(z)
            x = x.transpose(-2, -1)  # [*B, S, D]
            z = self.norm2(x)
            x = x + self.mlp(z)
        else:
            x = x.transpose(-2, -1)  # [*B, D, S]
            x = x + self.attn(x)
            x = self.norm1(x)  # Specifically doing norm on S.
            x = x.transpose(-2, -1)  # [*B, S, D]
            x = x + self.mlp(x)
            x = self.norm2(x)
        return x


class BatchNorm(nn.BatchNorm1d):
    """BatchNorm1d for (B, L, C) input."""

    def __init__(self, num_features: int, momentum: float = 0.4, eps: float = 1e-12):
        super().__init__(num_features, momentum=momentum, eps=eps, affine=False)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.reshape(-1, input.shape[-1])).reshape(*input.shape)


class GroupNorm(nn.GroupNorm):
    """GroupNorm for (B, L, C) input."""

    def __init__(self, num_features: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__(num_groups, num_features, eps=eps, affine=False)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.transpose(-2, -1)).transpose(-1, -2)


class Sequential(nn.Sequential):
    """Sequential that passes args and kwargs to blocks."""

    def forward(self, z: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        for block in self:
            z = block(z, *args, **kwargs)
        return z


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: ModuleProtocol, decay: float = 0.9):
        self.decay = decay
        self.shadow = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: dict[str, Tensor] = {}

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def update(self, model: ModuleProtocol) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].lerp_(p.data, 1 - self.decay)

    def apply(self, model: ModuleProtocol) -> None:
        self.backup = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model: ModuleProtocol) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}


class TRM(nn.Module):
    """Tiny Recursive Model."""

    @dataclasses.dataclass(slots=True, kw_only=True)
    class Config:
        vocab_size: int = 10 + 1 + 1
        seq_len: int = 9 * 9  # puzzle grid tokens only (no HALT)
        hidden_size: int = 512
        num_heads: int = 8
        num_layers: int = 2
        H_cycles: int = 6
        L_cycles: int = 9
        state_noise: float = 0.0

        rope_kwargs: dict[str, Any] = dataclasses.field(
            default_factory=lambda: {
                "base": 10e3,
                "axial": True,
                "learnable": False,
                "legacy": True,
            }
        )
        dtype: torch.dtype | None = torch.bfloat16
        head_bias: bool = True
        act: bool = True
        act_q_head_bias_init: float = -5.0
        block_fn: Callable[[int], nn.Module] = functools.partial(  # noqa: RUF009
            MLPMixerBlock,
            seq_len=seq_len,
        )
        use_rope: bool = False
        causal: bool = False
        no_grad_inner: bool = True  # False = BPTT through all H_cycles
        head_init_weight_fn: InitFn = normal_init_

        compile_core: bool = True
        compile_reasoning: bool = False
        # Compiles:
        # - Train
        #   - _core no grad context
        #   - _core yes grad context
        # - Eval
        #   - [no additional compiles assuming same bs]
        max_num_compile_core: int = 3

        @property
        def n_iters(self) -> int:
            return self.H_cycles * (self.L_cycles + 1)

        def setup(self, *args: Any, **kwargs: Any) -> TRM:
            return TRM(self, *args, **kwargs)

    def __init__(self, config: TRM1ConfigProtocol):
        super().__init__()
        self.config: TRM1ConfigProtocol = config
        self.embed_scale = config.hidden_size**0.5
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_weight_fn=functools.partial(
                trunc_normal_init_,
                std=1 / self.embed_scale,
            ),
        )
        self.head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=config.head_bias,
            init_weight_fn=config.head_init_weight_fn,
            init_bias_fn=nn.init.zeros_,
        )

        # RoPE (only for attention blocks)
        if config.use_rope:
            self.rope = RoPEMixed(
                config.hidden_size // config.num_heads,
                **config.rope_kwargs,
            )
        else:
            self.rope = None

        if config.act:
            self.q_head = Linear(
                config.hidden_size,
                1,
                bias=True,
                init_weight_fn=nn.init.zeros_,
                init_bias_fn=functools.partial(
                    nn.init.constant_,
                    val=config.act_q_head_bias_init,
                ),
            )
            # NOTE: Must consume RNG for backwards compat.
            _ = normal_init_(torch.empty_like(self.q_head.weight))
        else:
            self.q_head = None

        if config.block_fn is None:
            raise ValueError("block_fn is required")
        self.reasoning = Sequential(
            *[config.block_fn(config.hidden_size) for _ in range(config.num_layers)],
        )
        if config.compile_reasoning:
            # Doing as a side-effect means checkpoints are unaffected.
            self.reasoning.compile(mode="default", fullgraph=True)

        self.H_init: Tensor = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=config.dtype),
                std=1,
            ),
            persistent=True,
        )
        self.L_init: Tensor = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=config.dtype),
                std=1,
            ),
            persistent=True,
        )

    def core(
        self,
        input_emb: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compiled core - one H-cycle. Fixed signature for torch.compile.

        Args:
            input_emb: Embedded input tokens (B, L, hidden_size)
            z_H: H-state (B, L, hidden_size)
            z_L: L-state (B, L, hidden_size)
            cos_sin: RoPE embeddings or None

        Returns:
            (logits, q_halt, z_H, z_L) - all tensors, no conditionals

        """
        L_cycles = self.config.L_cycles
        c = z_H + input_emb
        for _ in range(L_cycles):
            z_L = self.reasoning(z_L + c, cos_sin)
        z_H = self.reasoning(z_H + z_L, cos_sin)

        if self.training and self.config.state_noise > 0:
            z_H = z_H + torch.randn_like(z_H) * self.config.state_noise

        logits = self.head(z_H)
        q_halt = (
            self.q_head(z_H[:, 0]).squeeze(-1)
            if self.q_head is not None
            else z_H.new_zeros(z_H.shape[0])
        )

        return logits, q_halt, z_H, z_L

    @torch.compile(mode="default", fullgraph=True)
    def core_compiled(
        self,
        input_emb: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if torch.compiler.is_compiling():
            _ = trace_compile(
                "core_compiled",
                max_compiles=self.config.max_num_compile_core,
                always_print=False,
            )
        return self.core(input_emb, z_H, z_L, cos_sin)

    def step(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Any]:
        """Single H-cycle step for ACT training/eval loops.

        Args:
            input_ids: Input token IDs (B, L)
            z_H: Current H-state (B, L, hidden_size)
            z_L: Current L-state (B, L, hidden_size)

        Returns:
            dict with keys:
                - logits: Logits (B, L, vocab_size)
                - q_halt: Halt probability logit (B,)
                - z_H: Updated H-state (B, L, hidden_size)
                - z_L: Updated L-state (B, L, hidden_size)

        """
        input_emb = self.embed_scale * self.embed_tokens(input_ids, self.config.dtype)
        cos_sin = (
            None
            if self.rope is None
            else self.rope(torch.arange(input_ids.shape[-1], device=input_ids.device))
        )
        core = self.core_compiled if self.config.compile_core else self.core
        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, cos_sin)
        return {
            "logits": logits,
            "q_halt": q_halt,
            "z_H": z_H.detach(),
            "z_L": z_L.detach(),
        }

    def forward(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Any]:
        """Convenience wrapper - runs H_cycles, collects outputs.

        Args:
            input_ids: Input token IDs (B, L)
            z_H: Initial H-state (B, L, hidden_size)
            z_L: Initial L-state (B, L, hidden_size)

        Returns:
            dict with keys:
                - logits: Final logits (B, L, vocab_size)
                - all_logits: List of logits at each H-cycle
                - q_halt: Halt probability logit (B,)
                - z_H: Final H-state (B, L, hidden_size)
                - z_L: Final L-state (B, L, hidden_size)
                - all_z_H: List of z_H at each H-cycle

        """
        H_cycles = self.config.H_cycles
        input_emb = self.embed_scale * self.embed_tokens(input_ids, self.config.dtype)
        cos_sin = (
            None
            if self.rope is None
            else self.rope(torch.arange(input_ids.shape[-1], device=input_ids.device))
        )

        all_logits = []
        all_z_H = []

        core = self.core_compiled if self.config.compile_core else self.core

        if self.config.no_grad_inner:
            with torch.no_grad():
                for _ in range(H_cycles - 1):
                    logits, q_halt, z_H, z_L = core(
                        input_emb.detach(),
                        z_H.detach(),
                        z_L.detach(),
                        cos_sin,
                    )
                    all_logits.append(logits)
                    all_z_H.append(z_H)
        else:
            for _ in range(H_cycles - 1):
                logits, q_halt, z_H, z_L = core(
                    input_emb,
                    z_H,
                    z_L,
                    cos_sin,
                )
                # TODO(josh): logits should probably be detached but to do so now means
                # we need to look at every downstream use.
                all_logits.append(logits)
                all_z_H.append(z_H.detach())

        # Final H-cycle with gradients
        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, cos_sin)
        # TODO(josh): Unclear if this logits should also be detached? Same issue as
        # above.
        all_logits.append(logits)
        all_z_H.append(z_H.detach())

        return {
            "logits": logits,
            "all_logits": all_logits,
            "q_halt": q_halt,
            "z_H": z_H.detach(),
            "z_L": z_L.detach(),
            "all_z_H": all_z_H,
        }


class TRM2(TRM):
    @dataclasses.dataclass(slots=True, kw_only=True)
    class Config(TRM.Config):
        block_fn: Callable[[int], nn.Module] = functools.partial(  # noqa: RUF009
            MLPMixerBlock,
            seq_len=9 * 9,  # 81 (puzzle grid only, no HALT)
            init_weight_fn=trunc_normal_init_,
        )
        head_init_weight_fn: InitFn = trunc_normal_init_

        def setup(self, *args: Any, **kwargs: Any) -> TRM2:
            return TRM2(self, *args, **kwargs)


def _block_fn_creates_mlp_mixer(block_fn: Callable[[int], nn.Module]) -> bool:
    """Check if block_fn creates MLPMixerBlock without instantiating.

    Uses `is` identity check intentionally - subclasses would need their own
    validation logic since they may handle seq_len/RoPE differently.
    """
    if isinstance(block_fn, functools.partial):
        return block_fn.func is MLPMixerBlock
    return block_fn is MLPMixerBlock


class TRM3(nn.Module):
    """Tiny Recursive Model."""

    @dataclasses.dataclass(slots=True, kw_only=True)
    class Config:
        vocab_size: int = 10 + 1 + 1  # values + unknown + halt
        puzzle_grid_shape: tuple[int, ...] = (9, 9)
        hidden_size: int = 512
        num_layers: int = 2
        H_cycles: int = 6
        L_cycles: int = 9

        # I dont like this dtype default!
        dtype: torch.dtype | None = torch.bfloat16
        device: torch.device | str | None = "cuda"

        head_bias: bool = True
        # NOTE: Default uses class-level puzzle_grid_shape (81) at definition time.
        # If you change puzzle_grid_shape, you must also provide a compatible block_fn.
        block_fn: Callable[[int], nn.Module] = functools.partial(  # noqa: RUF009
            MLPMixerBlock,
            seq_len=9 * 9 + 1,
            init_weight_fn=trunc_normal_init_,
        )
        block_kwargs_by_layer: dict[int, dict[str, Any]] = dataclasses.field(
            default_factory=dict,
        )
        compile_core: bool = True
        compile_reasoning: bool = False
        max_num_compile_core: int = 3

        # Head configuration
        K_H: int = 1  # z_H pool size
        K_L: int = 4  # z_L pool size
        K_H_active: int | None = None  # active per step (None = K_H)
        K_L_active: int | None = None  # active per step (None = K_L)

        # Pairing policy
        HL_policy: Literal["inner", "outer"] = "inner"
        HL_random_subset_size: int | None = None

        # Carry policy: how to update chains after each step
        # - "top1": only winner chain's state carried, others frozen (keep current state)
        # - "top2": top 2 chains' states carried, others frozen
        # - "copy_top1": all chains get winner's output copied
        # - "copy_top2": half get top1, half get top2 (randperm assignment)
        # - "all": all chains updated with their own output (no selection)
        # - "none": no carry - reset to init each step
        carry_H: Literal[
            "top1",
            "top2",
            "copy_top1",
            "copy_top2",
            "all",
            "none",
        ] = "top1"
        carry_L: Literal[
            "top1",
            "top2",
            "copy_top1",
            "copy_top2",
            "all",
            "none",
        ] = "all"

        # SVD init alignment
        z_H_init_svd: bool = False
        z_L_init_svd: bool = False

        # Embedding rescale trick (TRM-style): init with std=1/sqrt(C), multiply by sqrt(C)
        # This affects gradient magnitudes for Adam. True = TRM behavior, False = pre-scaled.
        embedding_rescale_trick: bool = True

        # Delta reparameterization: z = z_init + z_delta where delta is detached.
        # Allows gradients to flow to H_init/L_init in final H-cycle.
        z_delta_reparam: bool = False

        # Random z init: sample from trunc_normal(0, 1) instead of learned H_init/L_init
        z_H_random_init: bool = False
        z_L_random_init: bool = False

        use_rope: bool = False
        rope_kwargs: dict[str, Any] = dataclasses.field(
            default_factory=lambda: {
                "base": 10e3,
                "axial": True,
                "learnable": False,
                "legacy": True,
            }
        )
        num_heads: int = 8  # Only used for RoPE dim calculation

        # Prefix tokens: prepend before puzzle_grid
        # Sequence order: [puzzle_id_tokens..., register_tokens..., puzzle_grid...]
        # - puzzle_id_tokens: per-puzzle learned tokens (embedding table)
        # - register_tokens: shared learned tokens (puzzle-agnostic)
        # q_head reads from q_halt_seq_index position in the full sequence
        num_puzzle_id_tokens: int = 0  # Per-puzzle learned tokens
        num_register_tokens: int = (
            1  # Shared learned tokens (replaces old HALT token role)
        )
        register_token_init_std: float = 1.0  # 0 = zero-init (TRM reference)
        register_tokens_learnable: bool = (
            True  # False = fixed (reference uses zero-pad, not learnable)
        )
        num_puzzle_ids: int = 0  # Size of puzzle ID embedding table
        q_halt_seq_index: int = 0  # Sequence position from which q_head reads

        # If False, head outputs vocab_size-1 classes (no PAD logit).
        # Labels must be shifted: labels-1 with ignore_index=-1.
        label_smoothing_includes_pad_token: bool = True

        # Whether TRM3.__init__ casts all params/buffers to config.dtype via self.to().
        # Set False to keep RoPE buffers in float32 (needed for b4b match with reference).
        cast_model_to_dtype: bool = True

        # Damping factor for iterative reasoning: z = (1-α)·z + α·reasoning(z).
        # 0.0 = no damping (default), 0.5 = half step. Enforces contractivity
        # when reasoning block uses post-norm (which constrains output magnitude).
        core_damping: float = 0.0

        # Anchor: sequence index reset to a learned embedding every
        # reasoning() call in core(). Fixed reference point for RoPE-based
        # position recovery. None = off.
        anchor_seq_index: int | None = None

        @property
        def num_puzzle_grid_tokens(self) -> int:
            return math.prod(self.puzzle_grid_shape)

        @property
        def total_seq_len(self) -> int:
            """Total sequence length: puzzle_id + register + puzzle_grid."""
            return (
                self.num_puzzle_id_tokens
                + self.num_register_tokens
                + self.num_puzzle_grid_tokens
            )

        @property
        def K_L_eff(self) -> int:
            """Effective active L heads per step."""
            return self.K_L if self.K_L_active is None else self.K_L_active

        @property
        def K_H_eff(self) -> int:
            """Effective active H heads per step."""
            return self.K_H if self.K_H_active is None else self.K_H_active

        @property
        def num_effective_heads(self) -> int:
            """Number of (H, L) combinations per step."""
            if self.HL_policy == "inner":
                n = max(self.K_L_eff, self.K_H_eff)
            elif self.HL_policy == "outer":
                n = self.K_L_eff * self.K_H_eff
            else:
                raise ValueError(f"Unrecognized HL_policy={self.HL_policy}.")
            if self.HL_random_subset_size is None:
                return n
            return min(n, self.HL_random_subset_size)

        def setup(self, *args: Any, **kwargs: Any) -> TRM3:
            return TRM3(self, *args, **kwargs)

    def __init__(self, config: Config):
        super().__init__()
        if config.device is not None:
            config.device = torch.device(config.device)

        # Validate: MLPMixerBlock requires seq_len to match actual sequence length.
        # With prefix tokens, total_seq_len > num_puzzle_grid_tokens, so the
        # default block_fn (MLPMixerBlock with seq_len=num_puzzle_grid_tokens+1)
        # will have mismatched dimensions. User must provide a compatible block_fn.
        if (
            config.num_puzzle_id_tokens > 0 or config.num_register_tokens > 0
        ) and _block_fn_creates_mlp_mixer(config.block_fn):
            raise ValueError(
                "MLPMixerBlock cannot be used with prefix tokens "
                f"(num_puzzle_id_tokens={config.num_puzzle_id_tokens}, "
                f"num_register_tokens={config.num_register_tokens}). "
                "Use TransformerBlock or another block that doesn't require seq_len.",
            )

        # Validate: num_puzzle_ids must be set when using puzzle_id_tokens
        if config.num_puzzle_id_tokens > 0 and config.num_puzzle_ids <= 0:
            raise ValueError(
                f"num_puzzle_ids must be > 0 when num_puzzle_id_tokens > 0 "
                f"(got num_puzzle_ids={config.num_puzzle_ids})",
            )

        # Validate: MLPMixerBlock doesn't support RoPE (raises at runtime otherwise)
        if config.use_rope and _block_fn_creates_mlp_mixer(config.block_fn):
            raise ValueError(
                "MLPMixerBlock does not support RoPE. "
                "Use TransformerBlock or set use_rope=False.",
            )

        self.config = config

        # Embedding rescale trick: init small, multiply at runtime (like TRM).
        # This gives larger gradients -> different Adam dynamics than pre-scaled.
        if config.embedding_rescale_trick:
            self.embed_scale = config.hidden_size**0.5
        else:
            self.embed_scale = 1.0

        # NOTE: Init order must match TRM for RNG compatibility:
        # embed_tokens -> head -> q_head -> reasoning -> H_init -> L_init
        # PRNG_EQUIVALENCE: Don't pass dtype to Embedding - TRM doesn't, and it affects RNG.
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            init_weight_fn=functools.partial(
                trunc_normal_init_,
                std=1.0 / self.embed_scale,
            ),
        )
        self.head = Linear(
            config.hidden_size,
            config.vocab_size - (not config.label_smoothing_includes_pad_token),
            bias=config.head_bias,
            init_weight_fn=trunc_normal_init_,  # std = rsqrt(hidden_size)
            init_bias_fn=nn.init.zeros_,
        )
        self.q_head = Linear(
            config.hidden_size,
            1,
            bias=True,
            init_weight_fn=nn.init.zeros_,
            init_bias_fn=functools.partial(nn.init.constant_, val=-5.0),
        )
        # PRNG_EQUIVALENCE: Must consume RNG for backwards compat.
        _ = normal_init_(torch.empty_like(self.q_head.weight))
        self.reasoning = Sequential(
            *[
                config.block_fn(
                    config.hidden_size,
                    **config.block_kwargs_by_layer.get(i, {}),
                )
                for i in range(config.num_layers)
            ],
        )

        # Puzzle ID embedding: per-puzzle learned tokens
        if config.num_puzzle_id_tokens > 0:
            self.puzzle_id_embed: Embedding | None = Embedding(
                config.num_puzzle_ids,
                config.num_puzzle_id_tokens * config.hidden_size,
                init_weight_fn=functools.partial(
                    trunc_normal_init_,
                    std=0,
                ),  # Zero init
            )
        else:
            self.puzzle_id_embed = None

        # Register tokens: shared tokens (puzzle-agnostic)
        # Init with std=1/embed_scale so after scaling they have unit variance.
        # Created on CPU; moved to device by self.to() call later (standard nn.Parameter pattern).
        if config.num_register_tokens > 0:
            reg_data = trunc_normal_init_(
                torch.empty(config.num_register_tokens, config.hidden_size),
                std=config.register_token_init_std / self.embed_scale,
            )
            if config.register_tokens_learnable:
                self.register_tokens: nn.Parameter | nn.Buffer | None = nn.Parameter(
                    reg_data
                )
            else:
                self.register_tokens = nn.Buffer(reg_data, persistent=True)
        else:
            self.register_tokens = None

        if config.use_rope:
            self.rope = RoPEMixed(
                config.hidden_size // config.num_heads,
                **config.rope_kwargs,
            )
        else:
            self.rope = None

        # PRNG_EQUIVALENCE: H_init and L_init: Only create first row here.
        # x177 creates L_init_1..K-1 AFTER .to() using CUDA RNG (torch.empty_like on GPU).
        # For RNG compat, x179.setup_model() must expand L_init the same way AFTER .to().
        # TRM3 creates [1, hidden] here; x179 expands to [K_L, hidden] in setup_model.
        # PRNG_EQUIVALENCE: Must use dtype=config.dtype because uniform_() produces
        # different values for bfloat16 vs float32 tensors with the same RNG seed.
        init_fn = functools.partial(trunc_normal_init_, std=1)
        self.H_init: Tensor = nn.Buffer(
            init_fn(torch.empty([1, config.hidden_size], dtype=config.dtype)),
            persistent=True,
        )
        self.L_init: Tensor = nn.Buffer(
            init_fn(torch.empty([1, config.hidden_size], dtype=config.dtype)),
            persistent=True,
        )

        # PRNG_EQUIVALENCE: This gets the RNG the same as old way.
        # Must also convert to dtype so SVD alignment uses same precision as x177.
        if config.cast_model_to_dtype:
            self.to(device=config.device, dtype=config.dtype)
        else:
            self.to(device=config.device)

        # PRNG_EQUIVALENCE: Expand L_init from [1, hidden] to [K_L, hidden].
        # x177 creates L_init_1..K-1 after .to() using CUDA RNG (torch.empty_like on GPU tensor).
        # We must do the same for RNG compatibility.
        # Must create each row separately to match x177's RNG consumption order.
        # CUDA RNG advances differently for [K-1, hidden] vs K-1 separate [hidden] tensors.
        if config.K_L > 1:
            l_rows = [
                self.L_init.data[0],
                *(
                    init_fn(torch.empty_like(self.L_init.data[0]))
                    for _ in range(1, config.K_L)
                ),
            ]
            self.L_init = nn.Buffer(torch.stack(l_rows, dim=0), persistent=True)

        if config.K_H > 1:
            h_rows = [
                self.H_init.data[0],
                *(
                    init_fn(torch.empty_like(self.H_init.data[0]))
                    for _ in range(1, config.K_H)
                ),
            ]
            self.H_init = nn.Buffer(torch.stack(h_rows, dim=0), persistent=True)

        # SVD alignment must run before compile (needs subscriptable reasoning)
        if config.z_L_init_svd:
            self._equalize_svd_alignment(self.L_init)
        if config.z_H_init_svd:
            self._equalize_svd_alignment(self.H_init)

        if config.compile_reasoning:
            # Doing as a side-effect means checkpoints are unaffected.
            self.reasoning.compile(mode="default", fullgraph=True)

        self.register_buffer("_dummy", torch.empty(0))

    @property
    def device(self) -> torch.device:
        dev = self._dummy.device
        assert isinstance(dev, torch.device)
        return dev

    @property
    def dtype(self) -> torch.dtype:
        dt = self._dummy.dtype
        assert isinstance(dt, torch.dtype)
        return dt

    def init_carry(self, batch_size: int) -> CarryState:
        """Initialize model carry state with N = K_H * K_L chains."""
        # K_H * K_L for outer, max(K_H, K_L) for inner
        cfg = self.config
        N = cfg.num_effective_heads
        S = cfg.total_seq_len  # Use total_seq_len (includes puzzle_id_tokens)

        init_fn = functools.partial(trunc_normal_init_, std=1)

        # Only sample indices if needed (avoid wasteful call when using random init)
        h_indices: Tensor | None = None
        l_indices: Tensor | None = None
        if not cfg.z_H_random_init or not cfg.z_L_random_init:
            h_indices, l_indices = self._sample_head_indices()

        # Initialize z_H for each chain
        if cfg.z_H_random_init:
            z_H = init_fn(
                torch.empty(
                    [batch_size, N, S, cfg.hidden_size],
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            if cfg.z_H_init_svd:
                self._apply_svd_alignment_batched(z_H)
        else:
            # Each chain gets H_init[h_indices[chain_idx]]
            assert h_indices is not None
            z_H_pool = self.H_init.unsqueeze(-2).expand(-1, S, -1)  # [K_H, S, C]
            z_H = z_H_pool[h_indices].unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Initialize z_L for each chain
        if cfg.z_L_random_init:
            z_L = init_fn(
                torch.empty(
                    [batch_size, N, S, cfg.hidden_size],
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            if cfg.z_L_init_svd:
                self._apply_svd_alignment_batched(z_L)
        else:
            # Each chain gets L_init[l_indices[chain_idx]]
            assert l_indices is not None
            z_L_pool = self.L_init.unsqueeze(-2).expand(-1, S, -1)  # [K_L, S, C]
            z_L = z_L_pool[l_indices].unsqueeze(0).expand(batch_size, -1, -1, -1)

        carry_count = torch.zeros(
            batch_size,
            N,
            device=self.device,
            dtype=torch.long,
        )
        return {
            "z_H": z_H.contiguous(),
            "z_L": z_L.contiguous(),
            "carry_count": carry_count,
        }

    def reset_carry_at_indices(
        self,
        carry: CarryState,
        indices: Tensor,
        n_reset: int,
    ) -> CarryState:
        """Reset carry state at specified batch indices."""
        fresh_carry = self.init_carry(n_reset)
        for i in range(n_reset):
            carry["z_H"][indices[i]] = fresh_carry["z_H"][i]
            carry["z_L"][indices[i]] = fresh_carry["z_L"][i]
            carry["carry_count"][indices[i]] = fresh_carry["carry_count"][i]
        return carry

    def wta_forward(
        self,
        input_ids: Tensor,
        carry: CarryState,
        labels: Tensor,
        *,
        puzzle_ids: Tensor | None = None,
        label_smoothing: float = 0.0,
        z_L_noise: float = 0.0,
    ) -> WTAForwardOutput:
        """WTA forward pass with loss computation.

        Carry state has N chains, each with (z_H, z_L). We run all chains through
        the model and update based on carry_H/carry_L policies.
        """
        cfg = self.config
        B = input_ids.shape[0]
        num_input_tokens = cfg.num_puzzle_grid_tokens
        total_seq_len = cfg.total_seq_len
        hidden = cfg.hidden_size

        # Unpack carry: [B, N, total_seq_len, C] for each
        z_H = carry["z_H"]
        z_L = carry["z_L"]
        carry_count = carry["carry_count"]
        N = z_H.shape[1]

        # Get index mapping (for debugging/logging)
        h_indices, l_indices = self._sample_head_indices()

        # Apply noise if requested
        z_L_active = z_L
        if z_L_noise != 0:
            z_L_active = z_L + torch.randn_like(z_L) * z_L_noise

        # Batch for model: [B*N, S, C]
        inputs_batched = (
            input_ids.unsqueeze(1)
            .expand(B, N, num_input_tokens)
            .reshape(B * N, num_input_tokens)
        )
        z_H_batched = z_H.reshape(B * N, total_seq_len, hidden)
        z_L_batched = z_L_active.reshape(B * N, total_seq_len, hidden)

        # Expand puzzle_ids if provided: [B] -> [B*N]
        puzzle_ids_batched = None
        if puzzle_ids is not None:
            puzzle_ids_batched = puzzle_ids.unsqueeze(1).expand(B, N).reshape(B * N)

        # Forward pass
        out = self.forward(inputs_batched, z_H_batched, z_L_batched, puzzle_ids_batched)

        # Reshape back: [B*N, ...] -> [B, N, ...]
        logits_raw = out["logits"]
        q_halt_raw = out["q_halt"]
        z_H_out_raw = out["z_H"]
        z_L_out_raw = out["z_L"]
        assert isinstance(logits_raw, Tensor)
        assert isinstance(q_halt_raw, Tensor)
        assert isinstance(z_H_out_raw, Tensor)
        assert isinstance(z_L_out_raw, Tensor)

        # logits_raw is [B*N, num_puzzle_grid_tokens, V] (already sliced for prefix)
        logits_all = logits_raw.reshape(B, N, cfg.num_puzzle_grid_tokens, -1).float()
        q_halt_all = q_halt_raw.reshape(B, N).float()
        z_H_out = z_H_out_raw.reshape(B, N, total_seq_len, hidden)
        z_L_out = z_L_out_raw.reshape(B, N, total_seq_len, hidden)

        # Compute per-chain losses: [B, N]
        labels_flat = labels.reshape(B, -1)
        logits_flat = logits_all.reshape(B * N * cfg.num_puzzle_grid_tokens, -1)
        labels_expanded = (
            labels_flat.unsqueeze(1)
            .expand(B, N, cfg.num_puzzle_grid_tokens)
            .reshape(-1)
        )
        per_cell_loss = nn.functional.cross_entropy(
            logits_flat,
            labels_expanded.long(),
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(B, N, cfg.num_puzzle_grid_tokens)
        losses = per_cell_loss.mean(dim=-1)  # [B, N]

        # WTA winner
        winner_idx = losses.argmin(dim=-1)  # [B]

        # Update carry based on chain policies
        z_H_new, z_L_new, carry_count_new = self._update_carry(
            z_H,
            z_L,
            z_H_out,
            z_L_out,
            losses.detach(),
            carry_count,
        )

        return {
            "logits": logits_all,
            "q_halt": q_halt_all,
            "losses": losses,
            "winner_idx": winner_idx,
            "carry": {
                "z_H": z_H_new,
                "z_L": z_L_new,
                "carry_count": carry_count_new,
            },
            "h_indices": h_indices,
            "l_indices": l_indices,
        }

    def _get_cos_sin(self, device: torch.device | str) -> tuple[Tensor, Tensor] | None:
        if self.rope is None:
            return None
        cfg = self.config
        if len(self.rope.dims) == 1:
            positions = torch.arange(cfg.total_seq_len, device=device)
        else:
            positions = make_grid_positions(
                cfg.puzzle_grid_shape,
                cfg.num_puzzle_id_tokens + cfg.num_register_tokens,
                device=device,
            )
        return self.rope(positions)

    def _prepend_prefix(
        self,
        input_emb: Tensor,
        puzzle_ids: Tensor | None,
    ) -> Tensor:
        """Prepend prefix tokens to input embedding.

        Sequence order: [puzzle_id_tokens..., register_tokens..., input_emb]

        Args:
            input_emb: [B, S, C] - input embeddings (puzzle grid tokens)
            puzzle_ids: [B] - puzzle identifiers (required if num_puzzle_id_tokens > 0)

        Returns:
            [B, total_seq_len, C] - embeddings with prefix prepended

        """
        cfg = self.config
        if cfg.num_puzzle_id_tokens == 0 and cfg.num_register_tokens == 0:
            return input_emb

        B = input_emb.shape[0]
        prefix_parts: list[Tensor] = []

        # Puzzle ID tokens: per-puzzle learned embeddings
        if cfg.num_puzzle_id_tokens > 0:
            if puzzle_ids is None:
                raise ValueError("puzzle_ids required when num_puzzle_id_tokens > 0")
            if self.puzzle_id_embed is None:
                raise RuntimeError("puzzle_id_embed not initialized")
            if (puzzle_ids >= cfg.num_puzzle_ids).any():
                raise ValueError(
                    f"puzzle_ids has values >= num_puzzle_ids ({cfg.num_puzzle_ids})"
                )
            puzzle_emb = self.puzzle_id_embed(puzzle_ids, cfg.dtype)
            puzzle_emb = puzzle_emb.view(B, cfg.num_puzzle_id_tokens, cfg.hidden_size)
            puzzle_emb = self.embed_scale * puzzle_emb
            prefix_parts.append(puzzle_emb)

        # Register tokens: shared learned tokens (expanded to batch)
        if cfg.num_register_tokens > 0:
            if self.register_tokens is None:
                raise RuntimeError("register_tokens not initialized")
            reg_emb = self.register_tokens.unsqueeze(0).expand(B, -1, -1)
            reg_emb = self.embed_scale * reg_emb.to(dtype=cfg.dtype)
            prefix_parts.append(reg_emb)

        # Prefix: [puzzle_id_tokens..., register_tokens...]
        prefix = torch.cat(prefix_parts, dim=1)
        return torch.cat([prefix, input_emb], dim=1)

    def forward(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Multi H-cycle forward. Same shapes as step()."""
        if self.config.z_delta_reparam:
            return self._forward_delta_reparam(input_ids, z_H, z_L, puzzle_ids)
        return self._forward_simple(input_ids, z_H, z_L, puzzle_ids)

    def step(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        """Single H-cycle step.

        Shapes: B = batch, C = hidden_size, V = vocab_size.
        With prefix tokens, z_H/z_L have shape [B, total_seq_len, C].

        Sequence order: [puzzle_id_tokens..., register_tokens..., puzzle_grid...]

        Args:
            input_ids: [B, num_puzzle_grid_tokens] - puzzle grid tokens
            z_H: [B, total_seq_len, C]
            z_L: [B, total_seq_len, C]
            puzzle_ids: [B] optional puzzle identifiers (required if num_puzzle_id_tokens > 0)

        Returns:
            dict: logits [B, num_puzzle_grid_tokens, V], q_halt [B], z_H [B, total_seq_len, C], z_L [B, total_seq_len, C]

        """
        cfg = self.config
        input_emb = self.embed_scale * self.embed_tokens(input_ids, cfg.dtype)
        input_emb = self._prepend_prefix(input_emb, puzzle_ids)

        cos_sin = self._get_cos_sin(input_emb.device)
        core = self.core_compiled if cfg.compile_core else self.core
        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, cos_sin)

        # Slice output to exclude prefix positions (puzzle_id + register tokens)
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        if n_prefix > 0:
            logits = logits[:, n_prefix:]

        return {
            "logits": logits,
            "q_halt": q_halt,
            "z_H": z_H.detach(),
            "z_L": z_L.detach(),
        }

    def _equalize_svd_alignment(self, init_tensor: Tensor) -> None:
        """Adjust init directions to have equal alignment with up_proj's top singular vectors.

        Args:
            init_tensor: [K, hidden_size] tensor (H_init or L_init)

        """
        block: Any = self.reasoning[0]
        up_proj: Tensor = block.mlp.up_proj.weight
        _, _, V = torch.linalg.svd(up_proj.float(), full_matrices=False)
        V_top = V[:10, :].T.to(dtype=init_tensor.dtype)  # [hidden_size, 10]

        # Use first head as reference
        ref = init_tensor[0]
        ref_norm = ref / ref.norm()
        target_alignment = (ref_norm @ V_top).norm().item()

        for k in range(1, init_tensor.shape[0]):
            vec = init_tensor[k]
            orig_norm = vec.norm()

            proj_coef = vec @ V_top
            vec_parallel = proj_coef @ V_top.T
            vec_perp = vec - vec_parallel

            perp_norm_sq = vec_perp.norm() ** 2
            target_sq = target_alignment**2
            new_parallel_norm_sq = target_sq * perp_norm_sq / (1 - target_sq + 1e-8)
            new_parallel_norm = new_parallel_norm_sq.sqrt()

            if vec_parallel.norm() > 1e-8:
                vec_parallel = vec_parallel / vec_parallel.norm() * new_parallel_norm

            vec_new = vec_parallel + vec_perp
            vec_new = vec_new / vec_new.norm() * orig_norm
            init_tensor[k] = vec_new

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def _apply_svd_alignment_batched(self, z: Tensor) -> None:
        """Apply SVD alignment to batched random init tensor in-place.

        Args:
            z: [B, K, S, C] tensor to orthogonalize

        For each (batch, seq_pos), orthogonalizes the K vectors using the same
        SVD alignment logic as _equalize_svd_alignment.

        """
        block: Any = self.reasoning[0]
        up_proj: Tensor = block.mlp.up_proj.weight
        _, _, V = torch.linalg.svd(up_proj.float(), full_matrices=False)
        V_top = V[:10, :].T.to(dtype=z.dtype)  # [C, 10]

        B, K, S, C = z.shape
        del B, S, C

        # Use first head as reference for target alignment
        ref = z[:, 0, :, :]  # [B, S, C]
        ref_norm = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)
        target_alignment = (ref_norm @ V_top).norm(dim=-1)  # [B, S]

        for k in range(1, K):
            vec = z[:, k, :, :].clone()  # [B, S, C]
            orig_norm = vec.norm(dim=-1, keepdim=True)

            proj_coef = vec @ V_top  # [B, S, 10]
            vec_parallel = proj_coef @ V_top.T  # [B, S, C]
            vec_perp = vec - vec_parallel

            perp_norm_sq = (vec_perp**2).sum(dim=-1)  # [B, S]
            target_sq = target_alignment**2
            new_parallel_norm_sq = target_sq * perp_norm_sq / (1 - target_sq + 1e-8)
            new_parallel_norm = new_parallel_norm_sq.sqrt().unsqueeze(-1)

            parallel_norm = vec_parallel.norm(dim=-1, keepdim=True)
            vec_parallel = torch.where(
                parallel_norm > 1e-8,
                vec_parallel / (parallel_norm + 1e-8) * new_parallel_norm,
                vec_parallel,
            )

            vec_new = vec_parallel + vec_perp
            vec_new = vec_new / (vec_new.norm(dim=-1, keepdim=True) + 1e-8) * orig_norm
            z[:, k, :, :] = vec_new

    def _sample_head_indices(self) -> tuple[Tensor, Tensor]:
        """Sample active (H, L) index pairs for this step.

        Returns:
            h_indices: [N] indices into H_init pool
            l_indices: [N] indices into L_init pool

        """
        cfg = self.config
        k_l = cfg.K_L_eff
        k_h = cfg.K_H_eff
        device = self.H_init.device

        # Sample from pools if using population sampling
        if k_l < cfg.K_L:
            l_pool = torch.randperm(cfg.K_L, device=device)[:k_l]
        else:
            l_pool = torch.arange(cfg.K_L, device=device)

        if k_h < cfg.K_H:
            h_pool = torch.randperm(cfg.K_H, device=device)[:k_h]
        else:
            h_pool = torch.arange(cfg.K_H, device=device)

        # Create (H, L) pairs based on policy
        if cfg.HL_policy == "inner":
            if k_h == 1:
                h_indices = h_pool.expand(k_l)
                l_indices = l_pool
            elif k_l == 1:
                h_indices = h_pool
                l_indices = l_pool.expand(k_h)
            else:
                if k_l != k_h:
                    raise ValueError(
                        "inner requires K_L_eff == K_H_eff or 1 in (K_L_eff, K_H_eff)",
                    )
                h_indices = h_pool
                l_indices = l_pool
        else:  # outer
            h_indices = h_pool.repeat_interleave(k_l)
            l_indices = l_pool.repeat(k_h)

        # Subsample if requested
        if (
            cfg.HL_random_subset_size is not None
            and len(h_indices) > cfg.HL_random_subset_size
        ):
            perm = torch.randperm(
                len(h_indices),
                device=device,
            )[: cfg.HL_random_subset_size]
            h_indices = h_indices[perm]
            l_indices = l_indices[perm]

        return h_indices, l_indices

    def _update_carry(
        self,
        z_H: Tensor,
        z_L: Tensor,
        z_H_out: Tensor,
        z_L_out: Tensor,
        losses: Tensor,
        carry_count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update carry state based on chain-level carry_H and carry_L policies.

        Policies operate on chains:
        - top1: only winner chain's state carried, others frozen (keep current)
        - top2: top 2 chains' states carried, others frozen
        - copy_top1: all chains get winner's output copied
        - copy_top2: half get top1, half get top2 (randperm assignment)

        Args:
            z_H: Current H states [B, N, S, C]
            z_L: Current L states [B, N, S, C]
            z_H_out: Output H states [B, N, S, C]
            z_L_out: Output L states [B, N, S, C]
            losses: [B, N] per-chain losses
            carry_count: [B, N] current carry counts

        Returns:
            Updated (z_H, z_L, carry_count)

        """
        cfg = self.config
        B, N = z_H.shape[:2]
        device = z_H.device
        batch_idx = torch.arange(B, device=device)

        # Sort chains by loss (ascending = best first)
        sorted_idx = losses.argsort(dim=-1)  # [B, N]
        top1_idx = sorted_idx[:, 0]  # [B]
        top2_idx = sorted_idx[:, 1] if N > 1 else top1_idx  # [B]

        # Track which chains get updated (for carry_count)
        h_updated = torch.zeros(B, N, device=device, dtype=torch.bool)
        l_updated = torch.zeros(B, N, device=device, dtype=torch.bool)

        # Update z_H based on carry_H policy
        z_H_new = z_H.clone()
        if cfg.carry_H == "top1":
            z_H_new[batch_idx, top1_idx] = z_H_out[batch_idx, top1_idx].detach()
            h_updated[batch_idx, top1_idx] = True
        elif cfg.carry_H == "top2":
            z_H_new[batch_idx, top1_idx] = z_H_out[batch_idx, top1_idx].detach()
            z_H_new[batch_idx, top2_idx] = z_H_out[batch_idx, top2_idx].detach()
            h_updated[batch_idx, top1_idx] = True
            h_updated[batch_idx, top2_idx] = True
        elif cfg.carry_H == "copy_top1":
            winner_H = z_H_out[batch_idx, top1_idx].unsqueeze(1)  # [B, 1, S, C]
            z_H_new = winner_H.expand_as(z_H).contiguous().detach()
            h_updated[:] = True
        elif cfg.carry_H == "copy_top2":
            # Top1 gets winner, top2 gets runner, rest randomly assigned
            winner_H = z_H_out[batch_idx, top1_idx].unsqueeze(1)  # [B, 1, S, C]
            runner_H = z_H_out[batch_idx, top2_idx].unsqueeze(1)  # [B, 1, S, C]
            # Fully vectorized random assignment (excluding top1/top2)
            perm = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # [B, N]
            half = N // 2
            use_winner = (perm < half).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
            z_H_new = torch.where(
                use_winner,
                winner_H.expand_as(z_H),
                runner_H.expand_as(z_H),
            ).detach()
            # Ensure top1 gets winner, top2 gets runner (override random)
            z_H_new[batch_idx, top1_idx] = z_H_out[batch_idx, top1_idx].detach()
            z_H_new[batch_idx, top2_idx] = z_H_out[batch_idx, top2_idx].detach()
            h_updated[:] = True
        elif cfg.carry_H == "all":
            z_H_new = z_H_out.detach()
            h_updated[:] = True
        elif cfg.carry_H != "none":
            raise ValueError(f"Unsupported carry_H policy: {cfg.carry_H}")

        # Update z_L based on carry_L policy
        z_L_new = z_L.clone()
        if cfg.carry_L == "top1":
            z_L_new[batch_idx, top1_idx] = z_L_out[batch_idx, top1_idx].detach()
            l_updated[batch_idx, top1_idx] = True
        elif cfg.carry_L == "top2":
            z_L_new[batch_idx, top1_idx] = z_L_out[batch_idx, top1_idx].detach()
            z_L_new[batch_idx, top2_idx] = z_L_out[batch_idx, top2_idx].detach()
            l_updated[batch_idx, top1_idx] = True
            l_updated[batch_idx, top2_idx] = True
        elif cfg.carry_L == "copy_top1":
            winner_L = z_L_out[batch_idx, top1_idx].unsqueeze(1)
            z_L_new = winner_L.expand_as(z_L).contiguous().detach()
            l_updated[:] = True
        elif cfg.carry_L == "copy_top2":
            # Top1 gets winner, top2 gets runner, rest randomly assigned
            winner_L = z_L_out[batch_idx, top1_idx].unsqueeze(1)  # [B, 1, S, C]
            runner_L = z_L_out[batch_idx, top2_idx].unsqueeze(1)  # [B, 1, S, C]
            # Fully vectorized random assignment (excluding top1/top2)
            perm = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # [B, N]
            half = N // 2
            use_winner = (perm < half).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
            z_L_new = torch.where(
                use_winner,
                winner_L.expand_as(z_L),
                runner_L.expand_as(z_L),
            ).detach()
            # Ensure top1 gets winner, top2 gets runner (override random)
            z_L_new[batch_idx, top1_idx] = z_L_out[batch_idx, top1_idx].detach()
            z_L_new[batch_idx, top2_idx] = z_L_out[batch_idx, top2_idx].detach()
            l_updated[:] = True
        elif cfg.carry_L == "all":
            z_L_new = z_L_out.detach()
            l_updated[:] = True
        elif cfg.carry_L != "none":
            raise ValueError(f"Unsupported carry_L policy: {cfg.carry_L}")

        # Update carry_count: increment where chain was updated (H or L)
        updated = (h_updated | l_updated).long()
        carry_count_new = carry_count + updated

        return z_H_new, z_L_new, carry_count_new

    def _forward_simple(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Standard forward: H_cycles-1 under no_grad, final with grad."""
        cfg = self.config
        input_emb = self.embed_scale * self.embed_tokens(input_ids, cfg.dtype)
        input_emb = self._prepend_prefix(input_emb, puzzle_ids)

        cos_sin = self._get_cos_sin(input_emb.device)
        core = self.core_compiled if cfg.compile_core else self.core

        H_cycles = cfg.H_cycles
        all_logits = []
        all_z_H = []

        # Detach cos_sin for no_grad iterations to avoid retaining the
        # gradient graph when inv_freqs is a learnable Parameter.
        cos_sin_detach = (
            (cos_sin[0].detach(), cos_sin[1].detach()) if cos_sin is not None else None
        )
        with torch.no_grad():
            for _ in range(H_cycles - 1):
                logits, q_halt, z_H, z_L = core(
                    input_emb.detach(),
                    z_H.detach(),
                    z_L.detach(),
                    cos_sin_detach,
                )
                all_logits.append(logits)
                all_z_H.append(z_H)

        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, cos_sin)
        all_logits.append(logits.detach())
        all_z_H.append(z_H.detach())

        # Slice logits to exclude prefix positions (puzzle_id + register tokens).
        # z_H/z_L/all_z_H remain full total_seq_len - they're state vectors, not outputs.
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        if n_prefix > 0:
            logits = logits[:, n_prefix:]
            all_logits = [lg[:, n_prefix:] for lg in all_logits]

        return {
            "logits": logits,  # [B, num_puzzle_grid_tokens, V]
            "all_logits": all_logits,  # List[[B, num_puzzle_grid_tokens, V]]
            "q_halt": q_halt,  # [B]
            "z_H": z_H.detach(),  # [B, total_seq_len, C]
            "z_L": z_L.detach(),  # [B, total_seq_len, C]
            "all_z_H": all_z_H,  # List[[B, total_seq_len, C]]
        }

    def _forward_delta_reparam(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        puzzle_ids: Tensor | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Delta reparameterization: z = z_init + delta, gradients flow to init.

        Input z_H, z_L are [B*N, S, C] where N = num_effective_heads.
        Each group of N samples corresponds to chains with indices from
        _sample_head_indices(). We expand init vectors to match.
        """
        cfg = self.config
        if cfg.z_H_random_init or cfg.z_L_random_init:
            raise ValueError(
                "z_delta_reparam is incompatible with z_H_random_init/z_L_random_init. "
                "Delta reparam assumes z derives from H_init/L_init.",
            )
        input_emb = self.embed_scale * self.embed_tokens(input_ids, cfg.dtype)
        input_emb = self._prepend_prefix(input_emb, puzzle_ids)

        cos_sin = self._get_cos_sin(input_emb.device)
        core = self.core_compiled if cfg.compile_core else self.core

        H_cycles = cfg.H_cycles
        all_logits = []
        all_z_H = []

        # Detach cos_sin for no_grad iterations to avoid retaining the
        # gradient graph when inv_freqs is a learnable Parameter.
        cos_sin_detach = (
            (cos_sin[0].detach(), cos_sin[1].detach()) if cos_sin is not None else None
        )

        # z_H/z_L are [B*N, S, C], need to build matching init tensors
        BN, S, C = z_H.shape
        h_indices, l_indices = self._sample_head_indices()
        N = len(h_indices)
        if BN % N != 0:
            raise ValueError(
                f"Batch size {BN} not divisible by num_effective_heads {N}",
            )
        B = BN // N

        # Build init tensors: H_init[h_indices[n]] for each chain n, repeated B times
        # H_init is [K_H, C], expand to [N, S, C] then tile to [B*N, S, C]
        z_H_init_per_chain = self.H_init[h_indices].unsqueeze(1).expand(N, S, C)
        z_H_init = z_H_init_per_chain.unsqueeze(0).expand(B, N, S, C)
        z_H_init = z_H_init.reshape(BN, S, C)

        z_L_init_per_chain = self.L_init[l_indices].unsqueeze(1).expand(N, S, C)
        z_L_init = z_L_init_per_chain.unsqueeze(0).expand(B, N, S, C)
        z_L_init = z_L_init.reshape(BN, S, C)

        # Track delta = z - init throughout
        z_H_delta = z_H - z_H_init.detach()
        z_L_delta = z_L - z_L_init.detach()

        with torch.no_grad():
            for _ in range(H_cycles - 1):
                z_H_cur = z_H_init.detach() + z_H_delta
                z_L_cur = z_L_init.detach() + z_L_delta
                logits, q_halt, z_H_out, z_L_out = core(
                    input_emb.detach(),
                    z_H_cur.detach(),
                    z_L_cur.detach(),
                    cos_sin_detach,
                )
                z_H_delta = z_H_out - z_H_init.detach()
                z_L_delta = z_L_out - z_L_init.detach()
                all_logits.append(logits)
                all_z_H.append(z_H_out)

        # Final cycle: init has grad, delta detached
        z_H = z_H_init + z_H_delta
        z_L = z_L_init + z_L_delta
        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, cos_sin)
        all_logits.append(logits.detach())
        all_z_H.append(z_H.detach())

        # Slice logits to exclude prefix positions (puzzle_id + register tokens).
        # z_H/z_L/all_z_H remain full total_seq_len - they're state vectors, not outputs.
        n_prefix = cfg.num_puzzle_id_tokens + cfg.num_register_tokens
        if n_prefix > 0:
            logits = logits[:, n_prefix:]
            all_logits = [lg[:, n_prefix:] for lg in all_logits]

        return {
            "logits": logits,  # [B, num_puzzle_grid_tokens, V]
            "all_logits": all_logits,  # List[[B, num_puzzle_grid_tokens, V]]
            "q_halt": q_halt,  # [B]
            "z_H": z_H.detach(),  # [B, total_seq_len, C]
            "z_L": z_L.detach(),  # [B, total_seq_len, C]
            "all_z_H": all_z_H,  # List[[B, total_seq_len, C]]
        }

    # @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
    @torch.compile(mode="default", fullgraph=True)
    def core_compiled(
        self,
        input_emb: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if torch.compiler.is_compiling():
            _ = trace_compile(
                "core_compiled",
                max_compiles=self.config.max_num_compile_core,
                always_print=False,
            )
        return self.core(input_emb, z_H, z_L, cos_sin)

    def core(
        self,
        input_emb: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One H-cycle.

        Shapes: B = batch, S = total_seq_len, C = hidden_size.
        The K dimension is handled by caller (flatten to B*N before, unflatten after).

        Args:
            input_emb: [B, S, C]
            z_H: [B, S, C]
            z_L: [B, S, C]
            cos_sin: RoPE embeddings (cos, sin) or None

        Returns:
            logits: [B, S, vocab_size]
            q_halt: [B] (from z_H at q_halt_seq_index position)
            z_H: [B, S, C]
            z_L: [B, S, C]

        """
        L_cycles = self.config.L_cycles
        alpha = self.config.core_damping
        a = self.config.anchor_seq_index
        z_H_a = z_H[..., a, :].clone() if a is not None else z_H  # unused
        z_L_a = z_L[..., a, :].clone() if a is not None else z_L  # unused
        c = z_H + input_emb
        if alpha > 0:
            for _ in range(L_cycles):
                z_L = (1 - alpha) * z_L + alpha * self.reasoning(z_L + c, cos_sin)
                if a is not None:
                    z_L[..., a, :] = z_L_a
            z_H = (1 - alpha) * z_H + alpha * self.reasoning(z_H + z_L, cos_sin)
        else:
            for _ in range(L_cycles):
                z_L = self.reasoning(z_L + c, cos_sin)
                if a is not None:
                    z_L[..., a, :] = z_L_a
            z_H = self.reasoning(z_H + z_L, cos_sin)
        if a is not None:
            z_H[..., a, :] = z_H_a

        logits = self.head(z_H)
        q_halt = self.q_head(z_H[:, self.config.q_halt_seq_index]).squeeze(-1)

        return logits, q_halt, z_H, z_L
