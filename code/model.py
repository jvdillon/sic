"""Model building blocks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal, Protocol, Self, TypedDict

import dataclasses
import functools
import math
import traceback

from torch import Tensor, nn

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
    rope_theta: float
    head_bias: bool
    act: bool
    act_q_head_bias_init: float
    block_fn: Callable[[int], nn.Module]
    use_rope: bool
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
    """Config protocol for TRM3 models with K_H/K_L heads."""

    seq_len: int
    hidden_size: int
    vocab_size: int
    dtype: torch.dtype | None
    num_layers: int
    H_cycles: int
    L_cycles: int
    head_bias: bool
    block_fn: Callable[[int], nn.Module]
    compile_core: bool
    compile_reasoning: bool
    max_num_compile_core: int
    K_H: int
    K_L: int

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

    logits: Tensor  # [B, N, S-1, V] per-head logits (without HALT position)
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


def _normal_init_(tensor: Tensor, std: float | None = None) -> Tensor:
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


def _trunc_normal_init_(
    tensor: Tensor,
    *,
    std: float | None = None,
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


def _kaiming_uniform_init_(
    tensor: Tensor,
    *,
    c_in: int | None = None,
    bound: float | None = None,
) -> Tensor:
    if bound is None:
        if c_in is None:
            raise ValueError("_kaiming_uniform_init_ requires bound or c_in")
        bound = c_in**-0.5
    assert bound is not None
    nn.init.uniform_(tensor, -bound, bound)
    return tensor


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (_rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (_rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class Linear(nn.Module):
    """Linear with configurable init."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        bias: bool = True,
        init_weight_fn: InitFn = _normal_init_,
        init_bias_fn: InitFn = _kaiming_uniform_init_,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            init_weight_fn(
                torch.empty(
                    [c_out, c_in],
                    dtype=dtype,
                )
            )
        )
        if bias:
            # TODO(josh): Consider adding to this "if": "or has kwarg 'c_in'".
            if init_bias_fn is _kaiming_uniform_init_:
                init_bias_fn = functools.partial(_kaiming_uniform_init_, c_in=c_in)
            self.bias = nn.Parameter(
                init_bias_fn(
                    torch.empty(
                        c_out,
                        dtype=dtype,
                    )
                )
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        b = None if self.bias is None else self.bias.to(x.dtype)
        return nn.functional.linear(x, w, b)


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
        init_weight_fn: InitFn = _normal_init_,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            init_weight_fn(torch.empty([num_ensemble, c_out, c_in]))
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    num_ensemble,
                    c_out,
                )
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
        init_weight_fn: InitFn = _trunc_normal_init_,  # std = rsqrt(c_in)
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            init_weight_fn(
                torch.empty(
                    [num_embeddings, embedding_dim],
                    dtype=dtype,
                )
            )
        )

    def forward(self, x: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        return nn.functional.embedding(x, self.weight.to(dtype))


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self) -> tuple[Tensor, Tensor]:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head attention with fused QKV EnsembleLinear for Muon orthogonalization."""

    def __init__(
        self,
        c_in: int,
        *,
        num_heads: int,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.head_dim = c_in // num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # Fused QKV: one kernel, each head orthogonalized independently by Muon
        # Ensemble dim = num_heads + 2 * num_key_value_heads (Q heads + K heads + V heads)
        _init_fn = functools.partial(_normal_init_, std=0.02)
        self.qkv_proj = EnsembleLinear(
            c_in,
            self.head_dim,
            num_ensemble=num_heads + 2 * num_key_value_heads,
            init_weight_fn=_init_fn,
        )
        self.o_proj = Linear(
            c_in,
            c_in,
            bias=False,
            init_weight_fn=_init_fn,
        )

    def forward(
        self,
        x: Tensor,
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

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))  # S H D -> H S D
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.causal,
        )
        out = out.transpose(-3, -2).flatten(-2)  # H S D -> S (H D)
        return self.o_proj(out)


class AttentionSplitQKV(nn.Module):
    """Attention with separate QK and V projections for independent weight decay."""

    def __init__(
        self,
        c_in: int,
        *,
        num_heads: int,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.head_dim = c_in // num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # Separate QK from V for different weight decay
        _init_fn = functools.partial(_normal_init_, std=0.02)
        self.qk_proj = EnsembleLinear(
            c_in,
            self.head_dim,
            num_ensemble=num_heads + num_key_value_heads,
            init_weight_fn=_init_fn,
        )
        self.v_proj = EnsembleLinear(
            c_in,
            self.head_dim,
            num_ensemble=num_key_value_heads,
            init_weight_fn=_init_fn,
        )
        self.o_proj = Linear(
            c_in,
            c_in,
            bias=False,
            init_weight_fn=_init_fn,
        )

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        q, k = self.qk_proj(x).split(
            [self.num_heads, self.num_key_value_heads],
            dim=-2,
        )
        v = self.v_proj(x)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))  # S H D -> H S D
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.causal,
        )
        out = out.transpose(-3, -2).flatten(-2)  # H S D -> S (H D)
        return self.o_proj(out)


class AttentionWithEntropy(nn.Module):
    """Attention that computes entropy of attention weights for regularization."""

    def __init__(
        self,
        c_in: int,
        *,
        num_heads: int,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.head_dim = c_in // num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.scale = self.head_dim**-0.5

        _init_fn = functools.partial(_normal_init_, std=0.02)
        self.qkv_proj = EnsembleLinear(
            c_in,
            self.head_dim,
            num_ensemble=num_heads + 2 * num_key_value_heads,
            init_weight_fn=_init_fn,
        )
        self.o_proj = Linear(c_in, c_in, bias=False, init_weight_fn=_init_fn)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        q, k, v = self.qkv_proj(x).split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
            dim=-2,
        )

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = (t.transpose(-2, -3) for t in (q, k, v))  # S H D -> H S D

        # Manual attention to get entropy
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            S = logits.shape[-1]
            mask = torch.triu(torch.ones(S, S, device=logits.device), diagonal=1).bool()
            logits = logits.masked_fill(mask, float("-inf"))

        attn = logits.softmax(dim=-1)
        log_attn = logits - logits.logsumexp(dim=-1, keepdim=True)

        # Entropy: -sum(p * log(p)) per query position, averaged over batch/heads/positions
        entropy = -(attn * log_attn).sum(dim=-1).mean()

        out = torch.matmul(attn, v)
        out = out.transpose(-3, -2).flatten(-2)  # H S D -> S (H D)
        return self.o_proj(out), entropy


default_norm_fn = functools.partial(
    nn.RMSNorm,
    eps=1e-5,
    elementwise_affine=False,
)
default_act_fn = nn.functional.silu


class SwiGLU(nn.Module):
    """SwiGLU with optional weird mode (norm before down_proj for Muon).

    weird=True: sigmoid(gate) * norm(gate * x)
    weird=False: silu(gate) * x
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
        weird: bool = True,
        init_weight_fn: InitFn = _normal_init_,
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
        if weird:
            if act_fn is not nn.functional.silu:
                raise NotImplementedError
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
                x = torch.sigmoid(gate) * self.norm(gate * x)
        else:
            x = self.act(x)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP. Uses post-norm."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        # Attention specific kwargs.
        num_heads: int = 0,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        assert num_heads > 0
        super().__init__()
        self.attn = Attention(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
        )
        self.norm1 = norm_fn(c_in)
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.attn(x, cos_sin=cos_sin)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class TransformerBlockScaled(nn.Module):
    """Transformer block with scaled post-norm to prevent L-cycle contraction."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        num_heads: int = 0,
        num_key_value_heads: int | None = None,
        causal: bool = False,
        norm_scale: float = 3.0,
    ):
        assert num_heads > 0
        super().__init__()
        self.attn = Attention(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
        )
        self.norm1 = norm_fn(c_in)
        self.norm2 = norm_fn(c_in)
        self.scale1 = nn.Parameter(torch.tensor(norm_scale))
        self.scale2 = nn.Parameter(torch.tensor(norm_scale))

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.attn(x, cos_sin=cos_sin)
        x = self.norm1(x) * self.scale1
        x = x + self.mlp(x)
        x = self.norm2(x) * self.scale2
        return x


class TransformerBlockSeqNorm(nn.Module):
    """Transformer block with seq_len norm (like MLPMixer) to prevent L-cycle contraction."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        num_heads: int = 0,
        num_key_value_heads: int | None = None,
        causal: bool = False,
        seq_len: int = 82,
    ):
        assert num_heads > 0
        super().__init__()
        self.attn = Attention(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
        )
        self.norm1 = norm_fn(seq_len)  # seq_len norm like MLPMixer
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.attn(x, cos_sin=cos_sin)
        x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)  # norm on S dim
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class TransformerBlockPreNorm(nn.Module):
    """Transformer block with pre-norm (norm before attention/MLP, not after)."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        num_heads: int = 0,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        assert num_heads > 0
        super().__init__()
        self.attn = Attention(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
        )
        self.norm1 = norm_fn(c_in)
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.attn(self.norm1(x), cos_sin=cos_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlockSplitQKV(nn.Module):
    """Transformer block with split QK/V projections for selective weight decay."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        num_heads: int = 0,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        assert num_heads > 0
        super().__init__()
        self.attn = AttentionSplitQKV(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
        )
        self.norm1 = norm_fn(c_in)
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.attn(x, cos_sin=cos_sin)
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


class TransformerBlockWithEntropy(nn.Module):
    """Transformer block that returns attention entropy for regularization."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        num_heads: int = 0,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        assert num_heads > 0
        super().__init__()
        self.attn = AttentionWithEntropy(
            c_in,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            causal=causal,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
        )
        self.norm1 = norm_fn(c_in)
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        attn_out, entropy = self.attn(x, cos_sin=cos_sin)
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x, entropy


class MLPMixerBlock(nn.Module):
    """MLP-Mixer block with token-mixing and channel-mixing. Uses post-norm."""

    def __init__(
        self,
        c_in: int,
        *,
        expansion: float = 4.0,
        multiple_of: int = 256,
        norm_fn: Callable[[int], nn.Module] = default_norm_fn,
        act_fn: Callable[[Tensor], Tensor] = default_act_fn,
        weird: bool = True,
        gate: bool = True,
        # MLPMixer specific kwargs.
        seq_len: int = 0,
        token_multiple_of: int | None = None,
        init_weight_fn: InitFn = _normal_init_,
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
            weird=weird,
            init_weight_fn=init_weight_fn,
        )
        self.mlp = SwiGLU(
            c_in,
            expansion=expansion,
            multiple_of=multiple_of,
            norm_fn=norm_fn,
            act_fn=act_fn,
            gate=gate,
            weird=weird,
            init_weight_fn=init_weight_fn,
        )
        self.norm1 = norm_fn(seq_len)
        self.norm2 = norm_fn(c_in)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if cos_sin is not None:
            raise NotImplementedError("MLPMixerBlock does not support RoPE")
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

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(*x.shape)


class GroupNorm(nn.GroupNorm):
    """GroupNorm for (B, L, C) input."""

    def __init__(self, num_features: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__(num_groups, num_features, eps=eps, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(-2, -1)).transpose(-1, -2)


class Sequential(nn.Sequential):
    """Sequential that passes args and kwargs to blocks."""

    def forward(self, z: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        for block in self:
            z = block(z, *args, **kwargs)
        return z


class SequentialWithEntropy(nn.Sequential):
    """Sequential that accumulates entropy from TransformerBlockWithEntropy."""

    def forward(self, z: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        total_entropy = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        for block in self:
            z, entropy = block(z, *args, **kwargs)
            total_entropy = total_entropy + entropy
        return z, total_entropy


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
        seq_len: int = 9 * 9 + 1
        hidden_size: int = 512
        num_heads: int = 8
        num_layers: int = 2
        H_cycles: int = 6
        L_cycles: int = 9
        state_noise: float = 0.0

        rope_theta: float = 10e3
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
        head_init_weight_fn: InitFn = _normal_init_

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
                _trunc_normal_init_,
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
            self.rope = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len,
                base=config.rope_theta,
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
            _ = _normal_init_(torch.empty_like(self.q_head.weight))
        else:
            self.q_head = None

        if config.block_fn is None:
            raise ValueError("block_fn is required")
        self.reasoning = Sequential(
            *[config.block_fn(config.hidden_size) for _ in range(config.num_layers)]
        )
        if config.compile_reasoning:
            # Doing as a side-effect means checkpoints are unaffected.
            self.reasoning.compile(mode="default", fullgraph=True)

        self.H_init: Tensor = nn.Buffer(
            _trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=config.dtype),
                std=1,
            ),
            persistent=True,
        )
        self.L_init: Tensor = nn.Buffer(
            _trunc_normal_init_(
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
        cos_sin = None if self.rope is None else self.rope()
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
        cos_sin = None if self.rope is None else self.rope()

        all_logits = []
        all_z_H = []

        core = self.core_compiled if self.config.compile_core else self.core

        if self.config.no_grad_inner:
            # We ensure to detach all inputs to avoid graph recompilation with eval
            # (which does everything under no_grad).
            cos_sin_detach = (
                (cos_sin[0].detach(), cos_sin[1].detach())
                if cos_sin is not None
                else None
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
            seq_len=9 * 9 + 1,  # 82
            init_weight_fn=_trunc_normal_init_,
        )
        head_init_weight_fn: InitFn = _trunc_normal_init_

        def setup(self, *args: Any, **kwargs: Any) -> TRM2:
            return TRM2(self, *args, **kwargs)


class TRM3(nn.Module):
    """Tiny Recursive Model."""

    @dataclasses.dataclass(slots=True, kw_only=True)
    class Config:
        vocab_size: int = 10 + 1 + 1  # values + unknown + halt
        seq_len: int = 9 * 9 + 1  # grid + halt
        hidden_size: int = 512
        num_layers: int = 2
        H_cycles: int = 6
        L_cycles: int = 9

        # I dont like this dtype default!
        dtype: torch.dtype | None = torch.bfloat16
        device: torch.device | str | None = "cuda"

        head_bias: bool = True
        block_fn: Callable[[int], nn.Module] = functools.partial(  # noqa: RUF009
            MLPMixerBlock,
            seq_len=seq_len,
            init_weight_fn=_trunc_normal_init_,
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
                _trunc_normal_init_,
                std=1.0 / self.embed_scale,
            ),
        )
        self.head = Linear(
            config.hidden_size,
            config.vocab_size,
            bias=config.head_bias,
            init_weight_fn=_trunc_normal_init_,  # std = rsqrt(hidden_size)
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
        _ = _normal_init_(torch.empty_like(self.q_head.weight))
        self.reasoning = Sequential(
            *[config.block_fn(config.hidden_size) for _ in range(config.num_layers)]
        )

        # PRNG_EQUIVALENCE: H_init and L_init: Only create first row here.
        # x177 creates L_init_1..K-1 AFTER .to() using CUDA RNG (torch.empty_like on GPU).
        # For RNG compat, x179.setup_model() must expand L_init the same way AFTER .to().
        # TRM3 creates [1, hidden] here; x179 expands to [K_L, hidden] in setup_model.
        # PRNG_EQUIVALENCE: Must use dtype=config.dtype because uniform_() produces
        # different values for bfloat16 vs float32 tensors with the same RNG seed.
        init_fn = functools.partial(_trunc_normal_init_, std=1)
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
        self.to(device=config.device, dtype=config.dtype)

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
        N = self.config.num_effective_heads
        S = self.config.seq_len

        # Get h_indices, l_indices to know how to pair init vectors
        h_indices, l_indices = self._sample_head_indices()

        init_fn = functools.partial(_trunc_normal_init_, std=1)

        # Initialize z_H for each chain
        if self.config.z_H_random_init:
            del h_indices
            z_H = init_fn(
                torch.empty(
                    [batch_size, N, S, self.config.hidden_size],
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            if self.config.z_H_init_svd:
                self._apply_svd_alignment_batched(z_H)
        else:
            # Each chain gets H_init[h_indices[chain_idx]]
            z_H_pool = self.H_init.unsqueeze(-2).expand(-1, S, -1)  # [K_H, S, C]
            z_H = z_H_pool[h_indices].unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Initialize z_L for each chain
        if self.config.z_L_random_init:
            del l_indices
            z_L = init_fn(
                torch.empty(
                    [batch_size, N, S, self.config.hidden_size],
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            if self.config.z_L_init_svd:
                self._apply_svd_alignment_batched(z_L)
        else:
            # Each chain gets L_init[l_indices[chain_idx]]
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
        label_smoothing: float = 0.0,
        z_L_noise: float = 0.0,
    ) -> WTAForwardOutput:
        """WTA forward pass with loss computation.

        Carry state has N chains, each with (z_H, z_L). We run all chains through
        the model and update based on carry_H/carry_L policies.
        """
        cfg = self.config
        B = input_ids.shape[0]
        seq_len = cfg.seq_len
        hidden = cfg.hidden_size

        # Unpack carry: [B, N, S, C] for each
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
            input_ids.unsqueeze(1).expand(B, N, seq_len).reshape(B * N, seq_len)
        )
        z_H_batched = z_H.reshape(B * N, seq_len, hidden)
        z_L_batched = z_L_active.reshape(B * N, seq_len, hidden)

        # Forward pass
        out = self.forward(inputs_batched, z_H_batched, z_L_batched)

        # Reshape back: [B*N, ...] -> [B, N, ...]
        logits_raw = out["logits"]
        q_halt_raw = out["q_halt"]
        z_H_out_raw = out["z_H"]
        z_L_out_raw = out["z_L"]
        assert isinstance(logits_raw, Tensor)
        assert isinstance(q_halt_raw, Tensor)
        assert isinstance(z_H_out_raw, Tensor)
        assert isinstance(z_L_out_raw, Tensor)

        logits_all = logits_raw[..., 1:, :].reshape(B, N, seq_len - 1, -1).float()
        q_halt_all = q_halt_raw.reshape(B, N).float()
        z_H_out = z_H_out_raw.reshape(B, N, seq_len, hidden)
        z_L_out = z_L_out_raw.reshape(B, N, seq_len, hidden)

        # Compute per-chain losses: [B, N]
        labels_flat = labels.reshape(B, -1)
        logits_flat = logits_all.reshape(B * N * (seq_len - 1), -1)
        labels_expanded = labels_flat.unsqueeze(1).expand(B, N, seq_len - 1).reshape(-1)
        per_cell_loss = nn.functional.cross_entropy(
            logits_flat,
            labels_expanded.long(),
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(B, N, seq_len - 1)
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

    def forward(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Multi H-cycle forward. Same shapes as step()."""
        if self.config.z_delta_reparam:
            return self._forward_delta_reparam(input_ids, z_H, z_L)
        return self._forward_simple(input_ids, z_H, z_L)

    def step(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Any]:
        """Single H-cycle step.

        Shapes: B = batch, S = seq_len, C = hidden_size, V = vocab_size.

        Args:
            input_ids: [B, S]
            z_H: [B, S, C]
            z_L: [B, S, C]

        Returns:
            dict: logits [B, S, V], q_halt [B], z_H [B, S, C], z_L [B, S, C]

        """
        input_emb = self.embed_scale * self.embed_tokens(input_ids, self.config.dtype)
        core = self.core_compiled if self.config.compile_core else self.core
        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, None)
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
                        "inner requires K_L_eff == K_H_eff or 1 in (K_L_eff, K_H_eff)"
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
        updated = (h_updated | l_updated).float()
        carry_count_new = carry_count + updated

        return z_H_new, z_L_new, carry_count_new

    def _forward_simple(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Standard forward: H_cycles-1 under no_grad, final with grad."""
        input_emb = self.embed_scale * self.embed_tokens(input_ids, self.config.dtype)
        core = self.core_compiled if self.config.compile_core else self.core

        H_cycles = self.config.H_cycles
        all_logits = []
        all_z_H = []

        with torch.no_grad():
            for _ in range(H_cycles - 1):
                logits, q_halt, z_H, z_L = core(
                    input_emb.detach(),
                    z_H.detach(),
                    z_L.detach(),
                    None,
                )
                all_logits.append(logits)
                all_z_H.append(z_H)

        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, None)
        all_logits.append(logits.detach())
        all_z_H.append(z_H.detach())

        return {
            "logits": logits,
            "all_logits": all_logits,
            "q_halt": q_halt,
            "z_H": z_H.detach(),
            "z_L": z_L.detach(),
            "all_z_H": all_z_H,
        }

    def _forward_delta_reparam(
        self,
        input_ids: Tensor,
        z_H: Tensor,
        z_L: Tensor,
    ) -> dict[str, Tensor | list[Tensor]]:
        """Delta reparameterization: z = z_init + delta, gradients flow to init.

        Input z_H, z_L are [B*N, S, C] where N = num_effective_heads.
        Each group of N samples corresponds to chains with indices from
        _sample_head_indices(). We expand init vectors to match.
        """
        cfg = self.config
        input_emb = self.embed_scale * self.embed_tokens(input_ids, cfg.dtype)
        core = self.core_compiled if cfg.compile_core else self.core

        H_cycles = cfg.H_cycles
        all_logits = []
        all_z_H = []

        # z_H/z_L are [B*N, S, C], need to build matching init tensors
        BN, seq_len, C = z_H.shape
        h_indices, l_indices = self._sample_head_indices()
        N = len(h_indices)
        if BN % N != 0:
            raise ValueError(
                f"Batch size {BN} not divisible by num_effective_heads {N}"
            )
        B = BN // N

        # Build init tensors: H_init[h_indices[n]] for each chain n, repeated B times
        # H_init is [K_H, C], expand to [N, S, C] then tile to [B*N, S, C]
        z_H_init_per_chain = self.H_init[h_indices].unsqueeze(1).expand(N, seq_len, C)
        z_H_init = z_H_init_per_chain.unsqueeze(0).expand(B, N, seq_len, C)
        z_H_init = z_H_init.reshape(BN, seq_len, C)

        z_L_init_per_chain = self.L_init[l_indices].unsqueeze(1).expand(N, seq_len, C)
        z_L_init = z_L_init_per_chain.unsqueeze(0).expand(B, N, seq_len, C)
        z_L_init = z_L_init.reshape(BN, seq_len, C)

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
                    None,
                )
                z_H_delta = z_H_out - z_H_init.detach()
                z_L_delta = z_L_out - z_L_init.detach()
                all_logits.append(logits)
                all_z_H.append(z_H_out)

        # Final cycle: init has grad, delta detached
        z_H = z_H_init + z_H_delta
        z_L = z_L_init + z_L_delta
        logits, q_halt, z_H, z_L = core(input_emb, z_H, z_L, None)
        all_logits.append(logits.detach())
        all_z_H.append(z_H.detach())

        return {
            "logits": logits,
            "all_logits": all_logits,
            "q_halt": q_halt,
            "z_H": z_H.detach(),
            "z_L": z_L.detach(),
            "all_z_H": all_z_H,
        }

    @torch.compile(mode="default", fullgraph=True)
    def core_compiled(
        self,
        input_emb: Tensor,
        z_H: Tensor,
        z_L: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,  # Unused; for TRM signature compat
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
        cos_sin: tuple[Tensor, Tensor] | None,  # Unused; for TRM signature compat
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One H-cycle.

        Shapes: B = batch, S = seq_len (82), C = hidden_size (512).
        The K dimension is handled by caller (flatten to B*N before, unflatten after).

        Args:
            input_emb: [B, S, C]
            z_H: [B, S, C]
            z_L: [B, S, C]
            cos_sin: Unused (for TRM signature compatibility)

        Returns:
            logits: [B, S, vocab_size]
            q_halt: [B] (from z_H[:, 0] i.e. HALT token position)
            z_H: [B, S, C]
            z_L: [B, S, C]

        """
        L_cycles = self.config.L_cycles
        c = z_H + input_emb
        for _ in range(L_cycles):
            z_L = self.reasoning(z_L + c, cos_sin)
        z_H = self.reasoning(z_H + z_L, cos_sin)

        logits = self.head(z_H)
        q_halt = self.q_head(z_H[:, 0]).squeeze(-1)

        return logits, q_halt, z_H, z_L
