from collections.abc import Iterable
from typing import Any, Literal

import math

from torch import Tensor, nn

import torch


#############################################
#               Muon optimizer              #
#############################################


# https://kellerjordan.github.io/posts/muon/
# https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file
# https://github.com/KellerJordan/Muon?tab=readme-ov-file
# https://arxiv.org/pdf/2409.20325

# decomposition (unitary factor).
#
# For X = UΣV^T, this gives:
#   X(X^T X)^{-1/2}
#   = UΣV^T (VΣ^2V^T)^{-1/2}
#   = UΣV^T VΣ^{-1}V^T
#   = UV^T
#
#   msgn(X) = U sgn(Σ) V^T
#
#  Since singular valuesa are σᵢ ≥ 0 (by convention), sgn(σᵢ) should be:
#  - +1 if σᵢ > 0
#  - 0 if σᵢ = 0
#
#  So U sgn(Σ) V^T ≠ UV^T in general (rank-deficient case differs).

# https://docs.modula.systems/algorithms/newton-schulz/#a-cursed-quintic-iteration

# ~/research/.venv/lib/python3.12/site-packages/torch/optim/_muon.py


# @torch.compile(fullgraph=True, dynamic=True)
def matrix_signum_via_newtonschulz(
    X: Tensor,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """Newton-Schulz iteration for matrix signum. Expects 3D input [batch, c_out, c_in].

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of
    G. We opt to use a quintic iteration whose coefficients are selected to
    maximize the slope at zero. For the purpose of minimizing steps, it turns
    out to be empirically effective to keep increasing the slope at zero even
    beyond the point where the iteration no longer converges all the way to one
    everywhere on the interval. This iteration therefore does not produce UV^T
    but rather something like US'V^T where S' is diagonal with S_{ii}' ~
    Uniform(0.5, 1.5), which turns out not to hurt model performance at all
    relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    a, b, c = ns_coefficients
    X = X.bfloat16()
    X = X / X.norm(dim=[-2, -1], keepdim=True).clamp(min=eps)
    if transpose := X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
    for _ in range(ns_steps):
        A = X @ X.transpose(-1, -2)
        # B = b * A + c * A @ A
        # X = a * X + B @ X
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1)
    if transpose:
        X = X.transpose(-1, -2)
    return X


def _adjust_lr(
    lr: float,
    param: nn.Parameter,
    adjust_lr_fn: Literal[
        "original",
        "match_rms_adamw",
        "conv_heuristic",
        "keller",
        "other1",
        "other2",
    ] = "original",
    ensemble_dims: int = 0,
) -> float:
    """Learning rate adjustment used by Muon."""
    c_out = param.shape[ensemble_dims]
    c_in = math.prod(param.shape[ensemble_dims + 1 :])
    if adjust_lr_fn == "original":
        adjusted_ratio = max(1, c_out / c_in) ** 0.5
    elif adjust_lr_fn == "match_rms_adamw":
        # https://x.com/Kimi_Moonshot/status/1897929976948965870
        # https://github.com/MoonshotAI/Moonlight
        adjusted_ratio = 0.2 * max(c_out, c_in) ** 0.5
    elif adjust_lr_fn == "conv_heuristic":
        adjusted_ratio = param.data.norm() / c_out**0.5
    elif adjust_lr_fn == "other1":
        adjusted_ratio = c_in**0.5 / c_out**0.5
    elif adjust_lr_fn == "other2":
        adjusted_ratio = c_out**0.5 / c_in**0.5
    else:
        raise ValueError(
            f"Adjust learning rate function {adjust_lr_fn} is not supported",
        )
    return lr * adjusted_ratio


class Muon(torch.optim.Optimizer):
    """Optimizer that respects weight-scale invariance"""

    def __init__(
        self,
        # Type matches torch.optim.Optimizer signature; dict[str, Any] is forced by torch.
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        weight_decay: float | None = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: Literal[
            "original",
            "match_rms_adamw",
            "conv_heuristic",
            "keller",
            "other1",
            "other2",
        ] = "original",
        ensemble_dims: int = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "ensemble_dims": ensemble_dims,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            ensemble_dims = group["ensemble_dims"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if len(g.shape) < 2:
                    for p_ in group["params"]:
                        print(p_.shape, p_.dtype)
                    raise ValueError(
                        f"Muon is only defined for linear operators {p.shape}.",
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Keller had,
                #   buf.mul_(momentum).add_(g)
                #   g = g.add(buf, alpha=momentum) if nesterov else buf
                # Torch, does
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum) if nesterov else buf

                # Seems like we should be doing weight decay under the proximity norm?
                # if weight_decay is not None:
                #     g.mul_((1 + weight_decay * p.data.float()).to(g.dtype))

                msgn_g = matrix_signum_via_newtonschulz(
                    g.reshape(
                        math.prod(g.shape[:ensemble_dims]),
                        g.shape[ensemble_dims],
                        -1,
                    ),
                    ns_coefficients=ns_coefficients,
                    ns_steps=ns_steps,
                    eps=eps,
                ).reshape(g.shape)

                if weight_decay is not None:
                    p.data.mul_(1 - lr * weight_decay)

                if adjust_lr_fn == "keller":
                    p.data.mul_(
                        p.data.shape[0] ** 0.5 / p.data.norm(),
                    )  # RMS weight norm?
                    # Did well at calibrating the spectral norm to be approx 1.
                    # c_out, c_in, *r = msgn_g.shape
                    # p.data.mul_(math.prod(r) * (c_in / c_out)**0.5 / p.data.norm())
                    p.data.add_(msgn_g, alpha=-lr)
                else:
                    adjusted_lr = _adjust_lr(lr, p, adjust_lr_fn, ensemble_dims)
                    p.data.add_(msgn_g, alpha=-adjusted_lr)

                # torch
                #   p.data.mul_(1 - lr * weight_decay)
                #   alpha = -lr * sqrt(max(1, c_out / c_in))
                #   p.data.add_(msgn_g * alpha)
                # keller
                #   p.data.mul_(p.data.shape[0] ** 0.5 / p.data.norm())
                #   p.data.add_(msgn_g, alpha=-lr)
                # me
                #   p.data.mul_(1 - lr * weight_decay)
                #   alpha = -lr * p.data.norm() / p.data.shape[0] ** 0.5
                #   p.data.add_(msgn_g * alpha)

                # Keller Original:
                #   p.data.mul_(p.data.shape[0] ** 0.5 / p.data.norm())
                #   p.data.add_(msgn_g, alpha=-lr)
                #
                # I thought this was better:
                #   msgn_g *= -lr * p.data.norm() / p.data.shape[0] ** 0.5
                #   p.data.add_(msgn_g)
                #
                # Because:
                # Rather than scale msgn_g it was originally:
                #   p.data.mul_(p.data.shape[0] ** 0.5 / p.data.norm())
                #   p.data.add_(msgn_g, alpha=-lr)
                # But this doesn't make sense for three reasons:
                # 1. It is unstable when p.data.norm is small.
                # 2. It assumes you'll later be using something like batchnorm
                #    to undo the paramater rescaling.
                # 3. We need to scale the g by -lr anyway; why not just do one scaling.


class DummyOptimizer:
    def __init__(self) -> None:
        self.param_groups: list[dict[str, Any]] = []

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        pass
