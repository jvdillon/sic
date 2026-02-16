"""Unit tests for model.py."""

from __future__ import annotations

from typing import Any

import functools
import pathlib

from model import (
    EMA,
    TRM,
    Attention,
    BatchNorm,
    Embedding,
    GroupNorm,
    Linear,
    MLPMixerBlock,
    RoPE,
    RoPEMixed,
    Sequential,
    SwiGLU,
    TransformerBlock,
    _find_multiple,  # pyright: ignore[reportPrivateUsage]
    trunc_normal_init_,
)
from util import set_seed

import pytest
import torch


class TestModelUtils:
    def test_trunc_normal_init(self):
        t = torch.empty(100, 100)
        trunc_normal_init_(t, std=0.5)
        assert t.mean().abs() < 0.1
        assert t.std() < 1.0

    def testtrunc_normal_init_zero_std(self):
        t = torch.empty(10, 10)
        trunc_normal_init_(t, std=0.0)
        assert (t == 0).all()

    def test_find_multiple(self):
        assert _find_multiple(10, 8) == 16
        assert _find_multiple(16, 8) == 16
        assert _find_multiple(17, 8) == 24
        assert _find_multiple(1, 256) == 256


class TestLinear:
    def test_forward(self):
        lin = Linear(64, 32, bias=False)
        x = torch.randn(2, 10, 64)
        y = lin(x)
        assert y.shape == (2, 10, 32)

    def test_with_bias(self):
        lin = Linear(64, 32, bias=True)
        assert lin.bias is not None
        x = torch.randn(2, 10, 64)
        y = lin(x)
        assert y.shape == (2, 10, 32)

    def test_dtype_cast(self):
        lin = Linear(32, 16, bias=False)
        x = torch.randn(2, 5, 32, dtype=torch.float16)
        y = lin(x)
        assert y.dtype == torch.float16


class TestEmbedding:
    def test_forward(self):
        emb = Embedding(100, 64)
        x = torch.randint(low=0, high=100, size=(2, 10))
        y = emb(x, torch.float32)
        assert y.shape == (2, 10, 64)
        assert y.dtype == torch.float32

    def test_dtype_cast(self):
        emb = Embedding(50, 32)
        x = torch.randint(low=0, high=50, size=(2, 5))
        y = emb(x, torch.float16)
        assert y.dtype == torch.float16


class TestRoPE:
    # --- Shape tests -----------------------------------------------------

    @pytest.mark.parametrize(
        ("kwargs", "pos_shape", "expected_shape"),
        [
            # 1D.
            ({"dim": 32}, (128,), (128, 1, 16)),
            ({"dim": 16}, (64,), (64, 1, 8)),
            # 2D axial (iterable dim).
            ({"dim": [64, 64]}, (16, 2), (16, 1, 64)),
            # 3D axial (explicit per-axis dims).
            ({"dim": [44, 42, 42]}, (8, 3), (8, 1, 64)),
            # 3D axial (scalar dim replicated).
            ({"dim": 128, "base": [100.0, 100.0, 100.0]}, (8, 3), (8, 1, 192)),
            # Batched positions.
            ({"dim": 32}, (4, 10, 1), (4, 10, 1, 16)),
        ],
    )
    def test_shape(
        self,
        kwargs: dict[str, Any],
        pos_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ):
        rope = RoPE(**kwargs)
        positions = torch.arange(torch.tensor(pos_shape).prod().item()).float()
        positions = positions.reshape(pos_shape)
        cos, sin = rope(positions)
        assert cos.shape == expected_shape
        assert sin.shape == expected_shape

    # --- Value tests -----------------------------------------------------

    def test_values_bounded(self):
        rope = RoPE(dim=16, base=10_000.0)
        cos, sin = rope(torch.arange(64))
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0

    def test_pythagorean_identity(self):
        rope = RoPE(dim=32)
        cos, sin = rope(torch.arange(20))
        torch.testing.assert_close(cos**2 + sin**2, torch.ones_like(cos))

    def test_zero_position_is_identity(self):
        rope = RoPE(dim=32, base=10_000.0)
        cos, sin = rope(torch.zeros(1))
        torch.testing.assert_close(cos, torch.ones_like(cos))
        torch.testing.assert_close(sin, torch.zeros_like(sin))

    # --- Dim replication -------------------------------------------------

    def test_scalar_dim_replicated_by_bases(self):
        rope = RoPE(dim=128, base=[10_000.0, 10_000.0, 10_000.0])
        assert rope.dim == (128, 128, 128)

    def test_scalar_dim_replicated_2(self):
        rope = RoPE(dim=128, base=[10_000.0, 10_000.0])
        assert rope.dim == (128, 128)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            RoPE(dim=3, base=[10_000.0, 10_000.0])

    # --- Block-diagonal structure ----------------------------------------

    def test_axial_per_axis_freqs(self):
        rope = RoPE(dim=[16, 16])
        inv = rope._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert inv[0].numel() == 8
        assert inv[1].numel() == 8

    # --- dtype preservation through .to() --------------------------------

    def test_no_grad(self):
        rope = RoPE(dim=32)
        inv = rope._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert all(not f.requires_grad for f in inv if f.numel())

    def test_dtype_preserved_after_cast(self):
        rope = RoPE(dim=32)
        rope = rope.to(torch.bfloat16)
        inv = rope._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert all(f.dtype == torch.float32 for f in inv if f.numel())

    def test_output_dtype_matches_model(self):
        rope = RoPE(dim=32).to(torch.bfloat16)
        cos, _sin = rope(torch.arange(10))
        assert cos.dtype == torch.bfloat16

    # --- different bases -------------------------------------------------

    def test_different_base_changes_freqs(self):
        r1 = RoPE(dim=32, base=100.0)
        r2 = RoPE(dim=32, base=10_000.0)
        inv1 = r1._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        inv2 = r2._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert not torch.equal(inv1[0], inv2[0])


class TestRoPEMixed:
    # --- Shape tests -----------------------------------------------------

    @pytest.mark.parametrize(
        ("kwargs", "pos_shape", "expected_shape"),
        [
            # 1D multi-head axial.
            ({"dim": 32, "num_heads": 4}, (10,), (10, 4, 16)),
            # 1D learnable axial.
            ({"dim": 32, "num_heads": 4, "learnable": True}, (10,), (10, 4, 16)),
            # 2D multi-head axial.
            (
                {"dim": [64, 64], "num_heads": 4, "reduction_mode": "cat"},
                (16, 2),
                (16, 4, 64),
            ),
            # 2D sum (non-axial): dim=64 replicated, each axis has 32.
            (
                {
                    "dim": 64,
                    "num_heads": 8,
                    "base": [100.0, 100.0],
                    "reduction_mode": "sum",
                },
                (16, 2),
                (16, 8, 32),
            ),
            # 2D learnable sum (RoPE-Mixed).
            (
                {
                    "dim": 64,
                    "num_heads": 8,
                    "base": [100.0, 100.0],
                    "reduction_mode": "sum",
                    "learnable": True,
                },
                (16, 2),
                (16, 8, 32),
            ),
            # 3D multi-head axial (explicit per-axis dims).
            (
                {
                    "dim": [44, 42, 42],
                    "num_heads": 4,
                    "reduction_mode": "cat",
                },
                (8, 3),
                (8, 4, 64),
            ),
        ],
    )
    def test_shape(
        self,
        kwargs: dict[str, Any],
        pos_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ):
        rope = RoPEMixed(**kwargs)
        positions = torch.arange(torch.tensor(pos_shape).prod().item()).float()
        positions = positions.reshape(pos_shape)
        cos, sin = rope(positions)
        assert cos.shape == expected_shape
        assert sin.shape == expected_shape

    # --- Value tests -----------------------------------------------------

    def test_values_bounded(self):
        rope = RoPEMixed(dim=16, num_heads=4, base=10_000.0)
        cos, sin = rope(torch.arange(64))
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0

    def test_pythagorean_identity(self):
        rope = RoPEMixed(dim=32, num_heads=4, learnable=True)
        cos, sin = rope(torch.arange(20))
        torch.testing.assert_close(cos**2 + sin**2, torch.ones_like(cos))

    # --- Sum: dense structure --------------------------------------------

    def test_sum_dense(self):
        rope = RoPEMixed(dim=16, num_heads=4, base=[100.0, 100.0], reduction_mode="sum")
        inv = rope._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        active = [f for f in inv if f.numel()]
        assert len(active) == 2  # 2 axes
        assert all(f.shape[0] == 4 for f in active)  # 4 heads

    def test_axial_false_unequal_dims_raises(self):
        with pytest.raises(ValueError, match="uniform"):
            RoPEMixed(dim=[44, 42], num_heads=4, reduction_mode="sum")

    # --- learnable / requires_grad ---------------------------------------

    def test_learnable_has_grad(self):
        rope = RoPEMixed(dim=32, num_heads=4, learnable=True)
        inv = rope._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert all(f.requires_grad for f in inv if f.numel())

    def test_fixed_no_grad(self):
        rope = RoPEMixed(dim=32, num_heads=4, learnable=False)
        inv = rope._inv_freqs  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        assert all(not f.requires_grad for f in inv if f.numel())

    # --- dtype preservation through .to() --------------------------------

    def test_output_f32_after_cast(self):
        """Forward computes in float32 regardless of parameter dtype."""
        rope = RoPEMixed(dim=32, num_heads=4, learnable=True).to(torch.bfloat16)
        cos, sin = rope(torch.arange(10).unsqueeze(-1))
        assert cos.dtype == torch.bfloat16
        assert sin.dtype == torch.bfloat16

    def test_output_dtype_matches_model(self):
        rope = RoPEMixed(dim=32, num_heads=4).to(torch.bfloat16)
        cos, _sin = rope(torch.arange(10))
        assert cos.dtype == torch.bfloat16


class TestAttention:
    def test_forward_no_rope(self):
        attn = Attention(64, num_heads=4)
        x = torch.randn(2, 10, 64)
        y = attn(x, cos_sin=None)
        assert y.shape == (2, 10, 64)

    def test_forward_with_rope(self):
        attn = Attention(64, num_heads=4)
        rope = RoPE(dim=16, base=10000.0)
        x = torch.randn(2, 10, 64)
        cos, sin = rope(torch.arange(10))
        y = attn(x, cos_sin=(cos, sin))
        assert y.shape == (2, 10, 64)

    def test_causal(self):
        attn = Attention(64, num_heads=4, causal=True)
        x = torch.randn(2, 10, 64)
        y = attn(x, cos_sin=None)
        assert y.shape == (2, 10, 64)


class TestSwiGLU:
    def test_forward_with_gate(self):
        mlp = SwiGLU(64, expansion=4.0, gate=True)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == (2, 10, 64)

    def test_forward_no_gate(self):
        mlp = SwiGLU(64, expansion=4.0, gate=False)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == (2, 10, 64)

    def test_muon_modified_mode(self):
        mlp = SwiGLU(64, expansion=4.0, gate=True, muon_modified=True)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == (2, 10, 64)
        assert mlp.norm is not None

    def test_non_muon_modified_mode(self):
        mlp = SwiGLU(64, expansion=4.0, gate=True, muon_modified=False)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == (2, 10, 64)
        assert mlp.norm is None


class TestTransformerBlock:
    def test_forward(self):
        block = TransformerBlock(64, num_heads=4)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_with_rope(self):
        block = TransformerBlock(64, num_heads=4)
        rope = RoPE(dim=16, base=10000.0)
        x = torch.randn(2, 10, 64)
        cos, sin = rope(torch.arange(10))
        y = block(x, cos_sin=(cos, sin))
        assert y.shape == (2, 10, 64)


class TestMLPMixerBlock:
    def test_forward(self):
        block = MLPMixerBlock(64, seq_len=10)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_muon_modified_mode(self):
        block = MLPMixerBlock(64, seq_len=10, muon_modified=True)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_rope_raises(self):
        block = MLPMixerBlock(64, seq_len=10)
        x = torch.randn(2, 10, 64)
        with pytest.raises(NotImplementedError):
            block(x, cos_sin=(torch.randn(10, 16), torch.randn(10, 16)))


class TestNormLayers:
    def test_batch_norm(self):
        bn = BatchNorm(64)
        bn.train()
        x = torch.randn(4, 10, 64)
        y = bn(x)
        assert y.shape == (4, 10, 64)

    def test_group_norm(self):
        gn = GroupNorm(64, num_groups=8)
        x = torch.randn(2, 10, 64)
        y = gn(x)
        assert y.shape == (2, 10, 64)


class TestSequential:
    def test_forward_with_kwargs(self):
        blocks = Sequential(
            MLPMixerBlock(64, seq_len=10),
            MLPMixerBlock(64, seq_len=10),
        )
        x = torch.randn(2, 10, 64)
        y = blocks(x, cos_sin=None)
        assert y.shape == (2, 10, 64)


class TestEMA:
    def test_init(self):
        model = torch.nn.Linear(10, 5)
        ema = EMA(model, decay=0.9)
        assert len(ema.shadow) == 2

    def test_update(self):
        model = torch.nn.Linear(10, 5)
        ema = EMA(model, decay=0.9)
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        for p in model.parameters():
            p.data.add_(1.0)

        ema.update(model)

        for n, p in model.named_parameters():
            if p.requires_grad:
                assert not torch.equal(ema.shadow[n], original[n])
                assert not torch.equal(ema.shadow[n], p.data)

    def test_apply_restore(self):
        model = torch.nn.Linear(10, 5)
        ema = EMA(model, decay=0.9)

        for p in model.parameters():
            p.data.add_(1.0)
        ema.update(model)
        modified = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply(model)
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert torch.equal(p.data, ema.shadow[n])

        ema.restore(model)
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert torch.equal(p.data, modified[n])


class TestTRMConfig:
    def test_n_iters(self):
        config = TRM.Config()
        assert config.n_iters == 6 * (9 + 1)


class TestTRM:
    @pytest.fixture
    def config(self) -> TRM.Config:
        return TRM.Config(
            hidden_size=64,
            num_heads=4,
            num_layers=1,
            H_cycles=2,
            L_cycles=2,
            act=False,
            compile_core=False,
            seq_len=82,
            block_fn=functools.partial(MLPMixerBlock, seq_len=82),
        )

    @pytest.fixture
    def act_config(self) -> TRM.Config:
        return TRM.Config(
            hidden_size=64,
            num_heads=4,
            num_layers=1,
            H_cycles=2,
            L_cycles=2,
            compile_core=False,
            seq_len=82,
            block_fn=functools.partial(MLPMixerBlock, seq_len=82),
        )

    def test_forward_basic(self, config: TRM.Config):
        model = config.setup()
        x = torch.randint(low=0, high=12, size=(2, 82))
        z_H = model.H_init.expand(2, 82, -1)
        z_L = model.L_init.expand(2, 82, -1)
        out = model(x, z_H, z_L)
        assert out["logits"].shape == (2, 82, 12)

    def test_forward_with_z_states(self, act_config: TRM.Config):
        model = act_config.setup()
        x = torch.randint(low=0, high=12, size=(2, 82))
        z_H = model.H_init.expand(2, 82, -1)
        z_L = model.L_init.expand(2, 82, -1)
        out = model(x, z_H, z_L)
        assert "logits" in out
        assert "q_halt" in out
        assert "z_H" in out
        assert "z_L" in out
        assert out["logits"].shape == (2, 82, 12)
        assert out["q_halt"].shape == (2,)

    def test_all_logits(self, config: TRM.Config):
        model = config.setup()
        x = torch.randint(low=0, high=12, size=(2, 82))
        z_H = model.H_init.expand(2, 82, -1)
        z_L = model.L_init.expand(2, 82, -1)
        out = model(x, z_H, z_L)
        assert "all_logits" in out
        assert len(out["all_logits"]) == config.H_cycles

    def test_all_z_h(self, config: TRM.Config):
        model = config.setup()
        x = torch.randint(low=0, high=12, size=(2, 82))
        z_H = model.H_init.expand(2, 82, -1)
        z_L = model.L_init.expand(2, 82, -1)
        out = model(x, z_H, z_L)
        assert "all_z_H" in out
        assert len(out["all_z_H"]) == config.H_cycles
        assert out["all_z_H"][0].shape == (2, 82, config.hidden_size)

    def test_state_noise_training(self, config: TRM.Config):
        config.state_noise = 0.1
        model = config.setup()
        model.train()
        x = torch.randint(low=0, high=12, size=(2, 82))
        z_H = model.H_init.expand(2, 82, -1)
        z_L = model.L_init.expand(2, 82, -1)

        torch.manual_seed(42)
        out1 = model(x, z_H, z_L)
        torch.manual_seed(42)
        out2 = model(x, z_H, z_L)

        assert torch.equal(out1["logits"], out2["logits"])

    def test_no_block_fn_raises(self):
        config = TRM.Config(
            hidden_size=64,
            num_heads=4,
            num_layers=1,
            H_cycles=2,
            L_cycles=2,
            block_fn=None,  # pyright: ignore[reportArgumentType]
        )
        with pytest.raises(ValueError, match="block_fn is required"):
            config.setup()

    def test_step(self, act_config: TRM.Config):
        """Test step() for single H-cycle."""
        model = act_config.setup()
        x = torch.randint(low=0, high=12, size=(2, 82))
        z_H = model.H_init.expand(2, 82, -1)
        z_L = model.L_init.expand(2, 82, -1)
        out = model.step(x, z_H, z_L)
        assert out["logits"].shape == (2, 82, 12)
        assert out["q_halt"].shape == (2,)
        assert out["z_H"].shape == (2, 82, 64)
        assert out["z_L"].shape == (2, 82, 64)


def _trm_bitforbit_config() -> TRM.Config:
    return TRM.Config(
        hidden_size=4,
        num_heads=1,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
        act=False,
        compile_core=False,
        dtype=torch.float32,
        seq_len=82,
        block_fn=functools.partial(MLPMixerBlock, seq_len=82),
    )


def _trm_bitforbit_inputs() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 8 + [0, 1]])


def regenerate_trm_checkpoint() -> None:
    """Regenerate test_data/trm3.pt. Call when model init changes."""
    set_seed(42)
    config = _trm_bitforbit_config()
    model = config.setup()
    model.eval()

    inputs = _trm_bitforbit_inputs()
    z_H = model.H_init.expand(1, 82, -1).clone()
    z_L = model.L_init.expand(1, 82, -1).clone()

    with torch.no_grad():
        out = model(inputs, z_H, z_L)

    ckpt = {
        "state_dict": model.state_dict(),
        "expected_logits": out["logits"],
    }
    ckpt_path = pathlib.Path(__file__).resolve().parent / "test_data" / "trm3.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"Saved {ckpt_path}")


class TestTRMBitForBit:
    """Bit-for-bit reproducibility tests for TRM."""

    @pytest.fixture
    def checkpoint(self) -> dict[str, Any]:
        ckpt_path = pathlib.Path(__file__).resolve().parent / "test_data" / "trm3.pt"
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")
        return torch.load(ckpt_path, weights_only=False)

    def test_forward_bitforbit(self, checkpoint: dict[str, Any]):
        """Forward pass produces exact same logits as checkpoint."""
        set_seed(42)
        config = _trm_bitforbit_config()
        model = config.setup()
        model.eval()

        inputs = _trm_bitforbit_inputs()
        z_H = model.H_init.expand(1, 82, -1).clone()
        z_L = model.L_init.expand(1, 82, -1).clone()

        with torch.no_grad():
            out = model(inputs, z_H, z_L)

        expected = checkpoint["expected_logits"]
        actual = out["logits"]
        assert torch.equal(actual, expected), (
            f"Logits diverged: max_diff={(actual - expected).abs().max().item():.6e}"
        )

    def test_training_5_steps_bitforbit(self):
        """5 training steps produce exact same results."""
        config = _trm_bitforbit_config()
        inputs = _trm_bitforbit_inputs()
        labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0] * 8 + [1, 2]])

        # Run 1
        set_seed(42)
        model1 = config.setup()
        model1.train()
        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

        for _ in range(5):
            opt1.zero_grad()
            z_H = model1.H_init.expand(1, 82, -1).clone()
            z_L = model1.L_init.expand(1, 82, -1).clone()
            out = model1(inputs, z_H, z_L)
            loss = torch.nn.functional.cross_entropy(
                out["logits"].view(-1, 12),
                labels.view(-1),
            )
            loss.backward()
            opt1.step()

        # Run 2
        set_seed(42)
        model2 = config.setup()
        model2.train()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        for _ in range(5):
            opt2.zero_grad()
            z_H = model2.H_init.expand(1, 82, -1).clone()
            z_L = model2.L_init.expand(1, 82, -1).clone()
            out = model2(inputs, z_H, z_L)
            loss = torch.nn.functional.cross_entropy(
                out["logits"].view(-1, 12),
                labels.view(-1),
            )
            loss.backward()
            opt2.step()

        # Compare all parameters
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(),
            model2.named_parameters(),
            strict=False,
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), (
                f"{n1} diverged: max_diff={(p1 - p2).abs().max().item():.6e}"
            )


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
