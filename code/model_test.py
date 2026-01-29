"""Unit tests for model.py."""

from __future__ import annotations

from typing import Any

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
    RotaryEmbedding,
    Sequential,
    SwiGLU,
    TransformerBlock,
    _find_multiple,  # pyright: ignore[reportPrivateUsage]
    _trunc_normal_init_,  # pyright: ignore[reportPrivateUsage]
)
from util import set_seed

import pytest
import torch


class TestModelUtils:
    def test_trunc_normal_init(self):
        t = torch.empty(100, 100)
        _trunc_normal_init_(t, std=0.5)
        assert t.mean().abs() < 0.1
        assert t.std() < 1.0

    def test_trunc_normal_init_zero_std(self):
        t = torch.empty(10, 10)
        _trunc_normal_init_(t, std=0.0)
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


class TestRotaryEmbedding:
    def test_forward(self):
        rope = RotaryEmbedding(dim=32, max_position_embeddings=128, base=10000.0)
        cos, sin = rope()
        assert cos.shape == (128, 32)
        assert sin.shape == (128, 32)

    def test_values_bounded(self):
        rope = RotaryEmbedding(dim=16, max_position_embeddings=64, base=10000.0)
        cos, sin = rope()
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0


class TestAttention:
    def test_forward_no_rope(self):
        attn = Attention(64, num_heads=4)
        x = torch.randn(2, 10, 64)
        y = attn(x, cos_sin=None)
        assert y.shape == (2, 10, 64)

    def test_forward_with_rope(self):
        attn = Attention(64, num_heads=4)
        rope = RotaryEmbedding(dim=16, max_position_embeddings=32, base=10000.0)
        x = torch.randn(2, 10, 64)
        cos, sin = rope()
        y = attn(x, cos_sin=(cos[:10], sin[:10]))
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

    def test_weird_mode(self):
        mlp = SwiGLU(64, expansion=4.0, gate=True, weird=True)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == (2, 10, 64)
        assert mlp.norm is not None

    def test_non_weird_mode(self):
        mlp = SwiGLU(64, expansion=4.0, gate=True, weird=False)
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
        rope = RotaryEmbedding(dim=16, max_position_embeddings=32, base=10000.0)
        x = torch.randn(2, 10, 64)
        cos, sin = rope()
        y = block(x, cos_sin=(cos[:10], sin[:10]))
        assert y.shape == (2, 10, 64)


class TestMLPMixerBlock:
    def test_forward(self):
        block = MLPMixerBlock(64, seq_len=10)
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_weird_mode(self):
        block = MLPMixerBlock(64, seq_len=10, weird=True)
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
                out["logits"].view(-1, 12), labels.view(-1)
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
                out["logits"].view(-1, 12), labels.view(-1)
            )
            loss.backward()
            opt2.step()

        # Compare all parameters
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters(), strict=False
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), (
                f"{n1} diverged: max_diff={(p1 - p2).abs().max().item():.6e}"
            )


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
