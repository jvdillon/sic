"""Tests for Muon optimizer and related functions."""

from optimizer import (
    DummyOptimizer,
    Muon,
    _adjust_lr,
    matrix_signum_via_newtonschulz,
)
from torch import nn

import pytest
import torch


class TestMatrixSignumNewtonSchulz:
    """Test matrix signum computation via Newton-Schulz iteration."""

    def test_3d_input(self):
        """Test with 3D [batch, c_out, c_in]."""
        x = torch.randn(1, 5, 10)
        result = matrix_signum_via_newtonschulz(x)
        assert result.shape == x.shape

    def test_tall_matrix_transpose(self):
        """Test with tall matrix (more rows than cols - transpose path)."""
        x = torch.randn(1, 10, 5)
        result = matrix_signum_via_newtonschulz(x)
        assert result.shape == x.shape
        assert result.dtype == torch.bfloat16

    def test_batched(self):
        """Test with batch dimension > 1."""
        x = torch.randn(4, 5, 10)
        result = matrix_signum_via_newtonschulz(x)
        assert result.shape == x.shape

    def test_custom_steps(self):
        """Test with different number of iteration steps."""
        x = torch.randn(1, 5, 10)
        result = matrix_signum_via_newtonschulz(x, ns_steps=10)
        assert result.shape == x.shape

    def test_normalization(self):
        """Test that result has bounded norm."""
        torch.manual_seed(42)
        x = torch.randn(1, 10, 10)
        result = matrix_signum_via_newtonschulz(x, ns_steps=10)
        assert result.norm(dim=-1).mean() < 2.0


class TestAdjustLR:
    """Test learning rate adjustment functions."""

    def test_original_mode(self):
        """Test original Muon lr adjustment."""
        param = nn.Parameter(torch.randn(10, 5))  # c_out=10, c_in=5
        lr = 0.01
        adjusted = _adjust_lr(lr, param, "original")
        # max(1, 10/5)^0.5 = 2^0.5 ≈ 1.414
        assert adjusted == pytest.approx(lr * 1.414, rel=0.01)

    def test_match_rms_adamw_mode(self):
        """Test match_rms_adamw adjustment."""
        param = nn.Parameter(torch.randn(10, 5))
        lr = 0.01
        adjusted = _adjust_lr(lr, param, "match_rms_adamw")
        # 0.2 * max(10, 5)^0.5 = 0.2 * 10^0.5
        expected = lr * 0.2 * (10**0.5)
        assert adjusted == pytest.approx(expected, rel=0.01)

    def test_conv_heuristic_mode(self):
        """Test conv_heuristic adjustment."""
        param = nn.Parameter(torch.randn(10, 5, 3, 3))
        lr = 0.01
        adjusted = _adjust_lr(lr, param, "conv_heuristic")
        # Should use param norm / c_out^0.5
        expected = lr * param.data.norm().item() / (10**0.5)
        assert adjusted == pytest.approx(expected, rel=0.01)

    def test_conv_parameter(self):
        """Test with conv parameter (includes receptive field)."""
        param = nn.Parameter(torch.randn(8, 3, 5, 5))  # Conv2d
        lr = 0.01
        adjusted = _adjust_lr(lr, param, "original")
        # c_in *= 5*5 = 75, c_out = 8
        # max(1, 8/75)^0.5 = 1.0
        assert adjusted == pytest.approx(lr * 1.0, rel=0.01)

    def test_invalid_mode(self):
        """Test that invalid adjustment mode raises ValueError."""
        param = nn.Parameter(torch.randn(10, 5))
        with pytest.raises(ValueError, match="not supported"):
            _adjust_lr(0.01, param, "invalid_mode")  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]


class TestMuon:
    """Test Muon optimizer."""

    def test_init_basic(self):
        """Test basic initialization."""
        param = nn.Parameter(torch.randn(10, 5))
        opt = Muon([param], lr=0.01)
        assert opt.defaults["lr"] == 0.01
        assert len(opt.param_groups) == 1

    def test_invalid_lr(self):
        """Test that negative learning rate raises error."""
        param = nn.Parameter(torch.randn(10, 5))
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Muon([param], lr=-0.01)

    def test_invalid_momentum(self):
        """Test that negative momentum raises error."""
        param = nn.Parameter(torch.randn(10, 5))
        with pytest.raises(ValueError, match="Invalid momentum"):
            Muon([param], momentum=-0.1)

    def test_invalid_weight_decay(self):
        """Test that negative weight decay raises error."""
        param = nn.Parameter(torch.randn(10, 5))
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            Muon([param], weight_decay=-0.1)

    def test_nesterov_requires_momentum(self):
        """Test that Nesterov requires non-zero momentum."""
        param = nn.Parameter(torch.randn(10, 5))
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            Muon([param], nesterov=True, momentum=0.0)

    def test_invalid_eps(self):
        """Test that negative eps raises error."""
        param = nn.Parameter(torch.randn(10, 5))
        with pytest.raises(ValueError, match="Invalid eps"):
            Muon([param], eps=-1e-7)

    def test_step_basic(self):
        """Test basic optimization step."""
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(10, 5))
        opt = Muon([param], lr=0.01, momentum=0.9)

        # Set gradient
        param.grad = torch.randn(10, 5)

        # Take step
        param_before = param.data.clone()
        opt.step()

        # Parameter should have changed
        assert not torch.equal(param.data, param_before)

    def test_momentum_buffer_creation(self):
        """Test that momentum buffer is created on first step."""
        param = nn.Parameter(torch.randn(10, 5))
        opt = Muon([param], lr=0.01, momentum=0.9)
        param.grad = torch.randn(10, 5)

        # Before step - no buffer
        assert "momentum_buffer" not in opt.state[param]

        # After step - buffer exists
        opt.step()
        assert "momentum_buffer" in opt.state[param]
        assert opt.state[param]["momentum_buffer"].shape == param.shape

    def test_nesterov_momentum(self):
        """Test with Nesterov momentum enabled."""
        param = nn.Parameter(torch.randn(10, 5))
        opt = Muon([param], lr=0.01, momentum=0.9, nesterov=True)
        param.grad = torch.randn(10, 5)

        param_before = param.data.clone()
        opt.step()
        assert not torch.equal(param.data, param_before)

    def test_weight_decay(self):
        """Test weight decay application."""
        torch.manual_seed(42)
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(param1.data.clone())

        opt1 = Muon([param1], lr=0.01, weight_decay=0.1)
        opt2 = Muon([param2], lr=0.01, weight_decay=None)

        grad = torch.randn(10, 5)
        param1.grad = grad.clone()
        param2.grad = grad.clone()

        opt1.step()
        opt2.step()

        # With weight decay should have smaller norm
        assert param1.data.norm() < param2.data.norm()

    def test_skip_none_grads(self):
        """Test that parameters with None gradients are skipped."""
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10, 5))
        opt = Muon([param1, param2], lr=0.01)

        # Only set grad for param1
        param1.grad = torch.randn(10, 5)
        param2.grad = None

        param1_before = param1.data.clone()
        param2_before = param2.data.clone()

        opt.step()

        # param1 should change, param2 should not
        assert not torch.equal(param1.data, param1_before)
        assert torch.equal(param2.data, param2_before)

    def test_1d_param_raises_error(self):
        """Test that 1D parameters raise ValueError."""
        param = nn.Parameter(torch.randn(10))
        opt = Muon([param], lr=0.01)
        param.grad = torch.randn(10)

        with pytest.raises(ValueError, match="only defined for linear operators"):
            opt.step()

    def test_keller_adjust_mode(self):
        """Test keller learning rate adjustment mode."""
        param = nn.Parameter(torch.randn(10, 5))
        opt = Muon([param], lr=0.01, adjust_lr_fn="keller")
        param.grad = torch.randn(10, 5)

        param_before = param.data.clone()
        opt.step()
        assert not torch.equal(param.data, param_before)

    def test_multiple_param_groups(self):
        """Test with multiple parameter groups."""
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(8, 3))

        opt = Muon(
            [{"params": [param1], "lr": 0.01}, {"params": [param2], "lr": 0.001}],
        )

        param1.grad = torch.randn(10, 5)
        param2.grad = torch.randn(8, 3)

        opt.step()

        # Both should have momentum buffers
        assert "momentum_buffer" in opt.state[param1]
        assert "momentum_buffer" in opt.state[param2]

    def test_conv_parameters(self):
        """Test with convolutional layer parameters."""
        conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)
        opt = Muon(conv.parameters(), lr=0.01)

        # Create dummy gradient
        conv.weight.grad = torch.randn_like(conv.weight)

        weight_before = conv.weight.data.clone()
        opt.step()
        assert not torch.equal(conv.weight.data, weight_before)


class TestDummyOptimizer:
    """Test DummyOptimizer (no-op)."""

    def test_step_does_nothing(self):
        """Test that step does nothing."""
        opt = DummyOptimizer()
        opt.step()  # Should not raise

    def test_empty_param_groups(self):
        """Test that param_groups is empty."""
        opt = DummyOptimizer()
        assert opt.param_groups == []


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
