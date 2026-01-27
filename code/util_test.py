"""Unit tests for util.py."""

from __future__ import annotations

import io
import math
import pathlib
import sys
import tempfile

from util import Tee, entropy, jsd, log1mexp, logmeanexp

import torch


class TestTee:
    def test_captures_stdout(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        # Create mock stdout with proper .name attribute
        mock_stdout = io.StringIO()
        mock_stdout.name = "<stdout>"

        tee = Tee(log_path, stream=(mock_stdout,))
        with tee:
            sys.stdout.write("hello stdout\n")

        with open(log_path) as f:
            content = f.read()
        assert "hello stdout" in content

    def test_captures_stderr(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        mock_stderr = io.StringIO()
        mock_stderr.name = "<stderr>"

        tee = Tee(log_path, stream=(mock_stderr,))
        with tee:
            sys.stderr.write("hello stderr\n")

        with open(log_path) as f:
            content = f.read()
        assert "hello stderr" in content

    def test_captures_both_stdout_and_stderr(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        mock_stdout = io.StringIO()
        mock_stdout.name = "<stdout>"
        mock_stderr = io.StringIO()
        mock_stderr.name = "<stderr>"

        tee = Tee(log_path, stream=(mock_stdout, mock_stderr))
        with tee:
            sys.stdout.write("from stdout\n")
            sys.stderr.write("from stderr\n")

        with open(log_path) as f:
            content = f.read()
        assert "from stdout" in content
        assert "from stderr" in content

    def test_restores_streams_after_exit(self):
        mock_stdout = io.StringIO()
        mock_stdout.name = "<stdout>"
        mock_stderr = io.StringIO()
        mock_stderr.name = "<stderr>"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        # Save what pytest has as stdout/stderr
        pytest_stdout = sys.stdout
        pytest_stderr = sys.stderr

        tee = Tee(log_path, stream=(mock_stdout, mock_stderr))
        with tee:
            # Inside the context, sys.stdout/stderr are Tee objects
            assert isinstance(sys.stdout, Tee)
            assert isinstance(sys.stderr, Tee)

        # After exit, streams are restored to the mock streams (what we passed in)
        assert sys.stdout is mock_stdout
        assert sys.stderr is mock_stderr

        # Restore pytest's streams
        sys.stdout = pytest_stdout
        sys.stderr = pytest_stderr

    def test_captures_exception_traceback(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        mock_stdout = io.StringIO()
        mock_stdout.name = "<stdout>"
        mock_stderr = io.StringIO()
        mock_stderr.name = "<stderr>"

        try:
            with Tee(log_path, stream=(mock_stdout, mock_stderr)):
                sys.stdout.write("stdout message\n")
                sys.stderr.write("stderr message\n")
                raise ValueError("test exception")
        except ValueError:
            pass

        log_contents = pathlib.Path(log_path).read_text()
        assert "stdout message" in log_contents, "stdout not captured"
        assert "stderr message" in log_contents, "stderr not captured"
        assert "ValueError" in log_contents, "Exception type not captured in log"
        assert "test exception" in log_contents, "Exception message not captured in log"


class TestLog1mexp:
    def test_basic_values(self):
        # log(1 - exp(-1)) ≈ log(1 - 0.368) ≈ log(0.632) ≈ -0.459
        x = torch.tensor([-1.0])
        result = log1mexp(x)
        expected = math.log(1 - math.exp(-1))
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)

    def test_small_x(self):
        # For x << -log(2), uses log1p(-exp(x)) branch
        x = torch.tensor([-10.0])
        result = log1mexp(x)
        # log(1 - exp(-10)) ≈ log(1 - 4.5e-5) ≈ -4.5e-5
        expected = math.log1p(-math.exp(-10))
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)

    def test_near_log2(self):
        # Near -log(2) ≈ -0.693, both branches should give same result
        x = torch.tensor([-0.693])
        result = log1mexp(x)
        expected = math.log(1 - math.exp(-0.693))
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-3)

    def test_batched(self):
        x = torch.tensor([-0.1, -1.0, -5.0])
        result = log1mexp(x)
        expected = torch.tensor(
            [
                math.log(1 - math.exp(-0.1)),
                math.log(1 - math.exp(-1.0)),
                math.log(1 - math.exp(-5.0)),
            ]
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_grad_normal(self):
        # Gradient at x = -1: d/dx log(1 - e^x) = -e^x / (1 - e^x)
        x = torch.tensor([-1.0], requires_grad=True)
        y = log1mexp(x)
        y.backward()
        # Analytic: -exp(-1) / (1 - exp(-1)) ≈ -0.368 / 0.632 ≈ -0.582
        expected_grad = -math.exp(-1) / (1 - math.exp(-1))
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([expected_grad]), atol=1e-5)

    def test_grad_small_x(self):
        # For very negative x, gradient ≈ -exp(x) / 1 ≈ 0
        x = torch.tensor([-10.0], requires_grad=True)
        y = log1mexp(x)
        y.backward()
        assert x.grad is not None
        assert x.grad.abs() < 1e-4

    def test_grad_near_zero(self):
        # Near x=0, gradient should be large (approaches -inf at x=0)
        x = torch.tensor([-0.01], requires_grad=True)
        y = log1mexp(x)
        y.backward()
        # Gradient: -exp(-0.01) / (1 - exp(-0.01)) ≈ -0.99 / 0.01 ≈ -100
        expected = -math.exp(-0.01) / (1 - math.exp(-0.01))
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.tensor([expected]), rtol=0.01)

    def test_at_zero_returns_neg_inf(self):
        # log(1 - exp(0)) = log(0) = -inf
        x = torch.tensor([0.0])
        result = log1mexp(x)
        assert result.item() == float("-inf")

    def test_grad_at_zero(self):
        # At x=0, gradient is -inf (derivative of log(1-e^x) at x=0)
        # d/dx log(1 - e^x) = -e^x / (1 - e^x) -> -inf as x -> 0
        x = torch.tensor([0.0], requires_grad=True)
        y = log1mexp(x)
        y.backward()
        assert x.grad is not None
        # Gradient should be -inf at x=0
        assert (
            x.grad.item() == float("-inf") or x.grad.item() != x.grad.item()
        )  # -inf or nan


class TestLogmeanexp:
    def test_basic(self):
        # logmeanexp([0, 0]) = log(mean(exp([0, 0]))) = log(1) = 0
        x = torch.tensor([[0.0, 0.0]])
        result = logmeanexp(x, dim=-1)
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_different_values(self):
        # logmeanexp([0, ln(2)]) = log((1 + 2) / 2) = log(1.5)
        x = torch.tensor([[0.0, math.log(2)]])
        result = logmeanexp(x, dim=-1)
        expected = math.log(1.5)
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-6)

    def test_vs_logsumexp(self):
        x = torch.randn(3, 5)
        result = logmeanexp(x, dim=-1)
        expected = torch.logsumexp(x, dim=-1) - math.log(5)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_keepdim(self):
        x = torch.randn(2, 3, 4)
        result = logmeanexp(x, dim=1, keepdim=True)
        assert result.shape == (2, 1, 4)

    def test_grad(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = logmeanexp(x, dim=-1).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Gradient should be softmax(x) / K where K is the size of dim
        expected_grad = torch.softmax(x, dim=-1)
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)


class TestEntropy:
    def test_uniform(self):
        # Uniform over 4 classes: H = log(4)
        logp = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
        result = entropy(logp, dim=-1)
        expected = math.log(4)
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-5)

    def test_deterministic(self):
        # Delta distribution: H = 0
        logp = torch.tensor([[0.0, float("-inf"), float("-inf"), float("-inf")]])
        result = entropy(logp, dim=-1)
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-5)

    def test_binary(self):
        # Binary with p=0.5: H = log(2)
        logp = torch.log(torch.tensor([[0.5, 0.5]]))
        result = entropy(logp, dim=-1)
        expected = math.log(2)
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-5)

    def test_batched(self):
        logp = torch.log_softmax(torch.randn(5, 10), dim=-1)
        result = entropy(logp, dim=-1)
        assert result.shape == (5,)

    def test_keepdim(self):
        logp = torch.log_softmax(torch.randn(2, 3, 4), dim=-1)
        result = entropy(logp, dim=-1, keepdim=True)
        assert result.shape == (2, 3, 1)

    def test_grad_uniform(self):
        # For uniform, gradient of H wrt logits is 0 (at the maximum)
        logits = torch.zeros(1, 4, requires_grad=True)
        logp = torch.log_softmax(logits, dim=-1)
        H = entropy(logp, dim=-1).sum()
        H.backward()
        assert logits.grad is not None
        # At uniform, gradient should be ~0
        assert logits.grad.abs().max() < 1e-5

    def test_grad_non_uniform(self):
        logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        logp = torch.log_softmax(logits, dim=-1)
        H = entropy(logp, dim=-1).sum()
        H.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (1, 3)

    def test_handles_neg_inf(self):
        # Test that -inf values don't produce NaN
        logp = torch.tensor([[0.0, float("-inf"), float("-inf")]])
        result = entropy(logp, dim=-1)
        assert torch.isfinite(result).all()
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-5)

    def test_grad_with_neg_inf(self):
        # Verify -inf doesn't produce NaN gradient
        logp = torch.tensor([[0.0, float("-inf"), float("-inf")]], requires_grad=True)
        H = entropy(logp, dim=-1)
        H.sum().backward()
        assert logp.grad is not None
        assert torch.isfinite(logp.grad).all(), f"grad has non-finite: {logp.grad}"

    def test_grad_with_neg_inf_mixed(self):
        # Mixed finite and -inf values
        logp = torch.tensor([[-0.5, -1.0, float("-inf")]], requires_grad=True)
        H = entropy(logp, dim=-1)
        H.sum().backward()
        assert logp.grad is not None
        assert torch.isfinite(logp.grad).all(), f"grad has non-finite: {logp.grad}"


class TestJSD:
    def test_identical_distributions(self):
        # JSD of identical distributions = 0
        logp = torch.log_softmax(torch.randn(1, 10, 5).expand(4, 10, 5), dim=-1)
        result = jsd(logp, ensemble_dim=0, event_dim=-1)
        assert result.shape == (10,)
        assert torch.allclose(result, torch.zeros(10), atol=1e-5)

    def test_maximally_different(self):
        # JSD of K delta distributions on different classes approaches log(K)
        K, N, C = 4, 1, 4
        logp = torch.full((K, N, C), float("-inf"))
        for k in range(K):
            logp[k, :, k] = 0.0  # Delta on class k
        result = jsd(logp, ensemble_dim=0, event_dim=-1)
        # JSD = H(uniform) - mean(H(deltas)) = log(4) - 0 = log(4)
        expected = math.log(K)
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-5)

    def test_two_distributions(self):
        # JSD(p, q) for p uniform, q delta
        K, N, C = 2, 1, 4
        logp = torch.zeros(K, N, C)
        logp[0] = math.log(0.25)  # Uniform
        logp[1] = torch.tensor([[0.0, float("-inf"), float("-inf"), float("-inf")]])
        result = jsd(logp, ensemble_dim=0, event_dim=-1)
        # Mixture is (uniform + delta) / 2 = [0.625, 0.125, 0.125, 0.125]
        # H(mixture) - (H(uniform) + H(delta)) / 2 = H(mixture) - log(4)/2
        mix = torch.tensor([0.625, 0.125, 0.125, 0.125])
        H_mix = -(mix * mix.log()).sum()
        H_uniform = math.log(4)
        expected = H_mix - H_uniform / 2
        assert torch.allclose(result, torch.tensor([expected]), atol=1e-4)

    def test_shape(self):
        logp = torch.log_softmax(torch.randn(3, 20, 9), dim=-1)
        result = jsd(logp, ensemble_dim=0, event_dim=-1)
        assert result.shape == (20,)

    def test_grad(self):
        logits = torch.randn(4, 10, 5, requires_grad=True)
        logp = torch.log_softmax(logits, dim=-1)
        loss = jsd(logp, ensemble_dim=0, event_dim=-1).sum()
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (4, 10, 5)
        assert torch.isfinite(logits.grad).all()

    def test_grad_identical_distributions(self):
        # Gradient should push distributions apart
        logits = torch.zeros(4, 5, 3, requires_grad=True)
        logp = torch.log_softmax(logits, dim=-1)
        loss = jsd(logp, ensemble_dim=0, event_dim=-1).sum()
        loss.backward()
        assert logits.grad is not None
        # At identical distributions, gradient should be 0 (saddle point)
        assert logits.grad.abs().max() < 1e-5

    def test_nonnegative(self):
        # JSD is always >= 0
        for _ in range(10):
            logp = torch.log_softmax(torch.randn(4, 20, 9), dim=-1)
            result = jsd(logp, ensemble_dim=0, event_dim=-1)
            assert (result >= -1e-6).all()


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
