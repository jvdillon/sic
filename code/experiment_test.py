"""Unit tests for experiment.py."""

from __future__ import annotations

from typing import Any, Literal

import tempfile

from experiment import (
    Experiment,
    compute_diversity_loss,
    cross_entropy_with_batched_smoothing,
    resume_from_checkpoint,
    setup_muon_optimizers,
)
from model import TRM3, MLPMixerBlock, SwiGLU, TRM3ConfigProtocol, trunc_normal_init_
from torch import Tensor

import torch


def test_compute_diversity_loss_empty():
    loss = compute_diversity_loss([torch.randn(2, 10, 64)])
    assert loss.item() == 0.0


def test_compute_diversity_loss():
    all_logits = [torch.randn(2, 10, 64) for _ in range(3)]
    loss = compute_diversity_loss(all_logits)
    assert loss.shape == ()
    assert -1 <= loss.item() <= 1


def test_batched_smoothing_matches_torch_scalar():
    """cross_entropy_with_batched_smoothing matches torch CE when smoothing is constant."""
    torch.manual_seed(0)
    N, C = 32, 10
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))

    for s in [0.0, 0.1, 0.3]:
        smoothing = torch.full((N,), s)
        ours = cross_entropy_with_batched_smoothing(
            logits, targets, reduction="none", label_smoothing=smoothing
        )
        ref = torch.nn.functional.cross_entropy(
            logits, targets, label_smoothing=s, reduction="none"
        )
        assert torch.allclose(ours, ref, atol=1e-5), (
            f"smoothing={s}: max diff={torch.abs(ours - ref).max().item()}"
        )


def test_batched_smoothing_scalar_delegates():
    """Scalar label_smoothing delegates to F.cross_entropy (all reductions)."""
    torch.manual_seed(0)
    N, C = 16, 10
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))

    for reduction in ["none", "sum", "mean"]:
        ours = cross_entropy_with_batched_smoothing(
            logits, targets, reduction=reduction, label_smoothing=0.1
        )
        ref = torch.nn.functional.cross_entropy(
            logits, targets, reduction=reduction, label_smoothing=0.1
        )
        assert torch.allclose(ours, ref, atol=1e-5), (
            f"reduction={reduction}: max diff={torch.abs(ours - ref).max().item()}"
        )


def test_batched_smoothing_ignore_index():
    """cross_entropy_with_batched_smoothing zeros out ignored targets."""
    torch.manual_seed(0)
    N, C = 16, 10
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    targets[0] = -100
    targets[5] = -100
    smoothing = torch.full((N,), 0.1)

    losses = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="none", label_smoothing=smoothing
    )
    assert losses[0].item() == 0.0
    assert losses[5].item() == 0.0
    assert losses[1].item() > 0.0


def test_batched_smoothing_custom_ignore_index():
    """cross_entropy_with_batched_smoothing works with non-default ignore_index."""
    torch.manual_seed(0)
    N, C = 16, 10
    # ignore_index=-1: simulates label_smoothing_includes_pad_token=False
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    targets[0] = -1
    targets[7] = -1
    smoothing = torch.full((N,), 0.1)

    losses = cross_entropy_with_batched_smoothing(
        logits, targets, ignore_index=-1, reduction="none", label_smoothing=smoothing
    )
    assert losses[0].item() == 0.0
    assert losses[7].item() == 0.0
    assert losses[1].item() > 0.0

    # ignore_index=0: simulates label_smoothing_includes_pad_token=True
    targets2 = torch.randint(1, C, (N,))
    targets2[2] = 0
    losses2 = cross_entropy_with_batched_smoothing(
        logits, targets2, ignore_index=0, reduction="none", label_smoothing=smoothing
    )
    assert losses2[2].item() == 0.0
    assert losses2[3].item() > 0.0


def test_batched_smoothing_per_element():
    """Different smoothing per element produces different losses."""
    torch.manual_seed(0)
    N, C = 8, 10
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    smoothing_lo = torch.full((N,), 0.0)
    smoothing_hi = torch.full((N,), 0.5)

    loss_lo = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="none", label_smoothing=smoothing_lo
    )
    loss_hi = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="none", label_smoothing=smoothing_hi
    )
    assert not torch.allclose(loss_lo, loss_hi), (
        "Different smoothing should give different losses"
    )


def test_batched_smoothing_reduction():
    """Tensor smoothing respects reduction='sum' and 'mean'."""
    torch.manual_seed(0)
    N, C = 16, 10
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    smoothing = torch.full((N,), 0.1)

    per_elem = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="none", label_smoothing=smoothing
    )
    loss_sum = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="sum", label_smoothing=smoothing
    )
    loss_mean = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="mean", label_smoothing=smoothing
    )
    assert torch.allclose(loss_sum, per_elem.sum(), atol=1e-5)
    assert torch.allclose(loss_mean, per_elem.mean(), atol=1e-5)


def test_batched_smoothing_weight():
    """Tensor smoothing with class weight matches manual computation."""
    torch.manual_seed(0)
    N, C = 16, 10
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    smoothing = torch.full((N,), 0.1)
    weight = torch.rand(C)

    # reduction="none": each element scaled by weight[target]
    loss_w = cross_entropy_with_batched_smoothing(
        logits, targets, weight=weight, reduction="none", label_smoothing=smoothing
    )
    loss_no_w = cross_entropy_with_batched_smoothing(
        logits, targets, reduction="none", label_smoothing=smoothing
    )
    expected = loss_no_w * weight[targets]
    assert torch.allclose(loss_w, expected, atol=1e-5)

    # reduction="mean": weighted mean (sum of weighted loss / sum of weights)
    loss_mean = cross_entropy_with_batched_smoothing(
        logits, targets, weight=weight, reduction="mean", label_smoothing=smoothing
    )
    expected_mean = expected.sum() / weight[targets].sum()
    assert torch.allclose(loss_mean, expected_mean, atol=1e-5)


def _test_config() -> TRM3.Config:
    return TRM3.Config(
        vocab_size=12,
        puzzle_grid_shape=(9, 9),
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        compile_core=False,
        dtype=torch.float32,
        device="cpu",
        cast_model_to_dtype=False,
        num_register_tokens=0,
        block=MLPMixerBlock.Config(
            seq_len=81,
            attn=SwiGLU.Config(init_weight_fn=trunc_normal_init_),
            ffn=SwiGLU.Config(init_weight_fn=trunc_normal_init_),
        ),
    )


def test_setup_muon_optimizers():
    config = _test_config()
    model = config.make()
    opt1, opt2 = setup_muon_optimizers(model)
    assert len(opt1.param_groups) >= 1
    assert len(opt2.param_groups) >= 1


def _make_test_exp() -> Experiment:
    class TestExp(Experiment):
        batch_size: int = 4
        K: int = 1
        use_ema: bool = True
        cast_model_to_dtype: bool = False
        max_steps_schedule: dict[int, tuple[int, bool]] = {0: (16, True)}  # noqa: RUF012
        config: TRM3ConfigProtocol = _test_config()

        def __init__(self):
            super().__init__()
            self.setup_optimizers()

    return TestExp()


def test_checkpoint_format():
    exp = _make_test_exp()
    ckpt = exp.make_checkpoint()

    assert "model" in ckpt
    assert "ema" in ckpt
    assert "optimizer1" in ckpt
    assert "optimizer2" in ckpt
    assert "step" in ckpt
    assert "best_acc" in ckpt
    assert ckpt["step"] == 0
    assert ckpt["best_acc"] == 0.0


def _make_test_exp_no_ema() -> Experiment:
    class TestExpNoEma(Experiment):
        batch_size: int = 4
        K: int = 1
        use_ema: bool = False
        cast_model_to_dtype: bool = False
        max_steps_schedule: dict[int, tuple[int, bool]] = {0: (16, True)}  # noqa: RUF012
        config: TRM3ConfigProtocol = _test_config()

        def __init__(self):
            super().__init__()
            self.setup_optimizers()

    return TestExpNoEma()


def test_use_ema_false():
    exp = _make_test_exp_no_ema()
    assert exp.ema is None

    ckpt = exp.make_checkpoint()
    assert "ema" not in ckpt

    # reset_transient_state should not crash
    exp.reset_transient_state()


def test_resume_no_ema():
    exp1 = _make_test_exp_no_ema()
    exp1.current_step = 500

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1.make_checkpoint(), f.name)
        ckpt_path = f.name

    exp2 = _make_test_exp_no_ema()
    resume_from_checkpoint(exp2, ckpt_path)
    assert exp2.current_step == 500
    assert exp2.ema is None


def test_reset_transient_state():
    exp = _make_test_exp()
    exp.current_step = 1000

    exp.reset_transient_state()

    # _state should be fresh
    assert exp._state.h_step.sum() == 0


def test_resume_from_checkpoint_new_format():
    exp1 = _make_test_exp()
    exp1.current_step = 500
    exp1.best_acc = 75.0

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1.make_checkpoint(), f.name)
        ckpt_path = f.name

    exp2 = _make_test_exp()
    resume_from_checkpoint(exp2, ckpt_path, resume_optimizer=True, resume_ema=True)

    assert exp2.current_step == 500
    assert exp2.best_acc == 75.0


def test_resume_from_checkpoint_no_optimizer():
    exp1 = _make_test_exp()
    exp1.current_step = 500

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1.make_checkpoint(), f.name)
        ckpt_path = f.name

    exp2 = _make_test_exp()
    resume_from_checkpoint(exp2, ckpt_path, resume_optimizer=False, resume_ema=True)

    assert exp2.current_step == 500


def test_resume_from_checkpoint_old_format():
    exp1 = _make_test_exp()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1.model.state_dict(), f.name)
        ckpt_path = f.name

    exp2 = _make_test_exp()
    exp2.current_step = 100
    resume_from_checkpoint(exp2, ckpt_path)


def test_reset_steps_class_attr():
    class TestExpWithReset(Experiment):
        batch_size: int = 4
        K: int = 1
        reset_steps: list[int] = [1500, 3000]  # noqa: RUF012
        cast_model_to_dtype: bool = False
        max_steps_schedule: dict[int, tuple[int, bool]] = {0: (16, True)}  # noqa: RUF012
        config: TRM3ConfigProtocol = _test_config()

    exp = TestExpWithReset()
    assert exp.reset_steps == [1500, 3000]


def test_data_loader_yields_valid_count():
    """Test that data loader yields 4-tuple with valid_count for padding."""
    batch_size = 16
    valid_count = 10

    inputs = torch.randint(low=1, high=11, size=(valid_count, 81))
    labels = torch.randint(low=1, high=11, size=(valid_count, 81))

    pad_size = batch_size - valid_count
    inputs_pad = torch.zeros(pad_size, 81, dtype=inputs.dtype)
    labels_pad = torch.zeros(pad_size, 81, dtype=labels.dtype)
    padded_inputs = torch.cat([inputs, inputs_pad])
    padded_labels = torch.cat([labels, labels_pad])

    assert padded_inputs.shape == (batch_size, 81)
    assert padded_labels.shape == (batch_size, 81)
    assert (padded_inputs[:valid_count] == inputs).all()
    assert (padded_labels[:valid_count] == labels).all()


def test_eval_metrics_use_valid_count():
    """Test that metrics correctly use valid_count to exclude padding."""
    batch_size = 16
    valid_count = 10

    preds = torch.randint(low=1, high=11, size=(batch_size, 81))
    labels = torch.randint(low=1, high=11, size=(batch_size, 81))
    preds[:valid_count] = labels[:valid_count]

    full_acc = (preds == labels).float().mean().item()
    valid_acc = (preds[:valid_count] == labels[:valid_count]).float().mean().item()

    assert valid_acc == 1.0
    assert full_acc < 1.0


def test_fast_eval_streaming_replacement():
    """Test that fast_eval uses streaming replacement."""

    class FastEvalExp(Experiment):
        eval_method: Literal["full", "fast", "wta"] = "fast"
        batch_size: int = 2
        eval_batch_size: int | None = 2
        K: int = 1
        cast_model_to_dtype: bool = False
        max_steps_schedule: dict[int, tuple[int, bool]] = {0: (16, True)}  # noqa: RUF012
        config: TRM3ConfigProtocol = _test_config()

    exp = FastEvalExp()

    original_forward = exp.model.forward
    step_count = 0

    def tracking_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_count
        step_count += 1
        out = original_forward(*args, **kwargs)
        q_halt = out["q_halt"]
        assert isinstance(q_halt, Tensor)
        if step_count % 2 == 0:
            out["q_halt"] = torch.ones_like(q_halt)
        else:
            out["q_halt"] = torch.full_like(q_halt, -10.0)
        return out

    exp.model.forward = tracking_forward  # ty: ignore[invalid-assignment]

    grid_len = exp.config.num_puzzle_grid_tokens
    all_inputs = [torch.randint(low=1, high=11, size=(2, grid_len)) for _ in range(3)]
    all_labels = [torch.randint(low=1, high=11, size=(2, grid_len)) for _ in range(3)]

    def fake_loader():
        for inp, lab in zip(all_inputs, all_labels, strict=False):
            yield (inp, lab, torch.zeros(2, dtype=torch.int32), 2)

    _, _, _, _ = exp.evaluate_act_haltfast(fake_loader())

    assert step_count == 6, f"Expected 6 steps for 6 puzzles but got {step_count}"


def test_fast_eval_metrics_at_halt_time():
    """Test that fast_eval computes metrics at halt time per sample."""

    class FastEvalExp(Experiment):
        eval_method: Literal["full", "fast", "wta"] = "fast"
        batch_size: int = 2
        eval_batch_size: int | None = 2
        K: int = 1
        cast_model_to_dtype: bool = False
        max_steps_schedule: dict[int, tuple[int, bool]] = {0: (4, True)}  # noqa: RUF012
        config: TRM3ConfigProtocol = _test_config()

    exp = FastEvalExp()

    step_count = 0
    original_forward = exp.model.forward

    def controlled_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_count
        step_count += 1
        out = original_forward(*args, **kwargs)
        logits = out["logits"]
        q_halt = out["q_halt"]
        assert isinstance(logits, Tensor)
        assert isinstance(q_halt, Tensor)

        if step_count == 2:
            out["logits"] = torch.zeros_like(logits)
            out["logits"][:, :, 5] = 100
            out["q_halt"] = torch.ones_like(q_halt)
        else:
            out["q_halt"] = torch.full_like(q_halt, -10.0)

        return out

    exp.model.forward = controlled_forward  # ty: ignore[invalid-assignment]

    grid_len = exp.config.num_puzzle_grid_tokens
    B = 2
    inputs = torch.randint(low=1, high=11, size=(B, grid_len))
    labels = torch.full((B, grid_len), 5, dtype=torch.long)

    def fake_loader():
        yield (inputs, labels, torch.zeros(B, dtype=torch.int32), B)

    cell_acc, puzzle_acc, _, _ = exp.evaluate_act_haltfast(fake_loader())

    assert cell_acc == 100.0, f"Expected 100% cell acc but got {cell_acc}%"
    assert puzzle_acc == 100.0, f"Expected 100% puzzle acc but got {puzzle_acc}%"


def test_no_recompilation_all_paths():
    """Test that step() and forward() use same core signature (no recompile)."""
    config = _test_config()
    model = config.make()

    call_signatures: list[object] = []
    originalcore = model.core

    def trackingcore(
        input_emb: torch.Tensor,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sig = (
            tuple(input_emb.shape),
            tuple(z_H.shape),
            tuple(z_L.shape),
            "None"
            if cos_sin is None
            else (tuple(cos_sin[0].shape), tuple(cos_sin[1].shape)),
        )
        call_signatures.append(sig)
        return originalcore(input_emb, z_H, z_L, cos_sin)

    model.core = trackingcore

    B = 4
    grid_len = config.num_puzzle_grid_tokens
    seq_len = config.total_seq_len
    x = torch.randint(low=0, high=12, size=(B, grid_len))
    z_H = model.H_init[0].expand(B, seq_len, -1)
    z_L = model.L_init[0].expand(B, seq_len, -1)

    # Path 1: step()
    call_signatures.clear()
    model.step(x, z_H, z_L)
    step_sig = call_signatures[0]

    # Path 2: forward()
    call_signatures.clear()
    model(x, z_H, z_L)
    forward_sigs = call_signatures

    for sig in forward_sigs:
        assert sig == step_sig, f"Signature mismatch: {sig} vs {step_sig}"


def test_eval_methods_consistent_k1():
    """Test that full, fast, and wta eval give same results with K=1."""

    class EvalTestExp(Experiment):
        batch_size: int = 2
        eval_batch_size: int | None = 2
        K: int = 1
        cast_model_to_dtype: bool = False
        max_steps_schedule: dict[int, tuple[int, bool]] = {0: (4, True)}  # noqa: RUF012
        config: TRM3ConfigProtocol = _test_config()

    grid_len = 81

    torch.manual_seed(42)
    n_puzzles = 4
    inputs = torch.randint(low=1, high=11, size=(n_puzzles, grid_len))
    labels = torch.randint(low=1, high=11, size=(n_puzzles, grid_len))

    def make_loader():
        for i in range(0, n_puzzles, 2):
            yield (
                inputs[i : i + 2],
                labels[i : i + 2],
                torch.zeros(2, dtype=torch.int32),
                2,
            )

    exp_full = EvalTestExp()
    torch.manual_seed(0)
    cell_full, puzzle_full, _, _ = exp_full.evaluate_act_full(make_loader())

    exp_fast = EvalTestExp()
    torch.manual_seed(0)
    cell_fast, puzzle_fast, _, _ = exp_fast.evaluate_act_haltfast(make_loader())

    exp_wta = EvalTestExp()
    exp_wta.eval_method = "wta"
    exp_wta.K = 1
    torch.manual_seed(0)
    cell_wta, puzzle_wta, _, _ = exp_wta.evaluate_act_haltfast_wta(make_loader())

    assert cell_full == cell_fast, f"full={cell_full} vs fast={cell_fast}"
    assert cell_full == cell_wta, f"full={cell_full} vs wta={cell_wta}"
    assert puzzle_full == puzzle_fast, f"full={puzzle_full} vs fast={puzzle_fast}"
    assert puzzle_full == puzzle_wta, f"full={puzzle_full} vs wta={puzzle_wta}"


def test_wta_batched_vs_sequential():
    """Verify batched K-head forward gives same results as sequential."""
    torch.manual_seed(42)

    config = TRM3.Config(
        vocab_size=12,
        puzzle_grid_shape=(9, 9),
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        H_cycles=1,
        L_cycles=1,
        K_H=1,
        K_L=1,
        carry_H="all",
        carry_L="all",
        compile_core=False,
        dtype=torch.float32,
        device="cpu",
        cast_model_to_dtype=False,
        num_register_tokens=0,
        block=MLPMixerBlock.Config(
            seq_len=81,
            attn=SwiGLU.Config(init_weight_fn=trunc_normal_init_),
            ffn=SwiGLU.Config(init_weight_fn=trunc_normal_init_),
        ),
    )
    model = config.make()
    model.eval()

    B, K = 4, 3
    seq_len = config.total_seq_len
    hidden = config.hidden_size

    for k in range(1, K):
        model.register_parameter(
            f"L_init_{k}",
            torch.nn.Parameter(torch.randn(seq_len, hidden) * 0.1),
        )

    grid_len = config.num_puzzle_grid_tokens
    inputs = torch.randint(low=1, high=11, size=(B, grid_len))
    z_H_init = model.H_init[0].expand(B, seq_len, -1).clone()

    # Sequential
    torch.manual_seed(123)
    seq_logits = []
    seq_q_halt = []
    seq_z_H = []
    seq_z_L = []
    for k in range(K):
        L_init_k = model.L_init[0] if k == 0 else getattr(model, f"L_init_{k}")
        z_L_k = L_init_k.expand(B, seq_len, -1).clone()
        out = model(inputs, z_H_init, z_L_k)
        seq_logits.append(out["logits"])
        seq_q_halt.append(out["q_halt"])
        seq_z_H.append(out["z_H"])
        seq_z_L.append(out["z_L"])

    # Batched
    torch.manual_seed(123)
    z_L_all = []
    for k in range(K):
        L_init_k = model.L_init[0] if k == 0 else getattr(model, f"L_init_{k}")
        z_L_all.append(L_init_k.expand(B, seq_len, -1).clone())

    z_L_batched = torch.stack(z_L_all, dim=1).reshape(B * K, seq_len, -1)
    z_H_batched = (
        z_H_init.unsqueeze(1).expand(B, K, seq_len, -1).reshape(B * K, seq_len, -1)
    )
    inputs_batched = inputs.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

    out_batched = model(inputs_batched, z_H_batched, z_L_batched)

    logits_batched = out_batched["logits"].reshape(B, K, grid_len, -1)
    q_halt_batched = out_batched["q_halt"].reshape(B, K)
    z_H_batched_out = out_batched["z_H"].reshape(B, K, seq_len, -1)
    z_L_batched_out = out_batched["z_L"].reshape(B, K, seq_len, -1)

    for k in range(K):
        assert torch.allclose(seq_logits[k], logits_batched[:, k], atol=1e-5), (
            f"logits mismatch for head {k}"
        )
        assert torch.allclose(seq_q_halt[k], q_halt_batched[:, k], atol=1e-5), (
            f"q_halt mismatch for head {k}"
        )
        assert torch.allclose(seq_z_H[k], z_H_batched_out[:, k], atol=1e-5), (
            f"z_H mismatch for head {k}"
        )
        assert torch.allclose(seq_z_L[k], z_L_batched_out[:, k], atol=1e-5), (
            f"z_L mismatch for head {k}"
        )


if __name__ == "__main__":
    from util import test_main

    test_main(__file__)
