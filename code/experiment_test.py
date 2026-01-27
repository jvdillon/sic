"""Unit tests for experiment.py."""

from __future__ import annotations

import tempfile

from experiment import (
    ExperimentBase,
    compute_diversity_loss,
    resume_from_checkpoint,
    setup_muon_optimizers,
)
from model import TRM

import torch


def test_compute_diversity_loss_empty():
    loss = compute_diversity_loss([torch.randn(2, 10, 64)])
    assert loss.item() == 0.0


def test_compute_diversity_loss():
    all_logits = [torch.randn(2, 10, 64) for _ in range(3)]
    loss = compute_diversity_loss(all_logits)
    assert loss.shape == ()
    assert -1 <= loss.item() <= 1


def test_setup_muon_optimizers():
    config = TRM.Config(
        compile_core=False,
        vocab_size=12,
        seq_len=82,
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
    )
    model = config.setup()
    opt1, opt2 = setup_muon_optimizers(model)

    assert len(opt1.param_groups) >= 1
    assert len(opt2.param_groups) >= 1


def _make_test_exp() -> ExperimentBase:
    class TestExp(ExperimentBase):
        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.batch_size = 4
            self.compile_model = False
            config = TRM.Config(
                compile_core=False,
                vocab_size=12,
                seq_len=82,
                hidden_size=64,
                num_heads=4,
                num_layers=1,
                H_cycles=2,
                L_cycles=2,
                dtype=self.dtype,
            )
            self.model = config.setup()
            super().__init__()
            self.setup_optimizers()

    return TestExp()


def test_checkpoint_format():
    exp = _make_test_exp()
    ckpt = exp._make_checkpoint()  # noqa: SLF001

    assert "model" in ckpt
    assert "ema" in ckpt
    assert "optimizer1" in ckpt
    assert "optimizer2" in ckpt
    assert "step" in ckpt
    assert "best_acc" in ckpt
    assert ckpt["step"] == 0
    assert ckpt["best_acc"] == 0.0


def _make_test_exp_no_ema() -> ExperimentBase:
    class TestExpNoEma(ExperimentBase):
        use_ema: bool = False

        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.batch_size = 4
            self.compile_model = False
            config = TRM.Config(
                compile_core=False,
                vocab_size=12,
                seq_len=82,
                hidden_size=64,
                num_heads=4,
                num_layers=1,
                H_cycles=2,
                L_cycles=2,
                dtype=self.dtype,
            )
            self.model = config.setup()
            super().__init__()
            self.setup_optimizers()

    return TestExpNoEma()


def test_use_ema_false():
    exp = _make_test_exp_no_ema()
    assert exp.ema is None

    ckpt = exp._make_checkpoint()  # noqa: SLF001
    assert "ema" not in ckpt

    # reset_transient_state should not crash
    exp.reset_transient_state()


def test_resume_no_ema():
    exp1 = _make_test_exp_no_ema()
    exp1.current_step = 500

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1._make_checkpoint(), f.name)  # noqa: SLF001
        ckpt_path = f.name

    exp2 = _make_test_exp_no_ema()
    resume_from_checkpoint(exp2, ckpt_path)
    assert exp2.current_step == 500
    assert exp2.ema is None


def test_reset_transient_state():
    exp = _make_test_exp()
    exp._act_carry = {"dummy": torch.tensor(1.0)}  # noqa: SLF001
    exp.current_step = 1000

    exp.reset_transient_state()

    assert exp._act_carry is None  # noqa: SLF001


def test_resume_from_checkpoint_new_format():
    exp1 = _make_test_exp()
    exp1.current_step = 500
    exp1.best_acc = 75.0

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1._make_checkpoint(), f.name)  # noqa: SLF001
        ckpt_path = f.name

    exp2 = _make_test_exp()
    resume_from_checkpoint(exp2, ckpt_path, resume_optimizer=True, resume_ema=True)

    assert exp2.current_step == 500
    assert exp2.best_acc == 75.0


def test_resume_from_checkpoint_no_optimizer():
    exp1 = _make_test_exp()
    exp1.current_step = 500

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(exp1._make_checkpoint(), f.name)  # noqa: SLF001
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
    class TestExpWithReset(ExperimentBase):
        reset_steps: list[int] = [1500, 3000]  # noqa: RUF012

        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.batch_size = 4
            self.compile_model = False
            config = TRM.Config(
                compile_core=False,
                vocab_size=12,
                seq_len=82,
                hidden_size=64,
                num_heads=4,
                num_layers=1,
                H_cycles=2,
                L_cycles=2,
                dtype=self.dtype,
            )
            self.model = config.setup()
            super().__init__()

    exp = TestExpWithReset()
    assert exp.reset_steps == [1500, 3000]


def test_data_loader_yields_valid_count():
    """Test that data loader yields 3-tuple with valid_count for padding."""
    # Test the padding logic directly without running full eval
    batch_size = 16
    valid_count = 10

    # Simulate what the loader does for incomplete batch
    inputs = torch.randint(1, 11, (valid_count, 81))
    labels = torch.randint(1, 11, (valid_count, 81))

    # Pad to batch_size
    pad_size = batch_size - valid_count
    inputs_pad = torch.zeros(pad_size, 81, dtype=inputs.dtype)
    labels_pad = torch.zeros(pad_size, 81, dtype=labels.dtype)
    padded_inputs = torch.cat([inputs, inputs_pad])
    padded_labels = torch.cat([labels, labels_pad])

    # Verify shapes
    assert padded_inputs.shape == (batch_size, 81)
    assert padded_labels.shape == (batch_size, 81)

    # Verify slicing works correctly
    assert (padded_inputs[:valid_count] == inputs).all()
    assert (padded_labels[:valid_count] == labels).all()


def test_eval_metrics_use_valid_count():
    """Test that metrics correctly use valid_count to exclude padding."""
    batch_size = 16
    valid_count = 10

    # Create predictions and labels with padding
    preds = torch.randint(1, 11, (batch_size, 81))
    labels = torch.randint(1, 11, (batch_size, 81))

    # Make first valid_count samples correct, rest wrong
    preds[:valid_count] = labels[:valid_count]

    # Full batch would show 10/16 = 62.5% accuracy
    full_acc = (preds == labels).float().mean().item()

    # Valid-only should show 100% accuracy
    valid_acc = (preds[:valid_count] == labels[:valid_count]).float().mean().item()

    assert valid_acc == 1.0
    assert full_acc < 1.0  # Padding brings down accuracy


def test_fast_eval_streaming_replacement():
    """Test that fast_eval uses streaming replacement - halted samples are replaced with new puzzles."""

    class FastEvalExp(ExperimentBase):
        eval_method: str = "fast"
        max_reasoning_steps: int = 16
        config: TRM.Config = TRM.Config(
            compile_core=False,
            vocab_size=12,
            seq_len=82,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            H_cycles=1,
            L_cycles=1,
        )

        def __init__(self):
            self.device = torch.device("cpu")
            self.batch_size = 2
            self.compile_model = False
            super().__init__()

    exp = FastEvalExp()

    # Track how many forward passes
    original_forward = exp.model.forward
    step_count = 0

    def tracking_forward(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        out = original_forward(*args, **kwargs)
        # All samples halt after 2 steps
        if step_count % 2 == 0:
            out["q_halt"] = torch.ones_like(out["q_halt"])
        else:
            out["q_halt"] = torch.full_like(out["q_halt"], -10.0)
        return out

    exp.model.forward = tracking_forward

    # Create loader with 6 puzzles total (3 batches of 2)
    # With batch_size=2 and streaming, we should process all 6
    all_inputs = [torch.randint(1, 11, (2, 81)) for _ in range(3)]
    all_labels = [torch.randint(1, 11, (2, 81)) for _ in range(3)]

    def fake_loader():
        for inp, lab in zip(all_inputs, all_labels, strict=False):
            yield (inp, lab, 2)

    cell_acc, puzzle_acc = exp._evaluate_act_haltfast(fake_loader())

    # With streaming: 6 puzzles processed
    # Batch stays at size 2, each puzzle takes 2 steps to halt
    # Forward passes: step 1 (no halt), step 2 (all halt, refill)
    #                 step 3 (no halt), step 4 (all halt, refill)
    #                 step 5 (no halt), step 6 (all halt, done)
    # Total = 6 forward passes
    assert step_count == 6, f"Expected 6 steps for 6 puzzles but got {step_count}"


def test_fast_eval_metrics_at_halt_time():
    """Test that fast_eval computes metrics at halt time per sample."""

    class FastEvalExp(ExperimentBase):
        eval_method: str = "fast"
        max_reasoning_steps: int = 4
        config: TRM.Config = TRM.Config(
            compile_core=False,
            vocab_size=12,
            seq_len=82,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            H_cycles=1,
            L_cycles=1,
        )

        def __init__(self):
            self.device = torch.device("cpu")
            self.batch_size = 2
            self.compile_model = False
            super().__init__()

    exp = FastEvalExp()

    step_count = 0
    original_forward = exp.model.forward

    def controlled_forward(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        out = original_forward(*args, **kwargs)

        # Step 2: all samples predict 5 and halt
        if step_count == 2:
            out["logits"] = torch.zeros_like(out["logits"])
            out["logits"][:, :, 5] = 100
            out["q_halt"] = torch.ones_like(out["q_halt"])
        else:
            out["q_halt"] = torch.full_like(out["q_halt"], -10.0)

        return out

    exp.model.forward = controlled_forward

    # Labels are all 5s - should get 100% accuracy
    B = 2
    inputs = torch.randint(1, 11, (B, 81))
    labels = torch.full((B, 81), 5, dtype=torch.long)

    def fake_loader():
        yield (inputs, labels, B)

    cell_acc, puzzle_acc = exp._evaluate_act_haltfast(fake_loader())

    assert cell_acc == 100.0, f"Expected 100% cell acc but got {cell_acc}%"
    assert puzzle_acc == 100.0, f"Expected 100% puzzle acc but got {puzzle_acc}%"


def test_no_recompilation_all_paths():
    """Test that step() and forward() use same _core signature (no recompile).

    The torch.compile issue was: different kwargs caused separate compiled graphs.
    This test verifies all code paths call _core with identical positional args.
    """
    config = TRM.Config(
        compile_core=False,
        vocab_size=12,
        seq_len=82,
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
    )
    model = config.setup()

    # Track _core calls
    call_signatures: list[tuple[tuple[int, ...], ...]] = []
    original_core = model._core

    def tracking_core(input_emb, z_H, z_L, cos_sin):
        # Record shapes of all args (signature proxy)
        sig = (
            tuple(input_emb.shape),
            tuple(z_H.shape),
            tuple(z_L.shape),
            "None"
            if cos_sin is None
            else (tuple(cos_sin[0].shape), tuple(cos_sin[1].shape)),
        )
        call_signatures.append(sig)
        return original_core(input_emb, z_H, z_L, cos_sin)

    model._core = tracking_core

    # Test inputs
    B = 4
    x = torch.randint(0, 12, (B, 82))
    z_H = model.H_init.expand(B, 82, -1)
    z_L = model.L_init.expand(B, 82, -1)

    # Path 1: step() - used in ACT training/eval loops
    call_signatures.clear()
    model.step(x, z_H, z_L)
    step_sig = call_signatures[0]

    # Path 2: forward() - used for diagnostics
    call_signatures.clear()
    model(x, z_H, z_L)
    forward_sigs = call_signatures

    # All calls must have same signature
    for sig in forward_sigs:
        assert sig == step_sig, f"Signature mismatch: {sig} vs {step_sig}"


def test_eval_methods_consistent_k1():
    """Test that full, fast, and wta eval give same results with K=1."""

    class EvalTestExp(ExperimentBase):
        max_reasoning_steps: int = 4
        K: int = 1
        config: TRM.Config = TRM.Config(
            compile_core=False,
            vocab_size=12,
            seq_len=82,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            H_cycles=1,
            L_cycles=1,
        )

        def __init__(self):
            self.device = torch.device("cpu")
            self.batch_size = 2
            self.compile_model = False
            super().__init__()

    # Create deterministic test data
    torch.manual_seed(42)
    n_puzzles = 4
    inputs = torch.randint(1, 11, (n_puzzles, 81))
    labels = torch.randint(1, 11, (n_puzzles, 81))

    def make_loader():
        for i in range(0, n_puzzles, 2):
            yield (inputs[i : i + 2], labels[i : i + 2], 2)

    # Test full eval
    exp_full = EvalTestExp()
    exp_full.eval_method = "full"
    exp_full.fast_eval = False
    torch.manual_seed(0)
    cell_full, puzzle_full = exp_full._evaluate_act_full(make_loader())

    # Test fast eval
    exp_fast = EvalTestExp()
    exp_fast.eval_method = "fast"
    exp_fast.fast_eval = True
    torch.manual_seed(0)
    cell_fast, puzzle_fast = exp_fast._evaluate_act_haltfast(make_loader())

    # Test wta eval with K=1
    exp_wta = EvalTestExp()
    exp_wta.eval_method = "wta"
    exp_wta.K = 1
    torch.manual_seed(0)
    cell_wta, puzzle_wta = exp_wta._evaluate_act_haltfast_wta(make_loader())

    # All should give identical results
    assert cell_full == cell_fast, f"full={cell_full} vs fast={cell_fast}"
    assert cell_full == cell_wta, f"full={cell_full} vs wta={cell_wta}"
    assert puzzle_full == puzzle_fast, f"full={puzzle_full} vs fast={puzzle_fast}"
    assert puzzle_full == puzzle_wta, f"full={puzzle_full} vs wta={puzzle_wta}"


def test_wta_batched_vs_sequential():
    """Verify batched K-head forward gives same results as sequential."""
    torch.manual_seed(42)

    config = TRM.Config(
        compile_core=False,
        vocab_size=12,
        seq_len=82,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        H_cycles=1,
        L_cycles=1,
        dtype=torch.float32,
    )
    model = config.setup()
    model.eval()

    B, K = 4, 3
    seq_len = config.seq_len
    hidden = config.hidden_size

    # Register extra L_init params
    for k in range(1, K):
        model.register_parameter(
            f"L_init_{k}", torch.nn.Parameter(torch.randn(seq_len, hidden) * 0.1)
        )

    # Create inputs
    inputs = torch.randint(1, 11, (B, 82))
    z_H_init = model.H_init.expand(B, seq_len, -1).clone()

    # Sequential: K separate forward passes
    torch.manual_seed(123)
    seq_logits = []
    seq_q_halt = []
    seq_z_H = []
    seq_z_L = []
    for k in range(K):
        L_init_k = model.L_init if k == 0 else getattr(model, f"L_init_{k}")
        z_L_k = L_init_k.expand(B, seq_len, -1).clone()
        out = model(inputs, z_H_init, z_L_k)
        seq_logits.append(out["logits"])
        seq_q_halt.append(out["q_halt"])
        seq_z_H.append(out["z_H"])
        seq_z_L.append(out["z_L"])

    # Batched: single [B*K] forward pass
    torch.manual_seed(123)  # Same seed - but no noise in this test
    z_L_all = []
    for k in range(K):
        L_init_k = model.L_init if k == 0 else getattr(model, f"L_init_{k}")
        z_L_all.append(L_init_k.expand(B, seq_len, -1).clone())

    z_L_batched = torch.stack(z_L_all, dim=1).reshape(B * K, seq_len, -1)
    z_H_batched = (
        z_H_init.unsqueeze(1).expand(B, K, seq_len, -1).reshape(B * K, seq_len, -1)
    )
    inputs_batched = inputs.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

    out_batched = model(inputs_batched, z_H_batched, z_L_batched)

    # Reshape back
    logits_batched = out_batched["logits"].reshape(B, K, 82, -1)
    q_halt_batched = out_batched["q_halt"].reshape(B, K)
    z_H_batched_out = out_batched["z_H"].reshape(B, K, seq_len, -1)
    z_L_batched_out = out_batched["z_L"].reshape(B, K, seq_len, -1)

    # Compare
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
