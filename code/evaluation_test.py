"""Unit tests for evaluation.py."""

from __future__ import annotations

import torch

from research.projects.trm3.evaluation import (
    HALT_TOKEN_ID,
    cells_fixed_broken,
    evaluate_multistart,
    evaluate_single_start,
    h_cosine_similarities,
    per_h_accuracy,
    prepend_halt,
    run_act_steps,
    z_h_deltas,
)
from research.projects.trm3.model import TRM


def test_prepend_halt():
    inputs = torch.randint(0, 10, (2, 81))
    device = inputs.device
    out = prepend_halt(inputs, device)
    assert out.shape == (2, 82)
    assert (out[:, 0] == HALT_TOKEN_ID).all()
    assert torch.equal(out[:, 1:], inputs)


def test_run_act_steps():
    config = TRM.Config(
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
        compile_core=False,
    )
    model = config.setup()
    model.eval()

    device = torch.device("cpu")
    dtype = torch.float32
    inputs = torch.randint(0, 10, (2, 81))
    z_H = model.H_init.expand(2, 82, -1)
    z_L = model.L_init.expand(2, 82, -1)

    with torch.no_grad():
        preds, q_halt, z_H_out = run_act_steps(
            model, inputs, z_H, z_L, max_steps=2, device=device, dtype=dtype
        )

    assert preds.shape == (2, 81)
    assert q_halt.shape == (2,)
    assert z_H_out.shape == (2, 82, 64)


def test_evaluate_single_start():
    config = TRM.Config(
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
        compile_core=False,
    )
    model = config.setup()
    model.eval()

    inputs = torch.randint(0, 10, (4, 81))
    labels = torch.randint(1, 10, (4, 81))

    cell_acc, puzzle_acc = evaluate_single_start(
        model,
        inputs,
        labels,
        max_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert 0 <= cell_acc <= 100
    assert 0 <= puzzle_acc <= 100


def test_evaluate_multistart():
    config = TRM.Config(
        hidden_size=64,
        num_heads=4,
        num_layers=1,
        H_cycles=2,
        L_cycles=2,
        compile_core=False,
    )
    model = config.setup()
    model.eval()

    inputs = torch.randint(0, 10, (4, 81))
    labels = torch.randint(1, 10, (4, 81))

    cell_acc, puzzle_acc = evaluate_multistart(
        model,
        inputs,
        labels,
        max_steps=2,
        n_starts=3,
        z_L_noise=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert 0 <= cell_acc <= 100
    assert 0 <= puzzle_acc <= 100


def test_per_h_accuracy():
    all_logits = [torch.randn(4, 81, 10) for _ in range(3)]
    labels = torch.randint(0, 10, (4, 81))
    accs = per_h_accuracy(all_logits, labels)
    assert len(accs) == 3
    assert all(0 <= a <= 100 for a in accs)


def test_h_cosine_similarities():
    all_logits = [torch.randn(4, 81, 10) for _ in range(3)]
    cos_sims = h_cosine_similarities(all_logits)
    assert len(cos_sims) == 2
    assert all(-1 <= c <= 1 for c in cos_sims)


def test_z_h_deltas():
    z_H = [torch.randn(4, 82, 64) for _ in range(3)]
    deltas = z_h_deltas(z_H)
    assert len(deltas) == 2
    assert all(d >= 0 for d in deltas)


def test_cells_fixed_broken():
    labels = torch.tensor([[1, 2, 3]])
    all_logits = []

    # H0: predict [0, 0, 0] - all wrong
    l0 = torch.zeros(1, 3, 10)
    l0[0, :, 0] = 10.0
    all_logits.append(l0)

    # H1: predict [1, 2, 0] - 2 fixed
    l1 = torch.zeros(1, 3, 10)
    l1[0, 0, 1] = 10.0
    l1[0, 1, 2] = 10.0
    l1[0, 2, 0] = 10.0
    all_logits.append(l1)

    # H2: predict [0, 2, 3] - 1 fixed, 1 broken
    l2 = torch.zeros(1, 3, 10)
    l2[0, 0, 0] = 10.0
    l2[0, 1, 2] = 10.0
    l2[0, 2, 3] = 10.0
    all_logits.append(l2)

    fixed, broken = cells_fixed_broken(all_logits, labels)
    assert fixed == [2, 1]
    assert broken == [0, 1]


if __name__ == "__main__":
    from research.projects.trm3.util import test_main

    test_main(__file__)
