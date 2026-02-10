"""x000k: Match gaoxin492/TinyRecursiveModels maze config.

Changes from x000f (keeping our Muon+AdamW optimizer and standard CE loss):
  - label_smoothing: 0.2 -> 0.0 (reference uses no label smoothing)
  - L_cycles: 6 -> 4 (reference maze override)
  - ema_decay: 0.9 -> 0.999 (reference ema_rate)
  - ema_warmup_steps: 5000 -> 0 (reference applies EMA from step 0)
  - q_halt_weight: 0.05 -> 0.5 (reference: lm_loss + 0.5 * q_halt_loss)
  - lr_warmup_steps: 0 -> 2000 (reference lr_warmup_steps)
  - lr_min_ratio: 0.0 -> 1.0 (reference: constant LR after warmup)
  - grad_clip_max_norm: 1.0 -> None (reference: no gradient clipping)
  - seed: 42 -> 0 (reference seed)
  - num_puzzle_id_tokens: 0 -> 1, num_register_tokens: 0 -> 15 (reference puzzle_emb_len=16)
  - register_token_init_std: 1.0 -> 0.0 (reference uses zero-init)
  - loss_ignore_index: None -> 0 (reference ignores label=0 positions)
  - vocab_size: 12 -> 6 (reference charset "# SGo" + pad = 6)
  - register_tokens_learnable: True -> False (reference uses fixed zero-padding)
  - loss_sum_normalize: False -> True (reference uses sum then /global_batch_size)

Remaining known discrepancies vs reference (intentional or not yet addressed):

  Optimizer & related:
  - Optimizer: we use Muon+AdamW; reference uses AdamATan2 (INTENTIONAL)
  - SwiGLU activation: we use muon-modified sigmoid(gate)*norm(gate*x);
    reference uses standard silu(gate)*up (INTENTIONAL, consequence of Muon)
  - Attention o_norm: we add RMSNorm before o_proj (attn_muon_modified);
    reference does not (INTENTIONAL, consequence of Muon)
  - QKV weight shape: we use 3D EnsembleLinear (per-head orthogonalization by Muon);
    reference uses flat 2D CastedLinear (INTENTIONAL, consequence of Muon)
  - Weight decay (Muon params): we use ~0.02; reference uses 1.0 everywhere (UNCERTAIN)
  - Puzzle embedding optimizer: we use AdamW; reference uses SignSGD
    (low impact: only 1 embedding vector for maze)

  Architecture:
  - MLP hidden dim: we use multiple_of=128 -> 1408;
    reference uses multiple_of=256 -> 1536
  - lm_head bias: we use bias=True (head_bias default);
    reference uses bias=False
  - q_head output dim: we output 1 (halt only);
    reference outputs 2 (halt + continue, but continue loss=0 with no_ACT_continue)

  Loss & gradients:
  - Loss function: we use standard cross_entropy;
    reference uses stablemax_cross_entropy (INTENTIONAL)

  Training:
  - Batch size: we use 128 on 1 GPU; reference uses 768 across 4 GPUs
  - Total training steps: we use 8000; reference uses ~65K (50K epochs)
  - Augmentations: reference build_maze_dataset.py defaults aug=False, but README
    comment says "8 augments". Unclear which was used for reported results.

  Eval:
  - Eval halting: we halt early when q_halt>0; reference always runs max steps

  Precision:
  - Weight dtype: reference creates all weights in float32, casts at forward time
    via CastedLinear/CastedEmbedding; we create some weights in bfloat16 natively
  - q_halt precision: reference explicitly casts q_halt logits to float32;
    ours now promotes via autocast around loss computation (FIXED)
"""

from typing import cast

import dataclasses
import functools

from experiment import main
from maze.x000 import Experiment as Experiment000
from model import TRM3, TransformerBlock, TRM3ConfigProtocol, trunc_normal_init_


class Experiment(Experiment000):
    label_smoothing: float = 0.0

    use_ema: bool = False
    ema_decay: float = 0.999
    ema_warmup_steps: int = 0

    q_halt_weight: float = 0.5
    lr_warmup_steps: int = 2000
    lr_min_ratio: float = 1.0
    grad_clip_max_norm: float | None = None
    loss_ignore_index: int | None = 0
    cast_model_to_dtype: bool = False
    loss_sum_normalize: bool = True

    config: TRM3ConfigProtocol = dataclasses.replace(
        cast(TRM3.Config, Experiment000.config),
        vocab_size=6,
        L_cycles=4,
        num_puzzle_id_tokens=1,
        num_puzzle_ids=1,
        num_register_tokens=15,
        register_token_init_std=0.0,
        register_tokens_learnable=False,
        q_halt_seq_index=0,
        cast_model_to_dtype=False,
        block_fn=functools.partial(
            TransformerBlock,
            multiple_of=128,
            num_heads=8,
            mlp_init_weight_fn=trunc_normal_init_,
            attn_init_weight_fn=trunc_normal_init_,
            attn_muon_modified=True,
            attn_checkpoint_muon_norm=True,
        ),
    )


if __name__ == "__main__":
    main(Experiment())
