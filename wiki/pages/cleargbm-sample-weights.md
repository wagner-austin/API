---
title: ClearGBM sample weights — the class weight becomes the special case
tags: [ml, cleargbm, sample-weights, roadmap-p2]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-objective-seam]]"
  - "[[cleargbm-decorative-knob-class]]"
source_paths:
  - libs/cleargbm_rs/src/training/train.rs
  - libs/cleargbm_rs/src/training/labels.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p2_sample_weights.md
fact_checked: "2026-08-23"
confidence: high
hubs: [libs]
---

# ClearGBM sample weights — the class weight becomes the special case

P2 of the [[cleargbm-program-charter]] roadmap, landed 2026-08-23 (board
task `f8d93064`). Both objectives accept optional per-row weights:
gradients/hessians scale by the effective row weight, the base score and
both evaluation losses are weight-averaged, and validation splits carry
their own optional evaluation weights. Weights are DATA, not config —
nothing entered the config or the model JSON, so no artifact retraining
and every recorded manifest stays valid.

## The three identity claims, each pinned by a test

1. **`None` is bit-identical to history.** The weightless arms keep the
   exact historical expressions (no synthesized `* 1.0`; the base score
   keeps its closed-form `spw * count`). Corpus-level proof: the four-arm
   benchmark reproduces the knob-identity manifest 56/56 byte-for-byte
   (`BENCHMARK_MANIFEST_2026-08-23_p2_weight_identity.json`).
2. **All-ones is bit-identical to `None`** — IEEE multiply-by-1.0 plus
   exact integer-valued weight sums.
3. **`scale_pos_weight` is the derived special case**: spw=3.0 via config
   equals w=3.0-for-positives via weights (spw=1.0), bit for bit — proven
   at an integer-valued weight where the closed-form multiply and per-row
   accumulation provably coincide. The effective weight factorization
   (`eff = class_term * w_i`) is what makes both this and claim 1 true at
   once.

## Semantics

Rows carrying 50× weight fit visibly closer (|err| ~0.005 vs 0.12–0.21
on the alternating-target fixture). Zero, negative, non-finite and
wrong-length weights are rejected naming the offending index — a zero
weight can empty a leaf's hessian sum (0/0 at reg_lambda 0), so dropping
a row must be an explicit act, not a weight. A validation weight without
a validation split is likewise rejected.

## Surface

- Core: `train_gradient_boosting(x, labels, sample_weight, validation,
  config, names, runtime)`; `ValidationData { x, y, weight }`;
  `ResolvedValidation<Y>` carries the narrowed val split.
- pyo3/Python: 8-argument wire layout; `sample_weight=` /
  `val_sample_weight=` keyword-only defaults on
  `train_gradient_boosting` and `train_gradient_boosting_regression` —
  a data default that cannot silently change semantics.
- covenant_ml: unchanged (weights default off); backend-level exposure
  waits for the first genuinely weighted corpus (P6 science data).

This is the substrate GOSS (P5) and LambdaMART (P4) consume: both are
weighted-gradient algorithms.
