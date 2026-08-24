---
title: ClearGBM quantized training — packed integers, measured honestly
tags: [ml, cleargbm, quantization, roadmap-p5]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-continuation-and-goss]]"
  - "[[cleargbm-histogram-split-path]]"
source_paths:
  - libs/cleargbm_rs/src/training/quantize.rs
  - libs/cleargbm_rs/src/histogram/quantized.rs
  - libs/cleargbm_rs/src/split/threshold_quantized.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p5_quantized.md
fact_checked: "2026-08-23"
confidence: high
hubs: [libs]
---

# ClearGBM quantized training — packed integers, measured honestly

The final P5 landing of the [[cleargbm-program-charter]] (board task
`0c320137`). `quantized_gradient_bins` is config serde field 24
(required-with-null; even, in [2, 126] so values pack into int8;
single-score objectives only; exclusive with categorical features —
triggering artifact retrain round 6, all numbers exact).

## The shipped semantics, with determinism strengthened

The pass is LightGBM's CPU implementation of Shi 2022 @ 3ec5b99b (seven
tech-wiki pages pin the map): per-round scales from a global max scan
(`max|g|/(bins/2)`, `max h/bins`), stochastic rounding into one
interleaved int8 stream (hessian at 2i, gradient at 2i+1), per-node
16/32-bit packed histograms selected at `count x bins < 65536`, packed
sibling subtraction with mixed-width dispatch, and an exact integer
prefix scan converting to f64 only at the candidate boundary into the
shared gain formula. Stated divergences: the rounding randoms and the
per-round rotation offset are pure functions of `(random_state, global
round)` — so split training stays EXACT under quantization (3+3 == 6
bit for bit, tested), which LightGBM's stateful mt19937 stream cannot
offer; quantized values clamp to the stated range; no constant-hessian
special case; stochastic rounding always on. Leaf renewal
(`quant_train_renew_leaf`) is structural rather than a knob: ClearGBM
computes leaf values from the original float gradients in every mode,
so quantization only ever affects which splits are chosen.

## Measured, so it did happen — and the 2x didn't

Off is bit-identical history (identity 112/112,
`BENCHMARK_MANIFEST_2026-08-23_p5_quantized_identity.json`). On, at 4
gradient bins on the noisy-logistic corpus: held-out AUC IMPROVES for
both libraries (ClearGBM mean +0.0029 over all four seeds, LightGBM's
own +0.0016) — stochastic rounding regularizes. But the headline ~2x
speedup does not materialize single-threaded at 20k-200k rows for
EITHER library: ClearGBM's quantized arm costs ~1.1-1.5x its own float
path, and LightGBM's own only reaches break-even at 200k rows. The
per-round discretization is a fixed cost small corpora never amortize;
the paper's 2x lives at many-million-row, many-thread scale. The knob's
honest value today is regularization, not speed — recorded with both
manifests in `BENCHMARK_RESULTS_2026-08-23_p5_quantized.md`, alongside
the negative results this wiki already keeps (the f32 narrowing revert,
the SIMD-histogram non-recommendation).

## Where things live

Rust: `training/quantize.rs` (discretizer), `histogram/quantized.rs`
(packed accumulators, width selection, subtraction),
`split/threshold_quantized.rs` (integer scan), dispatched through
`tree/histograms.rs`'s `NodeHistograms` enum whose float arm is the
historical code operation for operation. Python: the config field plus
decode validation. Harness: `covenant_ml.benchmarking.quantized_quality`
+ `scripts.benchmark_cleargbm_quantized` (quality + per-arm fit wall
clock). P5 closes with EFB formally excluded: sparse one-hot habitat
absent from the registry, and a hardcoded 1/10000 conflict budget the
constitution refuses to turn into an uninvented knob.
