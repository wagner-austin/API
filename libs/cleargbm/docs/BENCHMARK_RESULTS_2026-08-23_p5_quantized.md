# P5 final landing — quantized training (2026-08-23)

`quantized_gradient_bins` lands as config serde field 24 (required-with-null;
even, in [2, 126]; single-score objectives only; exclusive with
`categorical_features`). The implementation is LightGBM's shipped shape
(`gradient_discretizer.cpp` + `serial_tree_learner.cpp` +
`feature_histogram.hpp` @ 3ec5b99b, pinned in the tech-wiki): per-round
global-max scales (`max|g| / (bins/2)`, `max h / bins`), stochastic rounding
from pre-generated per-row randoms with a per-round rotation offset, one
interleaved `int8` stream (hessian at `2i`, gradient at `2i + 1`), per-node
16/32-bit packed integer histograms at the `count x bins < 65536` threshold,
packed sibling subtraction with mixed-width dispatch, and an exact integer
prefix scan whose sums convert to f64 only at the candidate boundary, into
the shared gain formula.

Stated divergences, all deliberate: rounding randoms and rotation offsets
are pure functions of `(random_state, global round)` — which makes split
training EXACT under quantization (3 + 3 rounds equals a fresh 6-round run
bit for bit, held by tests) where LightGBM's stateful `mt19937` stream could
not resume; quantized values clamp to the stated range where LightGBM lets
an fp epsilon write one value past it; there is no constant-hessian special
case (under unit hessians the general scaling is exact anyway); stochastic
rounding is always on and leaf renewal is structural — ClearGBM's leaf
values ALWAYS come from the original float gradients (`compute_sums`), so
quantization can only ever change which splits are chosen, which is
LightGBM's `quant_train_renew_leaf=true` semantics by construction.

## Round-6 artifact retrain (config field 24)

| artifact | expected | reproduced |
|---|---|---|
| rw_matches `active_cgbm.json` | val 0.7790 / test 0.7142 / best 16 / spw 1.655 | identical |
| taiwan `taiwan_cleargbm_model.json` | val 0.9451 / 98 trees | identical |
| us `us_cleargbm_model.json` | val 0.7848 / 14 trees | identical |

## Identity gate

Four arms x seeds 42-45 reproduce the knob-identity manifest **112/112**
byte-for-byte with quantization off
(`BENCHMARK_MANIFEST_2026-08-23_p5_quantized_identity.json`).

## The measured verdict: a quality knob, not (yet) a speed lever

New harness: `covenant_ml.benchmarking.quantized_quality` +
`scripts.benchmark_cleargbm_quantized` — four arms per seed (each library
with float and quantized histograms at the same 4 gradient bins), quality on
a held-out quarter plus per-arm fit wall clock, single-threaded.

**Quality** (20000 x 8 noisy-logistic corpus, 200 trees depth 4): quantized
training IMPROVES held-out AUC for BOTH libraries on this noisy corpus —
stochastic rounding acts as a regularizer. ClearGBM's mean AUC delta is
**+0.0029** (positive on all four seeds); LightGBM's own is **+0.0016**.

| seed | cleargbm float | cleargbm quant | lightgbm float | lightgbm quant |
|---|---|---|---|---|
| 42 | 0.795201 | 0.798230 | 0.796829 | 0.799242 |
| 43 | 0.787940 | 0.790742 | 0.787530 | 0.791295 |
| 44 | 0.795337 | 0.798782 | 0.795461 | 0.795313 |
| 45 | 0.799012 | 0.801509 | 0.799182 | 0.799472 |

**Speed** (this machine, single-threaded, wall clock in the manifests): the
paper's ~2x does NOT materialize at these scales — for either library.
ClearGBM's quantized arm costs ~1.4x its own float path at 20000 rows and
~1.1-1.5x at 200000 x 16 depth 8; LightGBM's own quantized arm is ~1.2x
SLOWER at 20000 rows and only reaches ~0.95x (break-even) at 200000. The
per-round full-array discretization is a fixed cost that small corpora never
amortize, and the integer histogram's cache advantage needs SIMD-width
accumulation and many-core construction to become the headline 2x — Shi
2022's benchmarks are many-million-row, many-thread runs. Measured, so it
did happen: at ClearGBM's current corpus scales the knob's value is the
regularization, and the honest speed story is a cost, recorded in
`BENCHMARK_MANIFEST_2026-08-23_p5_quantized_quality.json` (default corpus)
and `BENCHMARK_MANIFEST_2026-08-23_p5_quantized_large.json` (200k corpus).
`fit_seconds` values are wall clocks of THIS run's environment, like the
identity manifests' timing columns; the quality values are deterministic and
reproduced byte-for-byte across reruns.

## Gates at landing

- cleargbm_rs: 1676 tests, 100.00% segment coverage, clippy clean.
- cleargbm: 265 tests, 100.00% (config field + decode validation + surface).
- covenant_ml: 2489 tests, 100.00% (quantized harness + CLI).
- covenant-radar-api: 2588 tests, 100.00%, zero source changes.

## EFB: formally excluded from P5

Exclusive feature bundling remains EXCLUDED this phase, as recommended when
the P5 research landed: its habitat is sparse one-hot data the dataset
registry lacks, and its hardcoded `total/10000` conflict budget would need a
knob the constitution refuses to invent without a corpus that names it. The
tech-wiki page `lightgbm-efb-bundling-implementation` holds the code-level
map for whenever a real corpus arrives.
