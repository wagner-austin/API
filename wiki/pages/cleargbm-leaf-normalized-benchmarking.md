---
title: ClearGBM vs LightGBM — leaf-normalized benchmarking
tags: [ml, cleargbm, performance, benchmarks, measurement]
related:
  - "[[cleargbm-perf-experiments-2026-07-21]]"
  - "[[cleargbm-perf-leaf-wise-growth]]"
  - "[[cleargbm-histogram-split-path]]"
sources:
  - libs/covenant_ml/src/covenant_ml/benchmarking/
  - libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py
  - libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json
fact_checked: "2026-07-24"
confidence: high
---

# ClearGBM vs LightGBM — leaf-normalized benchmarking

A wall-clock ratio between ClearGBM and LightGBM is not a statement about implementation quality, because the two learners do not build the same amount of work at the same configuration. ClearGBM grows **depth-wise**, bounded by `max_depth`; LightGBM grows **leaf-wise** and stops at `num_leaves`. At the benchmark's `max_depth=6, num_leaves=31` the two differ in tree size by roughly half again, so a raw ratio conflates "slower per unit of work" with "doing more work per tree".

Depth-wise is **not** the same as building a *full* tree, and the table below is the evidence: a full binary tree at `max_depth=6` has 64 leaves, but ClearGBM measures 57.9 there (47.15 on the authoritative run). Stopping criteria — `min_samples_split`, `min_samples_leaf`, and the absence of a positive-gain split — retire branches early, so leaves sit at a range of depths. See [[cleargbm-perf-leaf-wise-growth]] § "Interpretability cost" for why this matters to the interpretability argument.

## Measured tree-size divergence

Leaf counts per tree, read from each model's own serialization (20 trees, bankruptcy dataset):

| max_depth | ClearGBM leaves | LightGBM leaves | ratio |
|---|---|---|---|
| 3 | 8.0 | 8.0 | 1.00× |
| 4 | 16.0 | 15.8 | 1.01× |
| 5 | 31.2 | 30.6 | 1.02× |
| **6** | **57.9** | **31.0** | **1.87×** |

Below depth 5 the `num_leaves` cap does not bind and the two build identically-sized trees. The divergence appears only at the benchmark's operating point.

## Authoritative measurement (2026-07-24)

78,682 rows × 18 features, 200 trees, depth 6, `max_bins=64`, `n_jobs=1`, median of 5 timed fits × 3 seeds[^1]:

| | fit_time | leaves/tree | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| lightgbm | 0.8981s ± 0.0719 | 30.96 | 0.6881 | 0.1366 |
| cleargbm | 1.2809s ± 0.0638 | 47.15 | 0.6821 | 0.1414 |

- raw ratio **1.426×**
- leaf ratio **1.523×**
- **per-leaf ratio 0.937× — ClearGBM is ~6% faster than LightGBM at equal tree size.**

Quality is a statistical tie; ClearGBM leads on AUC-PR, the metric that matters on a 6.6%-positive class.

## Three protocol properties, each fixing a past wrong conclusion

1. **Both learners measured in the same run.** LightGBM's measured fit time has ranged 0.69s → 0.95s across sessions *with identical LightGBM code*, purely from machine conditions. Dividing a fresh ClearGBM number by a LightGBM number carried forward from an older manifest manufactures a gap that is not there — this produced a reported "1.40× gap" on 2026-07-24 when the contemporaneous gap was near parity.
2. **Canonical statistic is the median, not the minimum.** The first fits after an idle period run with full turbo headroom — a different power regime, not noise. Taking a minimum let one cold-start outlier (LightGBM seed 42: `min/med = 0.486/0.828`) set the canonical number and produced an uninterpretable 1.751× ± 0.538; the median over the same data gave 1.43× ± 0.02.
3. **Results normalized by tree size**, per the divergence above.

## Where it lives

`libs/covenant_ml/src/covenant_ml/benchmarking/` — layered: `types` (records + codecs), `protocols` (injected boundaries), `timing`/`splitting`/`quality`/`model_shape`/`reporting` (pure), `dataset` (only file I/O), `adapters` (both learners), `runner` (protocol), `factory` (sole namer of concretes). Entry point `scripts/benchmark_cleargbm_vs_lightgbm.py`.

It lives in `covenant_ml`, not `cleargbm`, because `covenant_ml` already depends on both learners; `cleargbm` declares only numpy and would otherwise take a dependency on its own competitor. An earlier harness lived only in a session scratchpad and was lost, after which every downstream perf claim was measured on a noisier shape and silently became incomparable.

[^1]: `libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json` — dataset sha256 `cff2c899a97ecd62…`, seeds 42/43/44, `repeats=5`, `warmups=2`, estimator `median_of_repeats_per_seed`.
