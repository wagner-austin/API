---
title: ClearGBM vs LightGBM — leaf-normalized benchmarking
tags: [ml, cleargbm, performance, benchmarks, measurement]
related:
  - "[[cleargbm-perf-experiments-2026-07-21]]"
  - "[[cleargbm-perf-leaf-wise-growth]]"
  - "[[cleargbm-histogram-split-path]]"
source_paths:
  - libs/covenant_ml/src/covenant_ml/benchmarking
  - libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py
  - libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/benchmarking": 3bfdf67a0d9c5510b8c41515ebd7514d4a4faaf3
  "libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py": ce22a0c0a3ad39c64cd2d5370143e3d6f8f1d506
  "libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json": e3e661369727af7b1d94feed04e06aff4374376e
fact_checked: "2026-08-14"
confidence: high
hubs: [libs]
---

# ClearGBM vs LightGBM — leaf-normalized benchmarking

A wall-clock ratio between ClearGBM and LightGBM is not a statement about implementation quality, because the two learners do not build the same amount of work at the same configuration. ClearGBM grows **depth-wise**, bounded by `max_depth`; LightGBM grows **leaf-wise** and stops at `num_leaves`. At the benchmark's `max_depth=6, num_leaves=31`[^2] the two differ in tree size by roughly half again[^1], so a raw ratio conflates "slower per unit of work" with "doing more work per tree"[^3].

Depth-wise is **not** the same as building a *full* tree, and the table below is the evidence: a full binary tree at `max_depth=6` has 64 leaves, but ClearGBM measures 57.9 there (47.15 on the authoritative run). Stopping criteria — `min_samples_split`, `min_samples_leaf`, and the absence of a positive-gain split — retire branches early, so leaves sit at a range of depths. See [[cleargbm-perf-leaf-wise-growth]] § "Interpretability cost" for why this matters to the interpretability argument.

## Measured tree-size divergence

Leaf counts per tree, read from each model's own serialization (20 trees, bankruptcy dataset)[^4]:

| max_depth | ClearGBM leaves | LightGBM leaves | ratio |
|---|---|---|---|
| 3 | 8.0 | 8.0 | 1.00× |
| 4 | 16.0 | 15.8 | 1.01× |
| 5 | 31.2 | 30.6 | 1.02× |
| **6** | **57.9** | **31.0** | **1.87×** |

Below depth 5 the `num_leaves` cap does not bind and the two build identically-sized trees. The divergence appears only at the benchmark's operating point of `max_depth=6, num_leaves=31`[^2].

## Authoritative measurement (2026-07-24)

78,682 rows × 18 features, 200 trees, depth 6, `max_bins=64`, `n_jobs=1`, median of 5 timed fits × 3 seeds[^1]:

| | fit_time | leaves/tree | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| lightgbm | 0.8981s ± 0.0719 | 30.96 | 0.6881 | 0.1366 |
| cleargbm | 1.2809s ± 0.0638 | 47.15 | 0.6821 | 0.1414 |

- raw ratio **1.426×**
- leaf ratio **1.523×**
- **per-leaf ratio 0.937× — ClearGBM is ~6% faster than LightGBM at equal tree size.**

Quality is a statistical tie; ClearGBM leads on AUC-PR (0.1414 vs 0.1366), the metric that matters on this class balance — the held-out positive rate runs 5.1%/6.8%/7.1% across seeds 44/42/43, mean 6.4%[^5].

## Three protocol properties, each fixing a past wrong conclusion

1. **Both learners measured in the same run.** LightGBM's measured fit time has ranged 0.69s → 0.95s across sessions *with identical LightGBM code*, purely from machine conditions. Dividing a fresh ClearGBM number by a LightGBM number carried forward from an older manifest manufactures a gap that is not there — this produced a reported "1.40× gap" on 2026-07-24 when the contemporaneous gap was near parity.
2. **Canonical statistic is the median, not the minimum.** The first fits after an idle period run with full turbo headroom — a different power regime, not noise. Taking a minimum let one cold-start outlier (LightGBM seed 42: `min/med = 0.486/0.828`) set the canonical number and produced an uninterpretable 1.751× ± 0.538; the median over the same data gave 1.43× ± 0.02.
3. **Results normalized by tree size**, per the divergence above.

## Where it lives

`libs/covenant_ml/src/covenant_ml/benchmarking/` — layered: `types` (records + codecs), `protocols` (injected boundaries), `timing`/`splitting`/`quality`/`model_shape`/`reporting` (pure), `dataset` (only file I/O), `adapters` (both learners), `runner` (protocol), `factory` (sole namer of concretes)[^6]. Entry point `scripts/benchmark_cleargbm_vs_lightgbm.py`[^7].

It lives in `covenant_ml`, not `cleargbm`, because `covenant_ml` already depends on both learners; `cleargbm` declares only `numpy` plus a path dependency on its own `cleargbm-rs` core[^8], and would otherwise take a dependency on its own competitor. An earlier harness lived only in a session scratchpad and was lost, after which every downstream perf claim was measured on a noisier shape and silently became incomparable.

[^1]: `libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json` — `dataset.sha256` `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`, `n_rows` 78682, `n_features` 18, `seeds` [42,43,44], `config.repeats` 5, `config.warmups` 2, `estimator` "median". Re-derived 2026-07-31 by averaging `timing.canonical_s` and `mean_leaves` over the three per-seed records per model: lightgbm 0.8981 ± 0.0719 s / 30.96 leaves / AUC-ROC 0.6881 / AUC-PR 0.1366; cleargbm 1.2809 ± 0.0638 s / 47.15 leaves / AUC-ROC 0.6821 / AUC-PR 0.1414 — every figure in the table above reproduces exactly.
[^2]: `libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json` `config` — `{"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "max_bins": 64, "min_data_in_leaf": 20, "num_leaves": 31, "reg_alpha": 0.0, "reg_lambda": 0.0, "n_jobs": 1, "repeats": 5, "warmups": 2}`.
[^3]: `libs/covenant_ml/src/covenant_ml/benchmarking/model_shape.py:5` — module doc states the premise directly: LightGBM stops "at ``num_leaves``, so the two can differ in tree size by roughly two-to-one".
[^4]: `libs/covenant_ml/src/covenant_ml/benchmarking/model_shape.py:27,62,85` — `mean_leaves_from_cleargbm_json(raw)` reads the document produced by `cleargbm.ensemble.export_model_json`; `mean_leaves_from_lightgbm_dump(dump)` reads `Booster.dump_model` and sums each tree's `num_leaves` field. Each learner's leaf count therefore comes from its own serialization, not from an external estimate.
[^5]: `libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json` — per-seed `quality.positive_rate`: seed 42 `0.06837160751565761`, seed 43 `0.07092320966350302`, seed 44 `0.05148…`; identical for both models at a given seed (same split). Mean 6.4%. Corrects this page's pre-2026-07-31 "6.6%-positive class", which matched no seed in the manifest.
[^6]: `libs/covenant_ml/src/covenant_ml/benchmarking/` directory listing (2026-07-31) — `types.py`, `protocols.py`, `timing.py`, `splitting.py`, `quality.py`, `model_shape.py`, `reporting.py`, `dataset.py`, `adapters.py`, `runner.py`, `factory.py`, plus `__init__.py` and `_test_hooks.py`.
[^7]: `libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py:8,64,102` — CLI entry point; docstring example `poetry run python -m scripts.benchmark_cleargbm_vs_lightgbm --repeats 5`.
[^8]: `libs/cleargbm/pyproject.toml:17-24` — `[tool.poetry.dependencies]` declares `python = "^3.11"`, `numpy = "^2.3.5"`, and `cleargbm-rs = { path = "../cleargbm_rs", develop = true }`. No `lightgbm` entry.
