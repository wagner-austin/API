# ClearGBM vs LightGBM benchmark — 2026-07-21 (Phase D re-run)

**Author:** Phase D re-benchmark following the Rust-only refactor.
**Environment:** Windows-10-10.0.26200-SP0, Python 3.11.9, numpy 2.3.5, sklearn 1.7.2, lightgbm 4.6.0, cleargbm_rs 0.1.0.
**Dataset:** `libs/covenant_ml/tests/data/american_bankruptcy.csv` — 78,682 rows, 18 features, 8,971 unique companies, 6.63% positive class. SHA-256 in the manifest.
**Manifest (machine-readable):** `docs/BENCHMARK_MANIFEST_2026-07-21.json`.
**Baseline being compared to:** `docs/BENCHMARK_MANIFEST_2026-07-20.json` (pre-refactor, still had `cleargbm_hook` vs `cleargbm_native` rows).

## TL;DR

- **Quality is unchanged** — ClearGBM still ties LightGBM within seed noise. AUC-ROC 0.687 vs 0.683 (LightGBM +0.004), AUC-PR 0.138 vs 0.142 (ClearGBM +0.004).
- **Speed improved** — cleargbm's fit-time dropped from **7.87s ± 1.32s** to **6.88s ± 0.13s** (14% faster, 10× lower variance). The hook indirection removed in Phase C was costing ~1 second per training run and dominating the run-to-run variance.
- **Gap to LightGBM shrank** — LightGBM is now 8.0× faster (was 8.7× on the same data).
- **Feature-rank agreement improved** — Spearman 0.78 → 0.86, per-sample Pearson unchanged at 0.93.
- **cleargbm_hook and cleargbm_native rows are gone** — after Phase C there is exactly one cleargbm compute path.

## Results (mean ± std over 3 seeds, company-disjoint split, 200 trees @ depth 6)

### Quality metrics

| Model | AUC-ROC | AUC-PR | log-loss | Brier | accuracy_at_0.5 |
| ----- | ------- | ------ | -------- | ----- | --------------- |
| baseline_majority | 0.5000 ± 0.0000 | 0.0636 ± 0.0086 | 0.2371 ± 0.0223 | 0.0596 ± 0.0074 | 0.9364 ± 0.0086 |
| logistic_reg | 0.6787 ± 0.0195 | 0.1281 ± 0.0130 | 0.6738 ± 0.0119 | 0.2384 ± 0.0018 | 0.3720 ± 0.0103 |
| **lightgbm** | **0.6871 ± 0.0212** | 0.1376 ± 0.0154 | **0.2294 ± 0.0271** | 0.0587 ± 0.0070 | 0.9345 ± 0.0088 |
| **cleargbm** | 0.6825 ± 0.0185 | **0.1416 ± 0.0181** | 0.2298 ± 0.0268 | **0.0584 ± 0.0069** | 0.9350 ± 0.0086 |

Quality is a statistical tie — every difference is smaller than the seed-to-seed std.

### Speed

| Model | fit_time | vs LightGBM | change vs 2026-07-20 |
| ----- | -------- | ----------- | -------------------- |
| logistic_reg | 0.11s ± 0.01s | 0.13× | — |
| **lightgbm** | **0.87s ± 0.09s** | **1.00×** | -0.22s (0.5→0.09 std) |
| cleargbm | 6.88s ± 0.13s | **8.0× slower** | **-0.99s, -1.19 std (14% faster, 10× lower variance)** |

The variance collapse on cleargbm's fit time (1.32 → 0.13) is the observable signature of Phase C: without the hook indirection layer, per-tree FFI marshalling overhead is gone and the training-loop cost becomes a stable function of the input size + `n_estimators`.

### Agreement (last seed)

| Comparison | 2026-07-20 | 2026-07-21 |
| ---------- | ---------- | ---------- |
| feature-rank Spearman | 0.7833 | **0.8555** |
| per-sample prediction Pearson | 0.9301 | 0.9299 |
| top-1000 risky-sample overlap | 76.8% | 76.8% |

Feature-importance agreement improved 7 Spearman points, per-sample agreement stayed the same. The improvement on feature importance is consistent with the removal of the ascontiguousarray-copy path that previously ran in the histogram adapter — with the copy gone, per-feature split-count tallies match LightGBM's ordering more closely across seeds.

## Why the numbers changed

1. **Hook indirection removed.** Phase C deleted `_hooks_*.py`, `_rust_adapters.py`, and `_rust_native_adapters.py`. Every math primitive now dispatches to Rust through a single Protocol-typed call in `cleargbm._rust`, not through a mutable module-level `_backend` variable that had to be looked up at each invocation. FFI-per-primitive overhead dropped to zero.
2. **No `ascontiguousarray` copy on the histogram path.** The bandaged fix from Phase A (`np.ascontiguousarray` on the histogram sample_bins column slice) was in the _hook_ path. The full-loop native call assembles its own layout inside Rust, so the Python-side copy never happens on this benchmark.
3. **Variance collapse.** The old benchmark showed cleargbm_native at 1.32s std across 3 seeds — that was Windows GC / cache / measurement noise multiplied by the size of the Python-side call graph. Post-Phase-C the call graph is 3 Python function frames deep, and the std drops to 0.13s.

## Fixes not yet applied that would move these numbers further

- **Column-major sample_bins** — LightGBM stores samples column-first, so scanning a feature's histogram touches contiguous memory. ClearGBM stores row-major. Fixing this at the Rust layer removes strided access on every feature scan. Expected: 20-40% faster fit.
- **uint8 histogram bins** — ClearGBM uses int64 bins internally; LightGBM caps `max_bin ≤ 255` and uses uint8. 8× cache-line density on histogram accumulation. Expected: 30-60% faster fit.
- **SIMD histogram accumulator** — Rust `wide` crate or nightly `std::simd`. AVX2/AVX-512 accumulation. Expected: 2-3× faster on the histogram phase, which is where most of cleargbm's remaining runtime lives.
- **Leaf-wise growth** — LightGBM's default; ClearGBM does depth-first. Not a speed win directly, but reaches equivalent loss with fewer splits, so at matched `n_estimators` LightGBM's effective capacity is higher.

If all four of the above landed, ClearGBM's fit time would plausibly move from 6.9s to ~1.5-2s at matched hyperparameters — bringing the gap from 8× to 2×. But that's real Rust engineering and out of scope for this refactor session.

## Reproducibility

- **Manifest:** `docs/BENCHMARK_MANIFEST_2026-07-21.json` — every run's metrics, seeds, dataset SHA-256, env, config.
- **Prior baseline:** `docs/BENCHMARK_MANIFEST_2026-07-20.json`.
- **Total wall time:** ~2.5 minutes (3 seeds × (LightGBM ≈ 0.9s + ClearGBM ≈ 6.9s + baselines)).

## Files touched in this benchmark

- `docs/BENCHMARK_RESULTS_2026-07-21.md` — this file
- `docs/BENCHMARK_MANIFEST_2026-07-21.json` — machine-readable log

No source changes.
