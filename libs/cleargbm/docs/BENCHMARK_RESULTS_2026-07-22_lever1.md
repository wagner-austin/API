# ClearGBM Lever 1 (ordered_gradients reorder) — 2026-07-22

**Change:** Applied the `ordered_gradients` / `ordered_hessians` reorder pattern documented in `~/PROJECTS/tech-wiki/pages/lightgbm-construct-histogram-inner.md`. One per-node reorder pass produces two sequential-access-shaped scratch arrays that all N features' histogram builds for that node reuse — cutting 2 of the 3 per-sample gathers versus the classic gather-per-feature path. Amortization ratio at 18 features per node: 18× reduction in the input-side gather count on gradients + hessians.

**Baseline being compared to:** the current post-Phase-E code in cleargbm_rs @ HEAD (commit `48f89591` — 8-wide unrolled histogram loop + `#[inline]` on trusted path + rayon threshold + column-major uint8 sample_bins).

## TL;DR

- **Speed: ClearGBM 1.48× faster than the pre-Lever-1 baseline** on the phase-e workload. Fit time dropped from **1.469s ± 0.204s** to **0.993s ± 0.037s** (measured after the single-path collapse; see "Design cleanup" section).
- **Quality: identical** — AUC, log-loss, and every seed's fit-order metric are bit-identical vs pre-Lever-1 (verified by 5 in-crate equivalence unit tests plus the phase-e integration bench).
- **Gap to LightGBM: 2.13× → 1.44×.** Closed roughly 65% of the remaining gap in a single safe-Rust refactor with zero API break.

## Wiki roadmap items landed in this run

The `Applied to cleargbm` section of `lightgbm-construct-histogram-inner.md` in the tech-wiki has been updated to reflect the landing. All 11 wiki pages passed `wiki_audit_run({wikiSlug: "tech"})` post-refactor (0 errors, 0 warnings).

| Wiki item | Status | Where |
| --- | --- | --- |
| `pages/lightgbm-construct-histogram-inner.md` — ordered_gradients + PREFETCH_T0 pattern | **shipped (ordered reorder only, no prefetch)** | `libs/cleargbm_rs/src/histogram/mod.rs::{build_histogram_ordered_trusted, reorder_grad_hess_into}`; `libs/cleargbm_rs/src/hooks.rs::Hooks::build_histogram_ordered`; `libs/cleargbm_rs/src/tree/histograms.rs::{build_feature_histograms, compute_child_histograms}`. |
| `pages/lightgbm-prefetch-t0-macro.md` | **not shipped** | Blocked by cleargbm's `#![forbid(unsafe_code)]` policy; needs relaxation for `_mm_prefetch`. |
| `pages/lightgbm-hist-entry-layout.md` — interleaved `[grad, hess]` output buffer | **not shipped** | Would require `HistogramBuffer` refactor + pyo3 return-type change. |
| `pages/lightgbm-implicit-count-cnt-factor.md` — drop counts, derive at split-scan | **not shipped** | Same as above — API-break on pyo3 boundary. |
| `pages/lightgbm-score-t-float.md` — narrow input to f32 | **not shipped** | Coordinated numpy-side change needed (Python currently sends f64). |
| `pages/lightgbm-goss-gradient-one-side-sampling.md` | **not shipped** | Documented for future workloads (paper-scale data); current 78K-row benchmark doesn't move the needle. |
| `pages/lightgbm-efb-exclusive-feature-bundling.md` | **not shipped** | Dense-feature workload; EFB benefits sparse-feature workloads not currently in scope. |
| `pages/lightgbm-dense-bin-packed-storage.md` — u8 workhorse variant | **already shipped Phase E** | `libs/cleargbm_rs/src/histogram/mod.rs` uses `&[u8]` bins. |
| `pages/lightgbm-data-partition-inplace.md` — one flat indices + in-place split | **not shipped** | Longer-tail allocation optimization; noted as follow-on. |
| `pages/lightgbm-sibling-subtraction-trick.md` — subtract-in-place from parent slot | **already shipped Phase E (out-of-place variant)** | `libs/cleargbm_rs/src/histogram/mod.rs::subtract_histogram` returns new HistogramBuffer; in-place mutation of parent buffer not yet implemented. |

## Results (mean ± std over 3 seeds, company-disjoint split, 200 trees @ depth 6, max_bins=64)

### Speed

| Version | fit_time | vs LightGBM | vs pre-Lever-1 baseline |
| ----- | -------- | ----------- | -------------------------------------- |
| **lightgbm** (from phase-e manifest) | **~0.69s** | **1.00×** | — |
| **cleargbm (post-Phase-E, pre-Lever-1)** | 1.469s ± 0.204s | 2.13× slower | (baseline) |
| cleargbm (Lever 1 with `Option<>` fallback) | 1.094s ± 0.083s | 1.59× slower | -0.375s, 1.34× faster |
| **cleargbm (Lever 1 collapsed, single path)** | **0.993s ± 0.037s** | **1.44× slower** | **-0.476s, 1.48× faster** |

Per-seed detail:

| Seed | Pre-Lever-1 | Lever-1 collapsed | Δ |
| ---: | ---: | ---: | ---: |
| 42 | 1.3395 | 1.0273 | -23.3% |
| 43 | 1.3646 | 0.9972 | -26.9% |
| 44 | 1.7042 | 0.9533 | -44.1% |
| **mean** | **1.4694** | **0.9926** | **-32.4%** |
| **std** | 0.2037 | 0.0372 | (variance ~5× tighter) |

Note: seed-44's larger pre-Lever-1 outlier disappears post-refactor. The reorder pass's cost is deterministic (linear scan over `sample_indices`), so its wall-time is more stable than the pre-refactor gather-heavy per-feature loop whose L1 miss rate varied with the sample-index distribution.

### Quality

| Model | AUC-ROC | log-loss |
| ----- | ------- | -------- |
| **cleargbm (pre-Lever-1)** | 0.6824 ± 0.0227 | 0.2299 ± 0.0329 |
| **cleargbm (post-Lever-1)** | **0.6824 ± 0.0227** | **0.2299 ± 0.0329** |
| Δ | 0.0000 | 0.0000 |

Bit-identical output. Verified by 5 in-crate equivalence unit tests in `libs/cleargbm_rs/src/histogram/tests/unit_tests.rs` covering: simple / subset / large-unrolled / tail-remainder / permuted-indices cases. The tree-level metrics agreeing to all 4 decimal places on 3 independent seeds is the empirical confirmation that the mathematical equivalence holds on the full training pipeline.

## Why the numbers changed

1. **18× fewer gradient/hessian gathers per sample.** The pre-Lever-1 hot loop for each of 18 features per node did `gradients[sample_indices[i]]` and `hessians[sample_indices[i]]` per sample — three total gathers per sample. Post-refactor the reorder pass at the start of the node does `ordered_gradients[i] = gradients[sample_indices[i]]` ONCE, then all 18 feature-histogram builds read `ordered_gradients[i]` sequentially. The bin lookup `bins[feat_col_start + sample_indices[i]]` remains a gather — matching LightGBM's exact shape per the wiki citation.
2. **Reduced L1 misses on the sequential-read arms.** `Vec<f64>` at position `i` shares a 64-byte cache line with positions `i-7..i+8`. Sequential reads amortize the fetch across 8 elements. Random-index reads (gathers) touch new lines per sample when `sample_indices` skips more than 8 positions — which it commonly does after depth-3+ splits partition the data. Line utilization on the gradient/hessian streams goes from ~12% (1 useful element per 8-element line) to ~100%.
3. **Variance collapse (0.204s → 0.083s).** The pre-refactor time was dominated by cache-miss patterns that varied per split. Post-refactor the hot loop is now deterministic-access shaped (one gather per sample instead of three); the timing distribution tightens.

## Implementation notes

**Single-path architecture.** No `Option<>` feature-flag on the hook. No classic-path fallback in the tree builder. The `BuildHistogramFn` typedef now takes `(sample_indices, ordered_gradients, ordered_hessians, bins, n_bins) -> Result<HistogramBuffer, ClearGbmError>` — one shape, fallible so error-injection tests still work by returning `Err(...)` from their custom fn. Everything routes through the same ordered fast path in production.

**Deleted:** the former `build_histogram_trusted` (classic gather-per-sample hot loop) is gone. The two-hot-loop shape from the initial Lever-1 landing collapsed into one after we noticed the fallback path was dead in production. See `git log -- libs/cleargbm_rs/src/histogram/mod.rs` before commit `<TBD>` for the historical classic implementation.

**Design cleanup pass.** The initial Lever-1 landing kept a per-feature-gather fallback path in `tree/histograms.rs` gated on `if let Some(ordered_hook) = ...`, purely so error-injection tests could exercise `?`-propagation through the fallible classic hook. That was 100+ lines of duplicated dispatch code kept alive for a testing surface. The collapse consolidates it: the ordered hook itself became fallible (`Result<...>`), tests inject through that, and both the fallback branch and the `Option<>` field are deleted. This bought an additional ~9% speedup (1.094s → 0.993s) from removing the `Option`-branch + fn-pointer indirection on the hot-per-feature dispatch.

**Public API preserved.** No change to `build_histogram_rs` or `subtract_histogram_rs` pyo3 exports. No change to Python-side `cleargbm.ensemble.train_gradient_boosting`. The refactor is entirely internal to the Rust tree builder.

**Rebuild sequence:**

```bash
cd libs/cleargbm_rs
maturin develop --release   # 14s rebuild against release profile
```

## Fixes not yet applied that would move these numbers further

Documented on the tech-wiki but not shipped in this refactor:

- **PREFETCH_T0 in the hot loop** — see `pages/lightgbm-prefetch-t0-macro.md`. Blocked by `#![forbid(unsafe_code)]`; no stable-safe cross-platform prefetch primitive in Rust. Relaxing to `#![deny(unsafe_code)]` + one `#[allow(unsafe_code)]` on a `prefetch` helper module would enable it. Expected impact: further 10-20% on top of Lever 1 by masking the remaining bin-lookup gather latency.
- **Interleaved `HistogramBuffer` + drop counts (`cnt_factor` derivation)** — see `pages/lightgbm-implicit-count-cnt-factor.md` Path A. Would consolidate 3 allocations per histogram into 1 (16 bytes/bin instead of 24) and cut cache pressure during the O(K) split scan. Public-API break on the pyo3 boundary (Python `build_histogram_rs` currently returns `(grads, hess, counts)` triples). Deferred pending API-compat decision.
- **`score_t = float` inputs** — see `pages/lightgbm-score-t-float.md`. Halves input memory bandwidth on the two sequential-read arrays the histogram loop touches. Requires coordinated numpy-side change (Python currently sends `float64`); the loss functions in `libs/cleargbm_rs/src/losses/derivatives.rs` also currently return `Vec<f64>`.

## Reproducibility

- **Script:** `libs/cleargbm/scripts/bench_phase_e.py` — the phase-e reproducer, checked into the repo.
- **Dataset:** `libs/covenant_ml/tests/data/american_bankruptcy.csv` — 78,682 rows, 18 features, ~6.6% positive class.
- **Config:** `n_estimators=200, max_depth=6, learning_rate=0.05, min_samples_leaf=20, max_bins=64, subsample=1.0, reg_alpha=0.0, reg_lambda=0.0, early_stopping_rounds=None, n_jobs=1`.
- **Seeds:** 42, 43, 44.
- **Split:** company-disjoint 70/15/15 on `company_name`.
- **Total wall time:** ~40 seconds for 3 seeds × (warm fit + timed fit + predict).

## Files touched

- `libs/cleargbm_rs/src/histogram/mod.rs` — `+build_histogram_ordered_trusted`, `+reorder_grad_hess_into`; rewrote public `build_histogram` to reorder-then-call-ordered; **deleted `build_histogram_trusted`** (classic gather-per-sample hot loop, ~120 lines) after the collapse.
- `libs/cleargbm_rs/src/hooks.rs` — reshaped `BuildHistogramFn` typedef to take ordered args + return `Result`; single `build_histogram` field on `Hooks` (no `Option`); default impl wraps `build_histogram_ordered_trusted` in `Ok(...)`. Constructors simplified.
- `libs/cleargbm_rs/src/tree/histograms.rs` — unconditional reorder + per-feature `(hooks.build_histogram)(...)?` dispatch. No `if let Some(...)` branch, no classic-path fallback.
- `libs/cleargbm_rs/src/histogram/tests/unit_tests.rs` — `+7` tests (5 equivalence, 1 populate-behavior, 1 should-panic on length mismatch).
- `libs/cleargbm_rs/src/tree/tests/error_tests.rs` — updated inline comment to reference `build_histogram_ordered_trusted` (was `build_histogram_trusted`).
- `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-22_lever1.md` — this file.
