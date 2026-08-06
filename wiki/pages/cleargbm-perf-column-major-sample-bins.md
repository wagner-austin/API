---
title: ClearGBM perf — column-major sample_bins layout
tags: [ml, cleargbm, rust, performance]
related:
  - "[[cleargbm-histogram-split-path]]"
source_paths:
  - libs/cleargbm_rs/src/binning/feature_bins.rs
  - libs/cleargbm_rs/src/binning/assignment.rs
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/tree/builder.rs
source_git_blobs:
  "libs/cleargbm_rs/src/binning/feature_bins.rs": 46d3455d66a7f6b667f4b12aa883272951872ca5
  "libs/cleargbm_rs/src/binning/assignment.rs": 78796752a4a5d2d81c73f95735f18703e6f30878
  "libs/cleargbm_rs/src/histogram/mod.rs": 930b2ce059cd5314ca5650a74cd44e31f8cfa8c8
  "libs/cleargbm_rs/src/tree/builder.rs": b28c382a41a489df68fa9cd7c2c87bafee26a6a9
fact_checked: "2026-07-31"
confidence: high
hubs: [libs]
---

# ClearGBM perf — column-major sample_bins layout

Reshapes `FeatureBins.sample_bins` from a per-row `Vec<Vec<usize>>` (each row heap-allocated separately)[^2] into a single flat column-major array indexed as `sample_bins[feat_idx * n_samples + sample_idx]`[^6].

> **SHIPPED 2026-07-21 (Phase E) — this page was a roadmap item and is now a record.** The layout landed in the same lift as the `u8` dtype from [[cleargbm-perf-uint8-histogram-bins]], so the flat array is `Vec<u8>` rather than the `Vec<usize>` this page originally proposed[^6]. The forecast was met and then some: Phase E measured fit time 6.88s → 2.47s and the LightGBM gap 8.0× → 3.4× for the combined lift[^7]. Audited 2026-07-31.

## What was wrong (pre-Phase-E)

`FeatureBins.sample_bins` was `Vec<Vec<usize>>` with the outer index the sample and the inner the feature[^2], and `precompute_feature_bins` filled it row-by-row so every inner row was an independent heap allocation[^3].

The consumer — the tree builder's histogram-construction path — needs the OPPOSITE access pattern. For each split candidate on feature `j`, it walks all sample indices at the current node and reads their bin ID on feature `j`. That was `sample_bins[i][j]` for successive `i`, jumping between disjoint heap allocations for every sample; cache prefetching cannot help when the addresses have no linear relationship[^8].

## What landed

**Storage layout.** `FeatureBins.sample_bins` is a flat `Vec<u8>` documented as "flat, column-major … bin `[feat_idx, sample_idx]` lives at `sample_bins[feat_idx * n_samples + sample_idx]`"[^6]. The dtype is `u8`, not the `usize` this page originally proposed, because the uint8 change rode the same lift.

**Accessor.** `pub fn bins_for_feature(&self, feat_idx: usize) -> &[u8]` returns `&self.sample_bins[feat_idx * self.n_samples..(feat_idx + 1) * self.n_samples]` — one contiguous slice, no strided access — and returns an empty slice for an out-of-range feature index rather than panicking[^9].

**Construction.** `precompute_feature_bins` allocates the flat array once (`vec![0_u8; n_samples * n_features]`) and fills it in feature-major order, hoisting `feat_col_start = feat_idx * n_samples` out of the inner loop[^10]. It rejects `max_bins > 255` up front, which is what lets the per-value `u8::try_from(...).unwrap_or(u8::MAX)` be a saturating rather than fallible conversion — the error arm would be statically dead[^10].

**Callers.** `HistogramRequest.bins` is `&'a [u8]`[^4].

## Testing strategy (as planned pre-lift; counts in steps 3-4 are 2026-07-21 figures, not re-verified)

1. Add a unit test in `libs/cleargbm_rs/src/binning/tests/feature_bins_tests.rs` asserting `bins_for_feature(j)` returns the same values that `sample_bins[i][j]` produced pre-refactor, for a fixed (x, max_bins) fixture.
2. Add a proptest in the same file: for random x + max_bins, the round-trip `assign_bin(row[j], edges[j]) == fb.bins_for_feature(j)[i]` for all (i, j).
3. Confirm `cargo test --all-features` stays 100% green (1,485 tests today).
4. Re-run the covenant_ml integration test suite (2,060 tests, 100% coverage) — this is the mixed-endpoint validation that any layout mistake would surface.

## Measured impact

The pre-lift baseline measured cleargbm at 6.88s ± 0.13s against LightGBM at 0.87s ± 0.09s (200 trees, 55K samples, 18 features, depth 6, `max_bins=64`)[^1]. The original forecast on this page was "20-40% faster fit", stated in the source doc as an expectation rather than a measurement[^1].

Phase E measured the combined column-major + uint8 + unrolled-accumulator lift at **6.88s ± 0.13s → 2.47s ± 0.03s**, a 2.79× speedup that took the LightGBM gap from 8.0× to 3.4× with quality unchanged inside seed noise[^7] — well past the 20-40% this page forecast, because the dtype and accumulator changes rode along. The per-feature scan now walks `n_samples` contiguous bytes instead of striding disjoint heap allocations, which on this dataset is the difference between an 8 MiB bin array and a 1 MiB one[^8].

The variance figure (10× lower variance vs the pre-refactor hook path[^11]) means this measurement is genuinely load-bearing — regressions of even 10% show up cleanly in a re-run.

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Speed" — cleargbm 6.88s ± 0.13s, LightGBM 0.87s ± 0.09s.
[^2]: `libs/cleargbm_rs/src/binning/feature_bins.rs:22` — `sample_bins: Vec<Vec<usize>>`; docstring `Shape: [n_samples][n_features]`.
[^3]: `libs/cleargbm_rs/src/binning/feature_bins.rs:105-117` — `for row in x { let mut row_bins = Vec::with_capacity(bin_edges.len()); for (feat_idx, be) in bin_edges.iter().enumerate() { ... row_bins.push(...) } sample_bins.push(row_bins); }`.
[^4]: `libs/cleargbm_rs/src/histogram/mod.rs:48` — `pub bins: &'a [u8]` on `HistogramRequest`. (Pre-Phase-E this page cited a `build_histogram(..., bins: &[usize], ...)` free-function signature; the request-struct form with a `u8` slice is what exists today.)
[^5]: `libs/cleargbm_rs/src/binning/feature_bins.rs` — the row-major `sample_bins(&self) -> &[Vec<usize>]` accessor this page proposed retiring is gone; the flat accessor at `:58` returns `&self.sample_bins` as `&[u8]`.
[^6]: `libs/cleargbm_rs/src/binning/feature_bins.rs:8-9,31-33` — module doc "`sample_bins` is a flat, column-major `Vec<u8>`: bin `[feat_idx, sample_idx]` lives at `sample_bins[feat_idx * n_samples + sample_idx]`. A per-feature …", and the field declaration `sample_bins: Vec<u8>`.
[^7]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21_phase_e.md:1-13` — "Phase E benchmark following the column-major + uint8 + unrolled histogram accumulator lift on the Rust core"; "Fit time dropped from **6.88s ± 0.13s** to **2.47s ± 0.03s**"; "**Gap to LightGBM: 8.0× → 3.4×.**"; quality "unchanged".
[^8]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21_phase_e.md:49` — "A per-feature histogram scan now walks `n_samples` contiguous bytes; the previous version strode through disjoint heap allocations. On a 55K-sample × 18-feature dataset that's the difference between an 8 MiB bin array (worse than L2 on most CPUs) and a 1 MiB bin array (fits comfortably in L2)."
[^9]: `libs/cleargbm_rs/src/binning/feature_bins.rs:61-74` — `pub fn bins_for_feature(&self, feat_idx: usize) -> &[u8]`, guarded by `if feat_idx >= self.n_features { return &[]; }` then slicing `start = feat_idx * self.n_samples` to `start + self.n_samples`.
[^10]: `libs/cleargbm_rs/src/binning/feature_bins.rs:150-190` — `precompute_feature_bins` rejects `max_bins > 255` at `:154-159`, allocates `let mut sample_bins = vec![0_u8; n_samples * n_features];` at `:171` under the comment "Flat column-major storage", then loops features-outer with `let feat_col_start = feat_idx * n_samples;` and writes `sample_bins[feat_col_start + sample_idx] = bin_idx;`. The saturating `u8::try_from(bin_idx_usize).unwrap_or(u8::MAX)` carries the comment explaining the error arm "would be statically dead: the guard has already run".
[^11]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md:12` — "cleargbm's fit-time dropped from **7.87s ± 1.32s** to **6.88s ± 0.13s** (14% faster, 10× lower variance)."
