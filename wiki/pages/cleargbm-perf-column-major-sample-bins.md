---
title: ClearGBM perf — column-major sample_bins layout
tags: [ml, cleargbm, rust, performance]
related: [[cleargbm-histogram-split-path]]
sources:
  - libs/cleargbm_rs/src/binning/feature_bins.rs
  - libs/cleargbm_rs/src/binning/assignment.rs
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/tree/builder.rs
fact_checked: 2026-07-21
confidence: high
---

# ClearGBM perf — column-major sample_bins layout

The single largest per-hour ROI on ClearGBM's runtime cost. Reshapes `FeatureBins.sample_bins` from a per-row `Vec<Vec<usize>>` (each row heap-allocated separately) into a single flat column-major `Vec<usize>` indexed as `sample_bins[feat_idx * n_samples + sample_idx]`. Expected: **20-40% faster fit** on the 6.88s / 200-tree benchmark[^1].

## What's wrong today

`FeatureBins.sample_bins` is `Vec<Vec<usize>>` where the outer index is the sample and the inner is the feature[^2]. The construction loop in `precompute_feature_bins` fills it row-by-row: `for row in x { let mut row_bins = Vec::with_capacity(bin_edges.len()); ... sample_bins.push(row_bins); }`[^3]. Every `row_bins` is an independent heap allocation.

The consumer — the tree builder's histogram-construction path — needs the OPPOSITE access pattern. For each split candidate on feature `j`, it walks all sample indices at the current node and reads their bin ID on feature `j`. That's `sample_bins[i][j]` for successive `i`, which jumps between disjoint heap allocations for every sample. Cache prefetching cannot help — the memory addresses have no linear relationship.

## What to change

**Storage layout** (in `libs/cleargbm_rs/src/binning/feature_bins.rs`):

```rust
pub struct FeatureBins {
    bin_edges: Vec<BinEdges>,
    // Column-major: sample_bins[feat_idx * n_samples + sample_idx]
    sample_bins: Vec<usize>,
    n_samples: usize,
    n_features: usize,
    n_regular_bins: usize,
}
```

Accessor becomes `pub fn bins_for_feature(&self, feat_idx: usize) -> &[usize]` returning `&self.sample_bins[feat_idx * self.n_samples..(feat_idx + 1) * self.n_samples]` — one contiguous slice, no strided access.

**Construction** (same file, `precompute_feature_bins`): rewrite the loop so it emits column-major output. Simplest: two passes — one to allocate the flat `Vec<usize>` sized `n_samples * n_features`, one to fill it with `sample_bins[feat_idx * n_samples + sample_idx] = assign_bin(...)`.

**Callers**: `libs/cleargbm_rs/src/histogram/mod.rs::build_histogram` already takes `bins: &[usize]` (per-feature)[^4] — no signature change. The tree builder in `libs/cleargbm_rs/src/tree/builder.rs` currently derives its per-feature bin slice through `sample_bins()[i][j]` walks; it should switch to calling `bins_for_feature(j)` once.

**Retire the `sample_bins()` accessor** that returns `&[Vec<usize>]`[^5] — that's the row-major shape.

## Testing strategy

1. Add a unit test in `libs/cleargbm_rs/src/binning/tests/feature_bins_tests.rs` asserting `bins_for_feature(j)` returns the same values that `sample_bins[i][j]` produced pre-refactor, for a fixed (x, max_bins) fixture.
2. Add a proptest in the same file: for random x + max_bins, the round-trip `assign_bin(row[j], edges[j]) == fb.bins_for_feature(j)[i]` for all (i, j).
3. Confirm `cargo test --all-features` stays 100% green (1,485 tests today).
4. Re-run the covenant_ml integration test suite (2,060 tests, 100% coverage) — this is the mixed-endpoint validation that any layout mistake would surface.

## Expected impact

The 2026-07-21 benchmark measures cleargbm at 6.88s ± 0.13s and LightGBM at 0.87s ± 0.09s (200 trees, 55K samples, 18 features, depth 6, `max_bins=64`)[^1]. Histogram construction is the dominant hot path inside the Rust training loop. A 20-40% reduction takes cleargbm to ~4-5.5s, closing the LightGBM gap from 8.0× to 5-6×.

The variance figure (10× lower variance vs the pre-refactor hook path) means this measurement is now genuinely load-bearing — regressions of even 10% will show up cleanly in a re-run.

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Speed" — cleargbm 6.88s ± 0.13s, LightGBM 0.87s ± 0.09s.
[^2]: `libs/cleargbm_rs/src/binning/feature_bins.rs:22` — `sample_bins: Vec<Vec<usize>>`; docstring `Shape: [n_samples][n_features]`.
[^3]: `libs/cleargbm_rs/src/binning/feature_bins.rs:105-117` — `for row in x { let mut row_bins = Vec::with_capacity(bin_edges.len()); for (feat_idx, be) in bin_edges.iter().enumerate() { ... row_bins.push(...) } sample_bins.push(row_bins); }`.
[^4]: `libs/cleargbm_rs/src/histogram/mod.rs:32-38` — `pub fn build_histogram(sample_indices: &[usize], gradients: &[f64], hessians: &[f64], bins: &[usize], n_bins: usize) -> Result<HistogramBuffer, ClearGbmError>`.
[^5]: `libs/cleargbm_rs/src/binning/feature_bins.rs:37-39` — `pub fn sample_bins(&self) -> &[Vec<usize>] { &self.sample_bins }`.
