---
title: ClearGBM perf — uint8 histogram bin dtype
tags: [ml, cleargbm, rust, performance]
related:
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-perf-column-major-sample-bins]]"
source_paths:
  - libs/cleargbm_rs/src/binning/feature_bins.rs
  - libs/cleargbm_rs/src/binning/assignment.rs
  - libs/cleargbm_rs/src/binning/edges.rs
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/training/config.rs
  - libs/cleargbm_rs/src/types/mod.rs
fact_checked: "2026-07-21"
confidence: high
hubs: [libs]
---

# ClearGBM perf — uint8 histogram bin dtype

Change the bin index dtype from `usize` (8 bytes on 64-bit) to `u8` (1 byte). LightGBM caps `max_bin ≤ 255` and uses `uint8` for this reason: 8× more bin values fit in a cache line. Expected: **30-60% faster fit** on the current benchmark[^1] — largest single-change perf gain after column-major layout.

## What's wrong today

`FeatureBins.sample_bins: Vec<Vec<usize>>`[^2] uses `usize` (8 bytes each on 64-bit). At the benchmark config `n_samples = 55,502`, `n_features = 18`, the total bin array is 55,502 × 18 × 8 = 8.0 MiB. Every histogram-scan tour of a feature reads all 55,502 bytes; at typical L2 sizes (256 KiB–1 MiB), the array does not fit in L2, so every scan is L3 or main-memory reads.

The information content is tiny — bins are integers in `0..=max_bins`, and `max_bins` is capped at `u32::MAX` in `compute_bin_edges` but is `64` by default and never realistically exceeds `256` (LightGBM's own default is 255). A `u8` (1 byte) covers `0..=255`.

## What to change

**Add an upper bound** in `libs/cleargbm_rs/src/training/config.rs::GradientBoostingConfig::new` alongside the existing `max_bins < 2` check[^3]:

```rust
if max_bins > 255_usize {
    return Err(ClearGbmError::InvalidParameter {
        name: "max_bins".to_string(),
        reason: "must be <= 255 (u8 bin dtype)".to_string(),
    });
}
```

**Change the storage dtype** (in `libs/cleargbm_rs/src/binning/feature_bins.rs` — best done in the same PR as the column-major refactor from [[cleargbm-perf-column-major-sample-bins]]):

```rust
pub struct FeatureBins {
    bin_edges: Vec<BinEdges>,
    sample_bins: Vec<u8>,                 // was Vec<Vec<usize>>
    n_samples: usize,
    n_features: usize,
    n_regular_bins: usize,
}
```

**Change the return type of `assign_bin`** in `libs/cleargbm_rs/src/binning/assignment.rs` from `usize` to `u8`. The binary search bounds are already `0..=len(edges)` where `len(edges) < max_bins ≤ 255`, so `u8` fits.

**Change the `bins` parameter** on `histogram::build_histogram` (`libs/cleargbm_rs/src/histogram/mod.rs:32-38`)[^4] from `&[usize]` to `&[u8]`. Inside the accumulation loop at lines 64-80, `bin` becomes `u8` and needs `bin as usize` for the `HistogramBuffer.accumulate(bin: usize, ...)` call — no signature change on `HistogramBuffer` itself (its `n_bins` is a count, not an index type).

## What NOT to change

- `HistogramBuffer.gradient_sums`, `.hessian_sums`, `.counts`[^5] all stay `Vec<f64>`/`Vec<usize>`. There are at most `max_bins + 1 ≤ 256` entries — the size question is on the sample-side bins array, not the per-bin accumulator.
- `BinEdges.edges` stays `Vec<f64>` — threshold values are real numbers, unaffected.
- `TreeNode.feature_index` stays `Option<usize>` — that's a feature index, not a bin index.

## Testing strategy

1. Add config validation test in `libs/cleargbm_rs/src/training/tests/config_tests.rs`: `max_bins = 256` returns `InvalidParameter`; `max_bins = 255` succeeds.
2. Reuse existing histogram tests — they'll catch any accidental sign or width extension bugs.
3. Add a proptest asserting `assign_bin(v, edges) as usize == old_impl(v, edges)` for random inputs (guards the width change).
4. Confirm `cargo test --all-features` stays green + `cargo llvm-cov` stays at 100% segment coverage.

## Composition with column-major

This change is only maximally useful in combination with [[cleargbm-perf-column-major-sample-bins]]. Row-major with `Vec<Vec<u8>>` still fragments the heap and defeats the cache-line-density win. Land column-major FIRST, then flip `usize` → `u8` on the same flat storage. Both changes together give 8× more values per cache line AND contiguous access → cache-friendly scans.

## Expected impact

Isolated: 30-60% faster fit on histogram construction (the hot path). Combined with column-major: additional multiplier — cache footprint drops from 8 MiB to 1 MiB on the benchmark config, which fits comfortably in most L2 caches.

Rough estimate assuming both land: cleargbm fit time 6.88s → 2-3s, LightGBM gap 8.0× → 2-3×[^1].

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Fixes not yet applied that would move these numbers further".
[^2]: `libs/cleargbm_rs/src/binning/feature_bins.rs:22` — `sample_bins: Vec<Vec<usize>>`.
[^3]: `libs/cleargbm_rs/src/training/config.rs:129-134` — the existing `if max_bins < 2_usize { return Err(...) }` block, next to which the `> 255` bound goes.
[^4]: `libs/cleargbm_rs/src/histogram/mod.rs:32-38` — current signature `bins: &[usize]`; accumulation loop at lines 64-80.
[^5]: `libs/cleargbm_rs/src/types/mod.rs:185-197` — `HistogramBuffer` fields: `gradient_sums: Vec<f64>`, `hessian_sums: Vec<f64>`, `counts: Vec<usize>`, `n_bins: usize`.
