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
source_git_blobs:
  "libs/cleargbm_rs/src/binning/feature_bins.rs": 46d3455d66a7f6b667f4b12aa883272951872ca5
  "libs/cleargbm_rs/src/binning/assignment.rs": 78796752a4a5d2d81c73f95735f18703e6f30878
  "libs/cleargbm_rs/src/binning/edges.rs": 0abcb089e821c3c97cba3d81a8e94be4e6bef406
  "libs/cleargbm_rs/src/histogram/mod.rs": 930b2ce059cd5314ca5650a74cd44e31f8cfa8c8
  "libs/cleargbm_rs/src/training/config.rs": b0b59d60edc871f4808c72dc582eaac15087f39b
  "libs/cleargbm_rs/src/types/mod.rs": b12e123111b6150ce710ed1cc6c03d2478d79922
fact_checked: "2026-07-31"
confidence: high
hubs: [libs]
---

# ClearGBM perf — uint8 histogram bin dtype

Change the bin index dtype from `usize` (8 bytes on 64-bit) to `u8` (1 byte). LightGBM caps `max_bin ≤ 255` and auto-compresses feature values to `uint8_t` at that cap for this reason[^10]: 8× more bin values fit in a cache line[^7].

> **SHIPPED 2026-07-21 (Phase E) — this page was a roadmap item and is now a record.** Three of the four changes below landed together with the column-major refactor: the `Vec<u8>` storage[^2], the `max_bins ≤ 255` config bound[^3], and the `&[u8]` histogram parameter[^4]. Only the `assign_bin` return type is still open, and it is now cosmetic (see § "What is still open"). The forecast in the original "Expected impact" section was **met**: Phase E measured fit time 6.88s → 2.47s and the LightGBM gap 8.0× → 3.4×[^6]. Audited 2026-07-31.

## What the code does now

`FeatureBins.sample_bins` is a flat, column-major `Vec<u8>` — bin `[feat_idx, sample_idx]` lives at `sample_bins[feat_idx * n_samples + sample_idx]`, so a per-feature scan walks `n_samples` contiguous bytes[^2]. On the benchmark config (`n_samples = 55,502`, `n_features = 18`) that is the difference between an 8 MiB bin array, which does not fit in L2, and a 1 MiB one that does[^7].

The information content is tiny — bins are integers in `0..=max_bins`. `compute_bin_edges` validates only that `max_bins ≤ u32::MAX`[^8], but `GradientBoostingConfig::new` now enforces the real `u8` ceiling of 255[^3], and the benchmark harness runs `max_bins = 64`[^9]. LightGBM's own default is 255[^10]. A `u8` (1 byte) covers `0..=255`.

## What landed

**The `max_bins ≤ 255` upper bound** is in `GradientBoostingConfig::new`, immediately after the existing `max_bins < 2` check, carrying the comment "Bin indices are packed into u8 for cache-line density (see FeatureBins storage layout). Enforce the u8 upper bound here so downstream code can rely on it without another check."[^3]

**The storage dtype** is `sample_bins: Vec<u8>` on `FeatureBins`, flat and column-major, landed in the same lift as the column-major refactor from [[cleargbm-perf-column-major-sample-bins]][^2].

**The histogram parameter** is `pub bins: &'a [u8]` on `HistogramRequest`[^4]. `HistogramBuffer::accumulate` still takes `bin: usize`[^5], so the widening happens at the call site — as predicted, no signature change was needed on the accumulator itself.

## What is still open

**The return type of `assign_bin`** in `libs/cleargbm_rs/src/binning/assignment.rs` is still `usize`[^11]. The binary search bounds are `0..=len(edges)` where `len(edges) < max_bins ≤ 255`, so `u8` fits. This is now **cosmetic rather than a perf item**: storage is already `u8`, so the value is narrowed at the write into `sample_bins` regardless, and no hot-path scan reads an `assign_bin` return. Binning runs once per fit, not per histogram build. Changing it would remove one conversion at the storage boundary and make the invariant type-level instead of comment-level; it would not move the benchmark.

## What NOT to change

- `HistogramBuffer.gradient_sums`, `.hessian_sums`, `.counts`[^5] all stay `Vec<f64>`/`Vec<usize>`. There are at most `max_bins + 1 ≤ 256` entries — the size question is on the sample-side bins array, not the per-bin accumulator.
- `BinEdges.edges` stays `Vec<f64>` — threshold values are real numbers, unaffected.
- `TreeNode.feature_index` stays `Option<usize>` — that's a feature index, not a bin index.

## Testing strategy for the residual

1. Add config validation test in `libs/cleargbm_rs/src/training/tests/config_tests.rs`: `max_bins = 256` returns `InvalidParameter`; `max_bins = 255` succeeds.
2. Reuse existing histogram tests — they'll catch any accidental sign or width extension bugs.
3. Add a proptest asserting `assign_bin(v, edges) as usize == old_impl(v, edges)` for random inputs (guards the width change).
4. Confirm `cargo test --all-features` stays green + `cargo llvm-cov` stays at 100% segment coverage.

## Composition with column-major

The dtype change was only ever useful in combination with [[cleargbm-perf-column-major-sample-bins]] — row-major `Vec<Vec<u8>>` still fragments the heap and defeats the cache-line-density win. In the event both landed in the same lift rather than sequentially[^6], giving 8× more values per cache line AND contiguous access.

## Measured impact

The original forecast on this page was a projection, not a measurement: the source doc lists it under fixes "not yet applied" with the wording "Expected: 30-60% faster fit"[^1]. Phase E then measured the combined column-major + uint8 + unrolled-accumulator lift: fit time **6.88s ± 0.13s → 2.47s ± 0.03s** (2.79× faster than the prior baseline) and the gap to LightGBM **8.0× → 3.4×**, with every quality metric inside seed-to-seed noise[^6]. That landed inside this page's own "6.88s → 2-3s" forecast range, which the Phase E write-up cites back explicitly[^6].

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md:59` — listed under fixes not yet applied, verbatim: "**uint8 histogram bins** — ClearGBM uses int64 bins internally; LightGBM caps `max_bin ≤ 255` and uses uint8. 8× cache-line density on histogram accumulation. Expected: 30-60% faster fit." The "Expected:" is load-bearing — this was a projection, and this page previously restated it as a measured figure.
[^2]: `libs/cleargbm_rs/src/binning/feature_bins.rs:8-9,31-33` — module doc: "`sample_bins` is a flat, column-major `Vec<u8>`: bin `[feat_idx, sample_idx]` lives at `sample_bins[feat_idx * n_samples + sample_idx]`"; the field is declared `sample_bins: Vec<u8>`, with the per-feature slice accessor at `:63-73`. Corrects this page's pre-2026-07-31 citation of `feature_bins.rs:22 — sample_bins: Vec<Vec<usize>>`, which described the pre-Phase-E layout.
[^3]: `libs/cleargbm_rs/src/training/config.rs:135-143` — after the `max_bins < 2` check, `if max_bins > 255_usize { return Err(ClearGbmError::InvalidParameter { name: "max_bins", reason: format!("must be <= 255 (u8 bin index), got {max_bins}") }) }`, preceded by the comment "Bin indices are packed into u8 for cache-line density (see FeatureBins storage layout)."
[^4]: `libs/cleargbm_rs/src/histogram/mod.rs:48` — `pub bins: &'a [u8]` on `HistogramRequest`.
[^5]: `libs/cleargbm_rs/src/types/mod.rs:236-238` — `pub fn accumulate(&mut self, bin: usize, ...)`; the buffer fields at `:185-197` are `gradient_sums: Vec<f64>`, `hessian_sums: Vec<f64>`, `counts: Vec<usize>`, `n_bins: usize`.
[^6]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21_phase_e.md:1-13` — "Phase E benchmark following the column-major + uint8 + unrolled histogram accumulator lift on the Rust core"; "Fit time dropped from **6.88s ± 0.13s** to **2.47s ± 0.03s** on the 200-tree / depth-6 / max_bins=64 config"; "**Gap to LightGBM: 8.0× → 3.4×.** Well inside the wiki's forecast range for column-major + uint8 combined ("6.88s → 2-3s")"; quality "unchanged".
[^7]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21_phase_e.md:49` — "On a 55K-sample × 18-feature dataset that's the difference between an 8 MiB bin array (worse than L2 on most CPUs) and a 1 MiB bin array (fits comfortably in L2)."
[^8]: `libs/cleargbm_rs/src/binning/edges.rs:112,122-128` — `compute_bin_edges` errors with "exceeds maximum supported value (u32::MAX)"; its doc at `:104` reads "Maximum number of bins per feature (>= 2, ≤ u32::MAX)".
[^9]: `libs/cleargbm/scripts/benchmark.py:140,215` — `max_bins: int = 64` and the labelled case `("max_bins=64 (default)", make_config(n_estimators=n_estimators, max_bins=64))`. Note this is the harness default; the library type requires `max_bins` explicitly (`libs/cleargbm/src/cleargbm/_types_model.py:70,143`).
[^10]: LightGBM upstream, [`include/LightGBM/config.h`](https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/config.h):680 — `int max_bin = 255;`, with the adjacent note at `:679` "LightGBM will auto compress memory according to ``max_bin``. For example, LightGBM will use ``uint8_t`` for feature value if ``max_bin=255``". Verified 2026-07-31 against the local checkout at `~/PROJECTS/LightGBM`; cited by URL because that repository is outside this workspace root.
[^11]: `libs/cleargbm_rs/src/binning/assignment.rs:87` — `pub(super) fn assign_bin(value: f64, edges: &[f64]) -> usize`. Still `usize` as of 2026-07-31.
