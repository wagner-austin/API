---
title: ClearGBM perf — SIMD histogram accumulator
tags: [ml, cleargbm, rust, performance, simd]
related: [[cleargbm-histogram-split-path]], [[cleargbm-perf-uint8-histogram-bins]]
sources:
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/types/mod.rs
  - libs/cleargbm_rs/Cargo.toml
fact_checked: 2026-07-21
confidence: medium
---

# ClearGBM perf — SIMD histogram accumulator

Replace the scalar per-sample histogram accumulation loop with a SIMD-vectorized version. LightGBM ships hand-tuned AVX2 / AVX-512 accumulators; ClearGBM currently does one scalar `+=` per bin per sample. Expected: **2-3× faster on the histogram construction phase** (the dominant hot path)[^1].

## What's wrong today

The core hot loop in `libs/cleargbm_rs/src/histogram/mod.rs::build_histogram` (lines 64-80) walks `sample_indices` one at a time and delegates to `HistogramBuffer::accumulate`[^2]:

```rust
for &idx in sample_indices {
    if idx >= n_samples {
        return Err(ClearGbmError::SampleIndexOutOfBounds { ... });
    }
    let bin = bins[idx];
    let grad = gradients[idx];
    let hess = hessians[idx];
    match histogram.accumulate(bin, grad, hess) {
        Ok(()) => {}
        Err(e) => return Err(e),
    }
}
```

`HistogramBuffer::accumulate` at `libs/cleargbm_rs/src/types/mod.rs:236-252` then does three scalar ops per sample: `gradient_sums[bin] += grad; hessian_sums[bin] += hess; counts[bin] += 1`[^3]. Zero SIMD, zero unrolling, one bounds check per sample.

On the benchmark (55K samples × ~10 splits per tree × 200 trees × 18 features = ~2 billion accumulate calls), this is where ~60-80% of cleargbm's wall-clock lives.

## What to change

Two viable approaches — the second is more work but higher ceiling.

### Approach 1: batched scatter with `wide` (stable Rust)

Add `wide = "0.7"` to `libs/cleargbm_rs/Cargo.toml`'s `[dependencies]`. Rewrite the loop to load 4 or 8 samples at a time via `f64x4` / `f64x8`, do the bounds check on the batch, then scatter-add into the histogram arrays. Scatter isn't a native SIMD op on x86, so this is really "load 4 vectorized, then unroll 4 scalar scatters" — but the load pipelining and reduced bounds-check overhead is still a real 30-60% win.

### Approach 2: bin-first reordering + gather-add (larger refactor)

Instead of "for each sample, add to its bin", flip to "for each bin, sum the samples that landed in it". This requires a bin-first pass that groups sample indices by bin (radix-sort by bin, O(N)) before accumulation. Once samples are grouped by bin, the per-bin accumulation is a straight SIMD `reduce` — every op is contiguous. LightGBM's fast path is a variant of this.

Complexity trade-off: Approach 1 keeps the algorithm shape and ships in ~200 lines. Approach 2 is closer to LightGBM's ceiling but changes the algorithm's control flow and adds a scratch buffer for the bin-grouping pass.

## Prerequisites

This work only pays off in combination with the two lower-level changes:

- [[cleargbm-perf-column-major-sample-bins]] — without contiguous per-feature bin slices, the SIMD loads pull non-contiguous data through the cache and the vectorization ceiling drops from 4-8× to ~2×.
- [[cleargbm-perf-uint8-histogram-bins]] — with `u8` bins, a single 256-bit AVX2 load pulls 32 bin values; with `usize` it pulls 4. Bin-side density directly multiplies the SIMD lane count.

Do those two first. SIMD on top of scalar-friendly memory layout is the classic order-of-operations.

## Testing strategy

1. Keep the existing scalar `HistogramBuffer::accumulate` path — the tests in `libs/cleargbm_rs/src/histogram/tests/` (helper_tests, proptest_tests, unit_tests) exercise it directly.
2. Add a parallel test module: for random `(sample_indices, gradients, hessians, bins, n_bins)`, assert the SIMD path produces bit-identical `gradient_sums`, `hessian_sums`, `counts` vs the scalar reference (compare via `HistogramBuffer::eq`, which is derived).
3. Bench in isolation via `criterion` (would need to add `criterion` under `[dev-dependencies]`). Report before/after in `libs/cleargbm/docs/`.
4. Re-run the covenant_ml integration tests — any subtle FP-order shift in the accumulator will show up as a per-sample prediction diff.

## What NOT to change

- The bounds check on `idx >= n_samples` must stay — it's a real correctness invariant, and a panic-in-release scatter is worse than a scalar bounds check.
- Do not use `unsafe` `_mm256_i32gather_pd` intrinsics unless you're prepared to prove alignment + memory-safety on every consumer path. `wide` and `std::simd` avoid the entire class of undefined-behavior bugs.
- Do not add `nightly` or `no_std` requirements — the workspace is stable-Rust.

## Expected impact

Approach 1 alone: 30-60% faster on histogram construction, translating to ~15-30% faster overall fit. Approach 1 + column-major + uint8: cleargbm's fit time approaches ~1.5-2s on the benchmark, closing the LightGBM gap to ~2×.

Approach 2 is the ceiling — with all three prerequisites landed, cleargbm could plausibly match LightGBM on this specific workload. Not certain — LightGBM has 8+ years of production tuning that includes prefetch hints, alignment guarantees, and cache-line padding that a first-cut SIMD implementation won't match.

[^1]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md` § "Fixes not yet applied that would move these numbers further".
[^2]: `libs/cleargbm_rs/src/histogram/mod.rs:63-80` — the "Core hot loop" comment on line 63 marks the target region for SIMD replacement.
[^3]: `libs/cleargbm_rs/src/types/mod.rs:236-252` — `HistogramBuffer::accumulate` body: bounds check on `bin >= self.n_bins`, then three scalar `+=` ops on the parallel `gradient_sums`, `hessian_sums`, `counts` arrays.
