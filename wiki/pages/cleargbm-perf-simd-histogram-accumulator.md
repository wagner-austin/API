---
title: ClearGBM perf — SIMD histogram accumulator
tags: [ml, cleargbm, rust, performance, simd]
related:
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-perf-uint8-histogram-bins]]"
  - "[[cleargbm-leaf-normalized-benchmarking]]"
  - "[[cleargbm-f32-score-narrowing-reverted]]"
source_paths:
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/types/mod.rs
  - libs/cleargbm_rs/Cargo.toml
source_git_blobs:
  "libs/cleargbm_rs/src/histogram/mod.rs": 930b2ce059cd5314ca5650a74cd44e31f8cfa8c8
  "libs/cleargbm_rs/src/types/mod.rs": b12e123111b6150ce710ed1cc6c03d2478d79922
  "libs/cleargbm_rs/Cargo.toml": 472c6cc568ce46dba53caba924b7fa1b7a3cf0d8
fact_checked: "2026-07-30"
confidence: medium
hubs: [libs]
---

# ClearGBM perf — SIMD histogram accumulator

The last unshipped item from the 2026-07-21 perf roadmap[^8]. **Its original premise is now obsolete**: the roadmap was written against a scalar accumulate loop that no longer exists[^1], and the version of "SIMD" it proposed as Approach 1[^9] has effectively already shipped by hand. What remains is the harder Approach 2[^9], against a much smaller gap than the roadmap assumed[^7].

## What the code actually does now

The hot loop is `build_histogram_ordered_trusted`, taking a `HistogramRequest`[^1]. It is manually unrolled 8-wide: eight `index_widen` calls, then eight `bins[idx]` gathers grouped together, then eight sequential `ordered_gradients[pos + k]` / `ordered_hessians[pos + k]` reads, then eight `gradient_sums[b] += g; hessian_sums[b] += h; counts[b] += 1` triples, with a scalar remainder tail[^1]. Bins are `&[u8]` and column-major[^2]; grad/hess are `f64` end to end[^3].

Three claims in the roadmap's original "what's wrong today"[^9] are now false[^1]:

- **"Delegates to `HistogramBuffer::accumulate`"** — it does not. `accumulate` still exists at `src/types/mod.rs:236`[^4] but every caller is under `src/**/tests/`; it is test-only surface, not the hot path.
- **"Zero unrolling"** — it is unrolled 8-wide[^1].
- **"One bounds check per sample"** — the trusted path has none. Validation moved to the top-level pyo3 boundary; the function establishes its invariants by construction and documents that a violation is a caller bug, not a recoverable error[^5].

Both prerequisites the page named are shipped: [[cleargbm-perf-column-major-sample-bins]] and [[cleargbm-perf-uint8-histogram-bins]] both landed in commit `6a2d15b7`[^2].

## Why Approach 1 is already done

The roadmap's Approach 1 conceded its own ceiling[^9]: scatter is not a native SIMD op on x86, so the proposal reduced to "load 4 vectorized, then unroll 4 scalar scatters." That is a description of the current loop[^1] — grouped gathers, sequential loads, then unrolled scalar read-modify-writes. Adding `wide` to express the same shape in `f64x4` types would restate what the code already does; the load pipelining and bounds-check savings it promised have both been collected.

The remaining cost is intrinsic to the operation[^10]. Accumulating into `gradient_sums[bin]` is an indexed RMW — a scatter — and two samples in the same 8-wide block can land in the same bin, so the adds carry a potential dependency that vector lanes cannot resolve without conflict detection (AVX-512CD). This is the structural reason the histogram accumulate resists vectorization, and no amount of restating the loop in SIMD types changes it.

## What is actually left: Approach 2

Bin-first reordering — group sample indices by bin (radix-sort by bin, O(N)) so per-bin accumulation becomes a contiguous SIMD reduce rather than a scatter[^9]. This is the only version of the idea with headroom left, and it changes the algorithm's control flow plus adds a scratch buffer for the grouping pass.

Weigh it against a real risk: ClearGBM has now measured **three consecutive losses** from splitting fused hot-loop work into more passes — ordered-arrays unpooled (−4%), counts-backfill (−5%), `cnt_factor` reconstruction (−16%), plus the f32 narrowing revert[^6][^3]. A bin-grouping pre-pass is exactly that shape: one more pass over the data to make a later pass cheaper. The recorded meta-lesson is that this consistently loses more to overhead than it gains, and Approach 2 has no measurement contradicting it.

## The gap this would be closing

The roadmap's expected impact was anchored to a baseline that no longer exists (6.88s fit, 8× LightGBM gap). Current authoritative numbers: cleargbm 1.2809s ± 0.0638 vs LightGBM 0.8981s ± 0.0719 — raw **1.426×**, but **per-leaf 0.937×**, i.e. ClearGBM is already ~6% faster than LightGBM at equal tree size, with quality a statistical tie[^7].

That reframes the whole item. The remaining raw gap is mostly a tree-shape artifact — ClearGBM grows 47.15 leaves/tree against LightGBM's 30.96 at the benchmark's depth 6[^7] — not a histogram-throughput deficit. Closing it by matching tree size is [[cleargbm-perf-leaf-wise-growth]]'s subject, not this page's. **Recommendation: do not start Approach 2 as a perf play.** The histogram loop is not where the residual 1.426× lives.

## Testing strategy (if it is attempted anyway)

1. Assert bit-identical output against the current loop for random `(sample_indices, ordered_gradients, ordered_hessians, bins, n_bins)` — the equivalence-test pattern already in `src/histogram/tests/unit_tests.rs` covers simple / subset / large-unrolled / tail-remainder / permuted-indices cases.
2. Keep `cargo test --all-features` green and segment coverage at 100% (`make rust-cov` enforces the threshold).
3. Benchmark with the median-of-repeats harness at `libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py`, baseline measured in the same session — not carried forward from a manifest[^7].

## What NOT to change

- Do not reintroduce a per-sample bounds check to make a vector path "safe." Validation belongs at the pyo3 boundary where it already is[^5].
- Do not use `unsafe` gather/scatter intrinsics — the crate is `#![forbid(unsafe_code)]`, and that policy is what already blocked the `PREFETCH_T0` item.
- Do not add nightly or `no_std` requirements; the workspace is stable-Rust.

[^1]: `libs/cleargbm_rs/src/histogram/mod.rs:83-198` — `build_histogram_ordered_trusted(request: HistogramRequest<'_>)`; `chunks_exact(8_usize)` main loop at `:109-186`, scalar remainder at `:188-195`.
[^2]: `libs/cleargbm_rs/src/binning/feature_bins.rs:8-13,33` — module doc "sample_bins is a flat, column-major Vec<u8>: bin [feat_idx, sample_idx] lives at sample_bins[feat_idx * n_samples + sample_idx]"; `HistogramRequest.bins: &'a [u8]` at `src/histogram/mod.rs:48`. Landed in commit `6a2d15b7` "cleargbm perf: column-major uint8 sample_bins + unrolled histogram loop (wiki items 1-3)".
[^3]: See [[cleargbm-f32-score-narrowing-reverted]], which cites `src/narrow.rs:1-12` and `src/training/train.rs:194-205`.
[^4]: `libs/cleargbm_rs/src/types/mod.rs:236` — `pub fn accumulate(`. [synthesis] grep for `\.accumulate\(` across `src/` returns callers only in `histogram/tests/`, `types/tests/`, and `tree/tests/`.
[^5]: `libs/cleargbm_rs/src/histogram/mod.rs:15-19` — "both functions establish invariants by construction... There is no validated entry point — validation happens at the top-level pyo3 boundary in `pyo3_module::training_fns`, not at the per-histogram level"; the `# Panics` doc at `:78-80` states a violation "is a bug in the caller, not a recoverable runtime error."
[^6]: [[cleargbm-perf-experiments-2026-07-21]] § negative results — the three measured regressions and the recorded meta-lesson that extra passes lose more to overhead than they gain from bandwidth reduction.
[^7]: [[cleargbm-leaf-normalized-benchmarking]] — authoritative 2026-07-24 table: lightgbm 0.8981s ± 0.0719 / 30.96 leaves, cleargbm 1.2809s ± 0.0638 / 47.15 leaves; raw ratio 1.426×, per-leaf ratio 0.937×. Harness and manifest cited there.
[^8]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md:60` — the roadmap entry this page derives from, verbatim: "**SIMD histogram accumulator** — Rust `wide` crate or nightly `std::simd`. AVX2/AVX-512 accumulation. Expected: 2-3× faster on the histogram phase, which is where most of cleargbm's remaining runtime lives." Listed under fixes not yet applied; "Expected" marks it a projection, never a measurement.
[^9]: This page's own pre-rewrite revision, commit `35700d41` "wiki: cleargbm perf roadmap — 4 atomic pages, each implementation-ready" [synthesis] — the Approach 1 / Approach 2 decomposition, the "what's wrong today" claims, and the bin-first proposal are that revision's text, superseded by commit `3166ebe4`. The original Approach 2 read: "Instead of 'for each sample, add to its bin', flip to 'for each bin, sum the samples that landed in it'. This requires a bin-first pass that groups sample indices by bin (radix-sort by bin, O(N)) before accumulation. Once samples are grouped by bin, the per-bin accumulation is a straight SIMD `reduce` — every op is contiguous. LightGBM's fast path is a variant of this." Recover with `git show 35700d41:wiki/pages/cleargbm-perf-simd-histogram-accumulator.md`.
[^10]: [synthesis] — read off the loop shape at `libs/cleargbm_rs/src/histogram/mod.rs:109-186`[^1]: the accumulate is `gradient_sums[b] += g` for a `b` derived from data, i.e. an indexed read-modify-write whose target is not known until the gather completes, and nothing in the loop constrains two lanes of one 8-wide block to distinct bins. The AVX-512CD (conflict-detection) reference is architectural background, not a measurement on this codebase; no SIMD variant of this loop has been benchmarked here.
