---
title: ClearGBM — f32 score narrowing, shipped then reverted
tags: [ml, cleargbm, rust, performance, negative-result]
related:
  - "[[cleargbm-perf-experiments-2026-07-21]]"
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-perf-simd-histogram-accumulator]]"
source_paths:
  - libs/cleargbm_rs/src/narrow.rs
  - libs/cleargbm_rs/src/training/train.rs
  - libs/cleargbm_rs/src/histogram/mod.rs
source_git_blobs:
  "libs/cleargbm_rs/src/narrow.rs": d12500f3eddc2f4f075274ad3e4492060bf9f381
  "libs/cleargbm_rs/src/training/train.rs": 9e8a944dcd1c2c12f222118181a64a5ef11bbe2d
  "libs/cleargbm_rs/src/histogram/mod.rs": 930b2ce059cd5314ca5650a74cd44e31f8cfa8c8
fact_checked: "2026-07-30"
confidence: medium
hubs: [libs]
---

# ClearGBM — f32 score narrowing, shipped then reverted

LightGBM's asymmetric-precision shape — `score_t = float` inputs, `hist_t = double` accumulator — was implemented in ClearGBM's Rust core on 2026-07-21 and taken back out on 2026-07-25. Gradients and hessians are `f64` end to end today[^1]. This page exists because two code comments cite it by slug and the reasoning lived nowhere else.

## What the current code does

`HistogramRequest.ordered_gradients` and `.ordered_hessians` are `&[f64]`[^2]; the training loop builds both as `Vec<f64>`[^3]; the accumulator fields were always `Vec<f64>`[^4]. There is no narrowing anywhere in the hot path, and `narrow.rs` now holds only `index_widen` — the `u32 → usize` index conversion — with no `as` casts left in the module[^1].

## What shipped, and what came back out

The narrowing landed as one bundled lift of ten commits on 2026-07-21, starting with `5295f6f9` (which added `narrow.rs` with a gated `score_narrow` alongside `index_widen`) and ending with `d1442dcf`[^5]. That series is titled for what it carried: **f32 scores, u32 sample indices, the ordered-arrays fast path, and the leaf-cache side-channel all moved together**, e.g. `ea9bec90` "histogram hot path — ordered-arrays fast path, f32 scores, u32 sample_indices"[^5].

The revert is inside commit `8c06e47b` (2026-07-25), which deleted `score_narrow` and rewrote the `narrow.rs` module doc[^6]. Everything else from the lift survived: `u32` indices, ordered arrays, and the leaf cache are all still in the code[^1][^2].

## Why it was reverted

The stated reason, from the comment the revert itself introduced: narrowing was **measured 8% slower** on this workload, because at the node sizes reached here both widths already fit in L2, so there is no bandwidth to save and each element pays a widening conversion before its accumulate[^1][^3].

**The 8% figure has no benchmark artifact.** It appears only in those two code comments. No `BENCHMARK_RESULTS_*` doc or `BENCHMARK_MANIFEST_*` json in `libs/cleargbm/docs/` mentions f32, narrowing, or a re-measurement after 2026-07-24[^7]. The reverting commit further states that its own contents were "not authored or verified here" — it is a rescue checkpoint of a concurrent session's working tree, and its message is read off the diff rather than from intent[^6]. So the mechanism is plausible and the direction is probably right, but the magnitude is unreproduced. That is why this page is `confidence: medium`.

## Reconciling with the phase-G "+15%" claim

[[cleargbm-perf-experiments-2026-07-21]] records phase G as f32 narrowing shipping for a 15% gain (1.23s → 1.05s), which reads as a direct contradiction of "8% slower." Both can hold, because they measure different things:

- Phase G's number was attributed to f32 by benchmark run, **not** by commit boundary — the commit series bundled f32 with the ordered-arrays path, `u32` indices, and the leaf cache in single commits[^5], so no commit isolates f32's individual contribution.
- The revert measured removing f32 from the *later* code, after the leaf cache had cut `predict_tree` out of the hot path and the ordered-arrays reorder had already removed two of three per-sample gathers. Once those landed, the remaining working set fits in L2 and the bandwidth argument stops applying.

The transferable lesson is the attribution error, not the sign flip: a phase that ships four changes and benchmarks once cannot apportion its win among them[^5].

## If someone retries this

Do not re-land it on the strength of `lightgbm-score-t-float` alone. Isolate it — one commit that changes only the two input dtypes, benchmarked with the median-of-repeats harness at `libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py`[^8], against a baseline measured in the same session, since no commit in the original lift isolates f32's contribution[^5]. The L2-residency argument above predicts it loses at the current benchmark's node sizes[^1][^3]; a genuinely larger dataset is where it would start to pay, so a retry should change the data scale, not just the dtype.

Restoring it also re-adds the crate's only `cast_precision_loss` exemption. The crate currently has no `as` casts in `narrow.rs` at all[^1].

[^1]: `libs/cleargbm_rs/src/narrow.rs:1-12` — module doc: "It contains no `as` casts. Gradients and hessians are `f64` end to end: narrowing them to `f32` was measured 8% SLOWER on this workload, because both widths already fit in L2 at the node sizes reached here, so there is no bandwidth to save and each element pays a widening conversion before its accumulate."
[^2]: `libs/cleargbm_rs/src/histogram/mod.rs:42,45` — `pub ordered_gradients: &'a [f64]`, `pub ordered_hessians: &'a [f64]`; the write-site comment at `:104-111` records the same revert.
[^3]: `libs/cleargbm_rs/src/training/train.rs:194-205` — "Kept in f64 end to end. Narrowing these two streams to f32 for the histogram hot loop was measured 8% SLOWER on this workload", followed by `let gradients: Vec<f64>` and `let hessians: Vec<f64>`.
[^4]: `libs/cleargbm_rs/src/types/mod.rs:185-193` — `HistogramBuffer` fields `gradient_sums: Vec<f64>`, `hessian_sums: Vec<f64>`, `counts: Vec<usize>`. The accumulator width never changed across either direction of this experiment.
[^5]: Commits `5295f6f9`, `27f6b702`, `ea9bec90`, `7567af2d`, `21fd8455`, `1752a873`, `28741949`, `1c3ecba4`, `7a7781a8`, `d1442dcf` (all 2026-07-21) [synthesis] — `git log --reverse 5295f6f9~1..8c06e47b -- libs/cleargbm_rs`; commit subjects name f32, u32, ordered-arrays, and leaf-cache changes together rather than in separable commits.
[^6]: Commit `8c06e47b` (2026-07-25) "cleargbm: checkpoint in-flight Rust core and ensemble work" — "Checkpoint of work in progress from a concurrent session, committed so it is recoverable rather than left only in the working tree. Not authored or verified here, so the summary below is read off the diff, not from intent." Its diff of `src/narrow.rs` deletes `pub const fn score_narrow` and its `#[expect(clippy::as_conversions, clippy::cast_precision_loss)]` gate.
[^7]: `libs/cleargbm/docs/` [synthesis] — grep for `f32|narrow|8%` across all files returns no hit relating to this experiment; the newest artifact is `BENCHMARK_MANIFEST_2026-07-24.json`, whose `timestamp_utc` is `2026-07-24T18:44:34` — before the revert.
[^8]: `libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py:8,64,102` — the CLI takes `--repeats` (docstring example `--repeats 5`) and threads it into the run. The median is the canonical statistic, not the minimum: `libs/covenant_ml/src/covenant_ml/benchmarking/timing.py:42-44` computes `median_s = statistics.median(ordered)` and assigns it to `canonical_s`, its module doc at `:18` stating "The canonical value is the median. A minimum would report the fastest"; `benchmarking/__init__.py:15` repeats it as a package-level rule and `benchmarking/runner.py:136` stamps `"estimator": "median"` into the manifest. Verified 2026-07-31.
