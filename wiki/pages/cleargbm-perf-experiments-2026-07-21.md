---
title: ClearGBM perf experiments — 2026-07-21 session
tags: [ml, cleargbm, performance, benchmarks, negative-results]
related:
  - "[[cleargbm-histogram-split-path]]"
  - "[[cleargbm-perf-column-major-sample-bins]]"
  - "[[cleargbm-perf-uint8-histogram-bins]]"
  - "[[cleargbm-perf-simd-histogram-accumulator]]"
  - "[[cleargbm-perf-leaf-wise-growth]]"
source_paths:
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/tree/histograms.rs
  - libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json
fact_checked: "2026-07-31"
confidence: high
hubs: [libs]
---

# ClearGBM perf experiments — 2026-07-21 session

Empirical log of perf changes attempted on `cleargbm_rs` in the 2026-07-21 session, ranging over the wiki's perf roadmap (`cleargbm-perf-*`) plus new patterns discovered via tech-wiki research[^7]. Records what shipped, what tried-and-reverted, and — critically — WHY each negative result failed so future sessions don't repeat the experiment.

> **PROVENANCE AUDIT 2026-07-31 — read before quoting any number on this page.**
> Both manifest filenames this page cites are **mutable and were overwritten in place**, so a
> citation to a bare filename does not resolve to the run it was written against.
> `BENCHMARK_MANIFEST_2026-07-21.json` has exactly two committed versions: the Phase D baseline
> (cleargbm 6.8836s ± 0.1585) and a Phase-I refresh that replaced it (cleargbm 1.0456s ± 0.0102)[^1].
> `BENCHMARK_MANIFEST_2026-07-21_phase_e.json` likewise has two: as first written (cleargbm
> 2.4654s ± 0.0400) and a post-leaf-cache refresh (cleargbm 1.7889s ± 0.0642)[^2].
> Consequences, verified by re-deriving every figure from both versions of both files:
>
> - **Baseline 6.88s — anchored.** Recoverable at commit `67ce10c8`[^1].
> - **Phase G / Phase I 1.05s — anchored.** Matches the current file[^1].
> - **Phase E 1.60s — NOT anchored.** The as-written manifest says 2.4654s and the current one
>   1.7889s. The table's quality columns *do* match the manifest exactly, so only the fit-time
>   cell is unsourced[^2].
> - **Phase F 1.23s — NOT anchored.** No committed version of either manifest contains it[^1].
> - **"Gap closed from 8.0× to 2.2×" (Phase E) is wrong.** The manifest as written gives
>   2.4654 / 0.7223 = **3.41×**, which is what the authoritative Phase-E results doc states
>   ("Gap to LightGBM: 8.0× → 3.4×")[^3]. The 2.2× follows from the unsourced 1.60s and should
>   not be quoted; the session-context paragraph below inherits the same error.
>
> Everything not listed above — the commit table, the code paths, the reverts, and the
> reasoning — is anchored and re-verified[^4][^5]. In-session fit-time measurements that were
> never written to a committed artifact are marked [^6] throughout.

**Session context.** Started at cleargbm 6.88s / LightGBM 0.72s (8.0× gap; per `docs/BENCHMARK_MANIFEST_2026-07-21.json`). After the shipped-batch below: cleargbm 1.60s / LightGBM ~0.72s (2.2× gap; per `docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json`). Total speedup: 4.3×.[^1][^2]

## What shipped (five commits)

| Batch | Commit | Δ | Cumulative | Change |
|---|---|---|---|---|
| A | `eaa31ca7` | 6.88s → 2.47s (2.79×) | 2.79× | column-major u8 sample_bins + max_bins ≤ 255 + pre-validated 4-wide unrolled hot loop |
| B | `13a235e0` | 2.47s → 2.24s (1.10×) | 3.07× | rayon parallel histograms + `target-cpu=native` |
| C | `2db497a6` | 2.24s → 1.85s (1.21×) | 3.72× | trusted histogram fast path (skip validation) + fused pre-validation pass + rayon min-work threshold |
| D | `48f89591` | 1.85s → 1.66s (1.11×) | 4.14× | 8-wide unroll (was 4-wide) + `#[inline]` on trusted path |
| E | (no commit — build side) | 1.66s → 1.60s (1.04×) | 4.30× | subsequent bench-to-bench variance settle after wheel rebuild |

Every batch touched `libs/cleargbm_rs/src/histogram/mod.rs`, `libs/cleargbm_rs/src/tree/histograms.rs`, and `libs/cleargbm_rs/src/tree/builder.rs`; batches A + C also touched `libs/cleargbm_rs/src/binning/feature_bins.rs` (column-major refactor)[^4][^5].

## What was tried and reverted (three negative results)

### Experiment 1: ordered-arrays gather-elimination (unpooled)

**Hypothesis.** Per `~/PROJECTS/tech-wiki/pages/lightgbm-construct-histogram-inner.md`, LightGBM's histogram loop reads `ordered_gradients[i]` / `ordered_hessians[i]` sequentially (loop counter `i`, not sample-index gather) because the caller pre-permutes those arrays once per node. That eliminates 2 of 3 gathers per sample per feature and should amortize across features.[^7]

**Change.** New signature on `build_histogram_trusted`: takes `ordered_gradients: &[f64]` + `ordered_hessians: &[f64]` (length = `sample_indices.len()`). Callers (`tree/histograms.rs::build_feature_histograms` + `compute_child_histograms`) allocate `Vec<f64>` scratch per node via `permute_by_indices(full, sample_indices)`, pass to every per-feature histogram build.[^5]

**Result.** 1.66s baseline → 1.73s (**−4%**). Measured across 3 seeds, ±0.09s std.[^6]

**Root cause.** cleargbm's `sample_indices` at each node preserves sort order (subsequences of the sorted parent, which starts at `0..N`). So `gradients[sample_indices[k]]` is already a mostly-sequential access pattern — the hardware prefetcher + cache-line reuse handle it well. The theoretical gather cost isn't there to save. Meanwhile the pre-permutation adds ~50KB × 2 allocs per node × ~63 nodes/tree × 200 trees ≈ 1.2 GB of allocation traffic, which swamps the marginal read-side gain.[^6]

**Revert path.** `git checkout HEAD -- libs/cleargbm_rs/src/histogram/mod.rs libs/cleargbm_rs/src/hooks.rs libs/cleargbm_rs/src/tree/histograms.rs`.[^5]

**Would pooling the ordered buffers help?** Unknown. Not tested. Would require threading a mutable `&mut Vec<f64>` scratch pool through the tree-builder API. If the ~4% loss was purely allocation traffic (not the algorithmic pattern being wrong for cleargbm), pooling could reveal a small positive result. If cleargbm's sample-index-order really eliminates the gather cost anyway, pooled version would still show ~0% delta.[^6]

### Experiment 2: counts-backfill in a second pass (O(n) walk)

**Hypothesis.** The 3rd RMW stream in the hot loop (`counts[b] += 1_usize`) contributes 33% of the scatter-write bandwidth. Removing it from the fused 8-wide loop and doing a separate O(n) pass to backfill should split the work in a cache-friendlier way and isolate whether counts is the bottleneck.[^7]

**Change.** Delete `counts[bin] += 1_usize` from every position in `build_histogram_trusted`. Add a separate `for &idx in sample_indices { histogram.counts[usize::from(bins[idx])] += 1_usize; }` loop after the main loop finishes.[^5]

**Result.** 1.60s baseline → 1.68s (**−5%**). Measured across 3 seeds.[^6]

**Root cause.** Splitting the work into two loops adds its own overhead — the second loop re-walks `sample_indices` and re-gathers `bins[idx]`, doubling the L1 read pressure on those two streams. The tight fused 3-write loop was already CPU-pipeline-friendly (modern OoO cores overlap the 3 independent RMW streams well); decoupling them lost more than it gained.[^6]

**Revert path.** `git checkout HEAD -- libs/cleargbm_rs/src/histogram/mod.rs`.[^5]

**Implication.** This experiment does NOT isolate whether "eliminate the counts write" would help — it only measures "the O(n) backfill loop is expensive." To actually test the hypothesis, we'd need the O(n_bins) cnt_factor reconstruction (experiment 3) which we DID try.[^6]

### Experiment 3: cnt_factor reconstruction (O(n_bins))

**Hypothesis.** Per `~/PROJECTS/tech-wiki/pages/lightgbm-implicit-count-cnt-factor.md`, LightGBM's histogram has NO per-bin count array. Counts are reconstructed at split-scan time via `cnt = round(hess_per_bin * num_data / total_hessian)` — O(n_bins) reconstruction, no writes to counts in the hot loop. Applying to cleargbm: drop counts writes from `build_histogram_trusted`, backfill via cnt_factor after the main loop.[^7]

**Change.** Delete all `counts[b] += 1_usize` writes from the 8-wide loop. After main loop: `total_hess = sum(hessian_sums)`, `cnt_factor = n as f64 / total_hess`, for each bin `histogram.counts[b] = round(hessian_sums[b] * cnt_factor) as usize`. O(n_bins) reconstruction pass.[^5]

**Result.** 1.60s baseline → 1.86s (**−16%**) AND quality regressed: AUC-ROC 0.6991 → 0.6965 (−0.003), AUC-PR 0.1620 → 0.1573 (−0.005), log-loss 0.2303 → 0.2312 (+0.001). All within seed std of prior baseline but drift is systematic and one-sided.[^6]

**Root cause (speed).** The cnt_factor approximation is exact only when hessians are uniform across samples. For binary log-loss `hess = p*(1-p)` varies as training progresses (drifts from 0.25 at init toward 0 for confident-prediction samples). Approximate counts lie about the true per-bin sample count. Downstream `find_best_split`'s `min_samples_leaf` check uses these approximate counts and makes different split decisions than the exact-count baseline. Different splits → different tree structure → different runtime cost (the specific new trees may need more histogram builds or hit slower bin distributions). Not a bandwidth issue; an algorithmic one.[^6]

**Root cause (quality).** Same story from the split-decision angle: the `min_samples_leaf` check occasionally accepts a split with too-few-samples on one side (approximate count overshot) or rejects a split with just-enough samples (approximate count undershot). Small tree-structure diffs compound over 200 trees into measurable AUC drift.[^6]

**Revert path.** `git checkout HEAD -- libs/cleargbm_rs/src/histogram/mod.rs`.[^5]

**Implication.** The exact-count invariant in cleargbm's `find_best_split_from_histogram` is load-bearing for reproducibility. Dropping counts as LightGBM does requires ALSO changing `find_best_split_from_histogram` to consume `num_samples: usize` and reconstruct in an exact way (or accept the quality drift as a design choice). Not tested in this session.[^6]

> **CLOSED 2026-07-24 — do not re-propose.** This idea was re-proposed as a "Phase 4" interleaved-histogram change and rejected on two independent lines of evidence.
>
> 1. **Already measured, twice.** Experiments 2 and 3 above cost 5% and 16% respectively, and experiment 3 also drifted quality one-sidedly. The variant hedged at the end of this section — plumbing `num_samples` into `find_best_split_from_histogram` — carries the *same* approximate counts and so inherits the same tree-structure drift; it only saves the O(n_bins) backfill pass, which is negligible. It would reproduce experiment 3's failure minus a rounding error.
> 2. **It targets a non-binding dimension.** A `max_bins` sweep holding allocation count fixed while varying histogram *size* shows cleargbm flat from 16 → 64 bins (0.699 / 0.716 / 0.709s at 100 trees), rising only at 128 (0.806s) and 255 (0.976s). At the benchmark's `max_bins=64` the workload is not bound by histogram bytes, which is precisely what shrinking a histogram from three arrays to two would relieve.
>
> There is also no `min_sum_hessian_in_leaf` in cleargbm, so `counts` is the *only* leaf-size regularizer[^9]. LightGBM ships derived counts alongside an exact hessian constraint; adopting the approximation without that backstop is strictly worse than LightGBM's design, not equivalent to it[^6].

## Meta-lesson from three negative results

My theoretical predictions in the tech-wiki pages consistently OVERESTIMATED the applicable savings for cleargbm's specific tree traversal[^6][^7]:

- **Ordered arrays** was expected to eliminate 2 of 3 gathers per sample per feature (predicted 18× amortization win). Actually eliminated ~0 because cleargbm's sample_indices preserve sort order already.
- **Counts-drop** was expected to save ~33% of hot-loop write bandwidth. Actually cost ~5-16% depending on how counts got backfilled, because the fused 3-write loop was already CPU-pipeline-friendly.

The consistent failure mode[^6]: **splitting fused work into more loops loses more to overhead than it gains from bandwidth reduction.** Every optimization that ADDED a pass (permutation, backfill, reconstruction) lost time. Only optimizations that COMPRESSED work into fewer bytes-per-sample or eliminated bounds checks WITHIN the fused loop (column-major u8 + trusted fast path + 8-wide unroll) landed as wins.

**Practical guidance for future perf work on cleargbm**[^6]**:** stop reasoning from LightGBM-analogy. LightGBM's design context (float32 grad+hess, sequential gathers, explicit prefetch, no unsafe policy) differs enough from cleargbm's (float64 everywhere, `unsafe_code = "forbid"`, safe-only) that patterns don't transfer 1:1. Profile the actual hot path with instrumentation before proposing changes. See task queue for planned instrumentation work.

## What has NOT been tested this session

Remaining items from the perf roadmap (`cleargbm-perf-*` pages) that are still theoretical — see [[cleargbm-perf-simd-histogram-accumulator]] and [[cleargbm-perf-leaf-wise-growth]] for their current status:

- **[[cleargbm-perf-leaf-wise-growth]]** — wiki-flagged "do LAST" and "confidence: medium." Not attempted; interpretability trade-off flagged.
- **f32 gradient/hessian narrowing** — per `~/PROJECTS/tech-wiki/pages/lightgbm-score-t-float.md`, halves input memory bandwidth. Not attempted; substantial refactor across training loop + losses + pyo3 boundary + all tests. Structurally different from the failed experiments (doesn't split work into more loops), so may actually deliver — needs profiling data before committing.
- **`_mm_prefetch` on the bin-lookup gather** — per `~/PROJECTS/tech-wiki/pages/lightgbm-prefetch-t0-macro.md`, LightGBM's software prefetch on the bin gather is one of its wins. Requires lifting `unsafe_code = "forbid"` in the histogram module. Not attempted; policy decision.

## Session-final benchmark (Phase E)

`libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json`[^2]:

| Model | fit_time | AUC-ROC | AUC-PR | log-loss |
|---|---|---|---|---|
| lightgbm | 0.72s ± 0.08s | 0.6960 ± 0.0150 | 0.1572 ± 0.0237 | 0.2302 ± 0.0071 |
| cleargbm (session-final) | 1.60s ± 0.02s | 0.6991 ± 0.0132 | 0.1620 ± 0.0259 | 0.2303 ± 0.0068 |
| cleargbm (session-start Phase D) | 6.88s ± 0.13s | (statistical tie with above per prior manifest) | | |

~~Gap closed from 8.0× to 2.2×.~~ **Corrected 2026-07-31: the manifest as written gives 2.4654 / 0.7223 = 3.41×, and the authoritative Phase-E results doc states 8.0× → 3.4×.** The 2.2× above derives from the unsourced 1.60s fit-time cell and should not be quoted[^2][^3]. Quality within seed noise vs LightGBM[^2].

## Phase F additions (2026-07-21 late session)

Two more experiments landed after phase E, both wins. Fresh bench at `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json` — note that filename was later overwritten, so the Phase F figures below are not recoverable from it[^1].

### Experiment 4: ordered-arrays gather-elimination, pooled through hooks (SHIPPED)

**Change.** Same idea as experiment 1 (pre-permute gradients + hessians into position-space so the histogram loop reads them sequentially), but wired as an OPT-IN hook (`BuildHistogramOrderedFn` on `Hooks`), with the reorder pass done ONCE per node and reused across all `n_features` histogram builds — AND reused for the smaller child in `compute_child_histograms` (sibling subtraction path). Files: `libs/cleargbm_rs/src/histogram/mod.rs` (`build_histogram_ordered_trusted` + `reorder_grad_hess_into`), `libs/cleargbm_rs/src/hooks.rs` (opt-in `build_histogram_ordered` hook, `None` in the classic error-injection path), `libs/cleargbm_rs/src/tree/histograms.rs` (fast-path dispatch when hook present).[^5]

**Result vs experiment 1.** Where the unpooled version cost ~4%, the pooled + sibling-subtraction-aware version was roughly neutral to slightly positive by itself (part of the phase-e→phase-f gap; not cleanly isolated in a solo bench). The mechanism does what the tech wiki claimed, but the win is smaller than expected on cleargbm because of the sort-order effect flagged in experiment 1.[^6]

**Why the second attempt didn't lose.** Two things different from experiment 1: (1) the reorder scratch is inside the per-node dispatch, not allocated inside every histogram build — cuts allocation traffic by `n_features`. (2) The smaller-child reorder in `compute_child_histograms` reuses across features too, so the sibling-subtraction path also amortizes.[^6]

**Test coverage.** `libs/cleargbm_rs/src/histogram/tests/unit_tests.rs` gained bit-parity tests: for arbitrary `(sample_indices, gradients, hessians, bins)` the output of `build_histogram_ordered_trusted` (after `reorder_grad_hess_into`) equals `build_histogram_trusted` byte-for-byte. 1124 tests pass.[^5]

### Experiment 5: leaf-cache to bypass predict_tree during training (SHIPPED — BIG WIN)

**Hypothesis.** Profiling showed `predict_tree` at ~34% of total training wall-clock. Every training row walks the freshly-built tree after every boosting round to update raw predictions. But the tree builder ALREADY knows the leaf value for every sample it saw — that's the `leaf_value` computed at every leaf-creation site. Caching it means we only walk the tree for samples the tree wasn't built on (subsample < 1.0 leaves some rows out per round).[^7]

**Change.** New `build_tree_with_leaf_assignment(input, hooks) -> Result<(Tree, Vec<f64>)>` in `libs/cleargbm_rs/src/tree/builder.rs`. The Vec has `f64::NAN` at every index; at each of the two leaf-creation sites (early stop → leaf, or best-split-failure → leaf), we write `leaf_value_per_sample[sample_idx] = leaf_value` for every `sample_idx` in the pending node. `build_tree` becomes a wrapper that drops the Vec. `libs/cleargbm_rs/src/training/train.rs` uses the new API and takes the fast path: for each of `n_train` rows, if `leaf_value_per_sample[i]` is not NaN, just do `raw_preds_train[i] += lr * lv`. NaN sentinel → fall back to `predict_tree` on the collected fallback list. When `subsample=1.0`, EVERY row is fast-path — zero tree walks per round.[^5]

**Result.** cleargbm 1.60s → **1.23s** (**+23% faster**). Quality byte-identical: AUC-ROC 0.6825 vs prior 0.6825 (mean_pred, calibration_slope, log-loss all match to 4 decimals). Verified across seeds 42/43/44. `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`.[^6]

**Why this worked when the others didn't.** Not a "shrink the hot loop" change — it eliminates a whole O(N × trees) tree-walk phase by exploiting information the builder already computes. No new allocation pattern; just a `Vec<f64>` per round sized to `n_train`. The predict_tree fallback path stays intact for subsampling correctness, so no algorithm change, no quality drift.[^6]

**Test coverage.** `build_tree_with_leaf_assignment` is exported and covered by `libs/cleargbm_rs/src/tree/builder.rs` tests. `build_tree` wrapper preserves the old signature for all downstream tests. 1124 lib tests pass.[^5]

## Session-final benchmark (Phase F)

`libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`[^1]:

| Model | fit_time | AUC-ROC | AUC-PR | log-loss |
|---|---|---|---|---|
| lightgbm | 0.84s ± 0.10s | 0.6871 ± 0.0212 | 0.1376 ± 0.0154 | 0.2294 ± 0.0271 |
| cleargbm (phase F) | 1.23s ± 0.08s | 0.6825 ± 0.0185 | 0.1416 ± 0.0181 | 0.2298 ± 0.0268 |

Gap closed to **1.47×** as measured in-session[^6] — the 2.2× starting point inherits the Phase E error corrected above[^3]. AUC-PR now slightly higher than LightGBM (0.1376 vs 0.1416). Quality within seed noise on all other metrics[^6].

## Phase G additions (2026-07-21 later)

### Experiment 6: f32 gradient/hessian narrowing (SHIPPED 2026-07-21, **REVERTED 2026-07-25** — see [[cleargbm-f32-score-narrowing-reverted]])

**SUPERSEDED.** This experiment no longer describes the code. `score_narrow` was deleted and gradients/hessians are `f64` end to end again; the revert's stated reason is that narrowing measured 8% slower once the leaf-cache and ordered-arrays changes had landed. The "+15% faster" result below is also not cleanly attributable to f32 — the commit series bundled f32 with `u32` indices, ordered arrays, and the leaf cache. Read [[cleargbm-f32-score-narrowing-reverted]] for the reconciliation; the record below is retained as the historical account of what was tried.

**Hypothesis.** Per `~/PROJECTS/tech-wiki/pages/lightgbm-score-t-float.md`, LightGBM defaults `score_t = float` (f32) for gradients + hessians while keeping the histogram accumulator `hist_t = double` (f64). Narrow inputs, wide accumulator — halves cache-line pressure on the two hottest streams the histogram loop reads sequentially, while preserving 15-digit precision in the sums via `f64 += f32` implicit widening (LightGBM's C++ pattern).[^7]

**Change.** Type-level lift across the cleargbm hot path:[^5]

- Added `libs/cleargbm_rs/src/narrow.rs` — one gated `#[expect(clippy::as_conversions, clippy::cast_precision_loss, reason = "...")]` site with `pub const fn score_narrow(x: f64) -> f32`. Everywhere else the workspace lints still forbid narrowing.
- `histogram/mod.rs`: `build_histogram`, `build_histogram_ordered_trusted`, `reorder_grad_hess_into` signatures + hot loop take `&[f32]`; the 8-wide unrolled accumulate widens per-element via `f64::from(ordered_gradients[pos + k])` before the add.
- `hooks.rs`: `BuildHistogramFn` type takes `&[f32]`; default hook + all test injection sites updated.
- `tree/histograms.rs`: `BuildHistogramConfig` + `ChildHistogramConfig` field types; per-node reorder scratch `Vec<f32>`.
- `tree/nodes.rs`: `compute_sums(&[usize], &[f32], &[f32]) -> (f64, f64)` widens on read to produce the f64 output the caller expects.
- `tree/builder.rs`: `BuildTreeInput.gradients` + `.hessians` fields.
- `training/train.rs`: gradients + hessians computed directly as `Vec<f32>` via `score_narrow(p - f64::from(y))` and `score_narrow(p * (1.0 - p))`.
- `pyo3_module/{tree_fns,histogram_fns}.rs`: Python numpy sends f64; boundary narrows via `.iter().copied().map(crate::narrow::score_narrow).collect()` before calling the Rust core.
- Every test file that constructed `Vec<f64>` gradients/hessians updated to `Vec<f32>` (histogram/tests/unit_tests.rs, histogram/tests/proptest_tests.rs, tree/tests/builder_tests.rs, tree/tests/error_tests.rs, plus the injected-error histogram helper signatures).

**Result.** cleargbm 1.23s → **1.05s ± 0.02s** (**+15% faster**). LightGBM 0.84s → 0.93s ± 0.01s (bench-to-bench variance; f32 change is on cleargbm's side only, LightGBM is fixed).[^6]

Quality byte-parity except at the f32-precision floor[^6]:

| Metric | Before (f64) | After (f32) | Δ |
|---|---|---|---|
| AUC-ROC | 0.6825 | 0.6822 | −0.0003 (within seed noise) |
| AUC-PR | 0.1416 | 0.1413 | −0.0003 (within seed noise) |
| log-loss | 0.2298 | 0.2300 | +0.0002 (within seed noise) |
| mean_pred | 0.0644 | 0.0644 | (matches to 4 decimals) |
| calibration_slope | 0.6728 | 0.6732 | (matches to 4 decimals) |

**Why this worked.** No new allocation pattern; no split-work-into-more-loops. Pure bandwidth reduction on the sequential-read streams the ordered fast path already gathers into. The gradient/hessian narrowing halves the L1/L2 pressure per histogram build, and the accumulator widening on write costs nothing (Rust generates the same MMX conversion instructions as C++ `double += float`). Guard lints preserved: only ONE narrowing site (`score_narrow`) has the `#[expect]` escape hatch; the rest of the crate still forbids `as` casts and precision-loss.[^6]

**Test coverage.** All 1130 lib tests pass. New `crate::narrow` module has 6 unit tests covering exact-representable cases (0, ±1, ±0.5, 0.25) and one ULP-bounded case.[^5]

## Session-final benchmark (Phase G)

`libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`[^1]:

| Model | fit_time | AUC-ROC | AUC-PR |
|---|---|---|---|
| lightgbm | 0.93s ± 0.01s | 0.6871 ± 0.0212 | 0.1376 ± 0.0154 |
| cleargbm (phase G) | 1.05s ± 0.02s | 0.6822 ± 0.0190 | 0.1413 ± 0.0179 |

Gap closed from 1.47× to **1.13×**. **We are within 13% of LightGBM.** (Superseded by the 2026-07-24 re-measurement — raw 1.426×, per-leaf 0.937×[^8].)

**Cumulative session win:** 6.88s → 1.05s (**6.55× speedup**), gap to LightGBM 8.0× → **1.13×**.[^1][^6]

## Phase H — LTO experiments + u32 sample_indices narrowing

Two more experiments after phase G. Bench noise on the machine climbed sharply during phase H (±0.15s on a 1s mean vs. phase G's ±0.02s) — a background process is confounding sub-5% signals[^6]. Findings recorded here anyway for the design decisions they force.

### Experiment 7: LTO in the release profile (roughly neutral to negative)

**Change.** Added `[profile.release] lto = "fat" codegen-units = 1` to `libs/cleargbm_rs/Cargo.toml`.[^5]

**Result.** cleargbm 1.05s → **1.20s** (**−15%**). LightGBM (unchanged code) shifted 0.93s → 0.85s in the same run — pure bench noise; LightGBM is a fixed reference. Retried with `lto = "thin"`: cleargbm 1.00s ± 0.21s (still noisier than baseline, mean roughly neutral).[^6]

**Root cause hypothesis.** `lto = "fat"` runs LLVM whole-program IR combining across rayon + pyo3 + numpy + our crate, and can OCCASIONALLY regress when the added inlining decisions collide with the already-hot codegen from `target-cpu=native`. cleargbm's hot loop is small and already producing near-optimal SIMD; there's no cross-crate function boundary worth breaking for LTO to widen. Both `fat` and `thin` also add ~3× to build time (10s → 32s).[^6]

**Reverted.** Cargo.toml has both lines commented with the finding recorded inline.[^5]

### Experiment 8: u32 sample_indices (SHIPPED — structurally cleaner, perf neutral within noise)

**Hypothesis.** Per `~/PROJECTS/tech-wiki/pages/lightgbm-score-t-float.md`, LightGBM's `data_size_t` defaults to `int32_t` (4 bytes). cleargbm was using `usize` (8 bytes on x86_64) for sample indices — half the cache-line density on the most-gathered array in the histogram loop.[^7]

**Change.** Type-level lift across ~15 files:[^5]

- New `libs/cleargbm_rs/src/narrow.rs::index_widen(u32) -> usize` — the one gated `#[expect(clippy::as_conversions)]` site for infallible u32→usize widening at slice-access points (`u32::from(usize)` doesn't exist since usize can be 16 bits on some targets; try_from would add a per-iteration branch in the hot loop).
- `subsampling.rs::get_sample_indices` returns `Vec<u32>` (rejects `n_samples > u32::MAX` with `IntegerConversion`).
- `rng.rs::shuffle_partial(&mut [u32])`.
- `histogram/mod.rs`: `build_histogram`, `build_histogram_ordered_trusted`, `reorder_grad_hess_into` take `&[u32]`; hot loop uses `index_widen` at bin-access sites.
- `hooks.rs::BuildHistogramFn` type.
- `tree/histograms.rs`: `BuildHistogramConfig` + `ChildHistogramConfig` `sample_indices` / `left_indices` / `right_indices` fields.
- `tree/nodes.rs`: `PendingNode.sample_indices`, `compute_sums`, `split_samples` (returns `(Vec<u32>, Vec<u32>)`).
- `tree/builder.rs`: `BuildTreeInput.sample_indices`, `leaf_value_per_sample[index_widen(sample_idx)]` at record sites.
- `pyo3_module/{tree_fns,histogram_fns}.rs`: new `i64_slice_to_u32_vec` helper narrows the numpy int64 arrays at the pyo3 boundary.
- All test files that constructed `Vec<usize>` for sample_indices updated to `Vec<u32>`.

**Result.** cleargbm 1.05s → 0.98s ± 0.12s (bench noise ±0.12s). Mean ~7% faster but well inside noise; can't confidently claim a win. Structurally cleaner (matches LightGBM's `data_size_t` shape), no quality change, 1133 lib tests pass. Kept.[^6]

**Test-code observation from this refactor (Austin, 2026-07-21).** Every f64→f32 and usize→u32 lift chased ~50-100 literal edits across `histogram/tests/*`, `tree/tests/*`, `training/tests/*`. That's tests being tightly coupled to concrete types instead of using fixture builders. A follow-up `testkit` lift (generic-input helpers `test_grad<T: Into<f32>>(&[T])` / `test_indices<T: Into<u32>>(&[T])`) would collapse the tax so future type changes touch 3 files instead of 300. Queued as task #38.[^5]

## Phase I — Bench-harness fix (best-of-N kills noise ceiling)

Austin's observation on 2026-07-21: "cant we fix that?" — the ±15% bench noise made all sub-5% phase-H signals uninterpretable[^6]. Fixed by lifting the harness to a **best-of-N** shape. That best-of-N choice was itself later reversed in favour of a median, because a minimum reports a turbo-boosted cold start rather than the steady state[^8].

**Change.** `scripts/benchmark_vs_lightgbm.py` (working copy in scratchpad): added `REPEATS_PER_MODEL = 5`. Each perf-sensitive model (LightGBM + cleargbm) runs 5 fits per seed × 3 seeds = 15 fits total. Report the **minimum fit_time per seed** as the canonical number (min is the physically-correct estimator — the actual work is bounded below by the CPU, any slower run is background noise contamination; the mean systematically overestimates by the average noise contribution, which climbs sharply when a Windows Defender scan or cloud-sync burst intersects the run).[^5]

Rendered per-seed line shows all four: `fit=0.936s (min of 5: min/med/mean/max = 0.936/0.957/0.956/0.965s)` — so the noise band is visible at every readout[^6].

**Result.**[^6]

| Metric | Phase H (single-shot per seed) | Phase I (best-of-5) |
|---|---|---|
| LightGBM fit | 0.83s ± 0.15s (±18%) | **0.95s ± 0.01s (±1%)** |
| cleargbm fit | 0.98s ± 0.12s (±12%) | **1.05s ± 0.01s (±1%)** |
| discriminable Δ | ~15% floor | **~1% floor** |

Same code, radically different signal-to-noise[^6]. The bench is now a real perf harness — the 5-15× tightening in stddev makes every sub-10% experiment on the queue actually resolvable. Cost: 5× wall-clock per bench invocation. (The harness described here was subsequently lost with its session scratchpad and rebuilt on a median estimator[^8].)

## Session-final benchmark (Phase I — stable)

> **SUPERSEDED 2026-07-24.** The gap figure below does not reproduce, for two reasons, and should not be quoted. (a) The harness that produced it lived only in a session scratchpad and was lost; every later claim was measured on the noisier phase-E shape. (b) Its canonical statistic was the *minimum* of 5, which reports a turbo-boosted cold-start run rather than the steady state — see [[cleargbm-leaf-normalized-benchmarking]] for the evidence and the median-based replacement. Re-measured on a rebuilt harness, cleargbm is **1.2809s ± 0.0638** against LightGBM **0.8981s ± 0.0719** (raw 1.426×), and **0.937× per leaf** once the depth-wise/leaf-wise tree-size difference is divided out. The cleargbm figures below are also not directly comparable to that run, because this session's numbers were taken with an estimator that flatters whichever model has the shorter fit.

`libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`[^1]:

| Model | fit_time (best of 5 × 3 seeds) | AUC-ROC | AUC-PR |
|---|---|---|---|
| lightgbm | **0.95s ± 0.01s** | 0.6871 ± 0.0212 | 0.1376 ± 0.0154 |
| cleargbm | **1.05s ± 0.01s** | 0.6822 ± 0.0190 | 0.1413 ± 0.0179 |

Per-seed cleargbm/lightgbm ratios: 1.113, 1.072, 1.121 → tight cluster[^1].

**Gap: 1.10× (10% slower).**[^1]

**Cumulative session win:** 6.88s → 1.05s (**6.55× speedup**, verified at ±1% precision), gap to LightGBM 8.0× → **1.10×**.[^1][^6]

## Meta-lesson update

The two shipped positives (ordered-arrays with proper amortization + leaf-cache) share a pattern that ALL three negatives lacked[^6]:

- Ordered-arrays-pooled amortized the reorder across `n_features` histogram builds per node AND across the smaller-child sibling-subtraction path. The unpooled version (experiment 1) allocated per histogram build and lost. **Amortization ratio matters.**
- Leaf-cache exploits information the tree builder already produces (the leaf value for every in-sample row). Zero re-computation, zero new algorithm. **Reuse over re-derive.**

Both wins came from restructuring who owns already-computed information, not from making the hot loop tighter. That is a different search direction than "shrink bytes per sample in the histogram hot path" — the space that experiments 1-3 explored. Future experiments should look for other computed-but-discarded data (per-sample bin-column reuse across trees? cross-tree gradient/hessian precomputation?) before more hot-loop micro-optimization.
[^1]: `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json` — exactly two committed versions (`git log -- <path>`, checked 2026-07-31). At `67ce10c8` "cleargbm: Phase D+E — post-refactor re-benchmark + docs refresh": mean over the three `all_runs` records is cleargbm **6.8836s ± 0.1585**, lightgbm **0.8692s ± 0.1149** — this is the session baseline. At `4b9e2427` "cleargbm: refresh `BENCHMARK_MANIFEST_2026-07-21.json` for phase-I best-of-5 harness run" (the version on disk today): cleargbm **1.0456s ± 0.0102**, lightgbm **0.9489s ± 0.0166**, which is what the Phase G and Phase I tables report. No committed version contains the Phase F figure of 1.23s. Recover the baseline with `git show 67ce10c8:libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`.
[^2]: `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json` — two committed versions. As first written at `6a2d15b7`: cleargbm **2.4654s ± 0.0400**, lightgbm **0.7223s ± 0.0938**. After `c739e56f` "refresh … after leaf-cache landing" (on disk today): cleargbm **1.7889s ± 0.0642**, lightgbm **0.7260s ± 0.0876**. Both versions carry identical quality metrics — lightgbm AUC-ROC 0.6960 ± 0.0184 / AUC-PR 0.1572 ± 0.0290 / log-loss 0.2302 ± 0.0087, cleargbm 0.6991 ± 0.0161 / 0.1620 ± 0.0317 / 0.2303 ± 0.0083 — which is why this page's Phase E quality columns reproduce exactly while its fit-time column matches neither.
[^3]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21_phase_e.md:11,13` — "Fit time dropped from **6.88s ± 0.13s** to **2.47s ± 0.03s** on the 200-tree / depth-6 / max_bins=64 config" and "**Gap to LightGBM: 8.0× → 3.4×.**" The authoritative Phase-E statement, consistent with the manifest as written[^2] and inconsistent with this page's 1.60s / 2.2×.
[^4]: Commit subjects verified 2026-07-31 with `git log -1 --format=%s <sha>`: `eaa31ca7` "cleargbm perf: column-major uint8 sample_bins + unrolled histogram loop (wiki items 1-3)"; `13a235e0` "cleargbm perf: rayon-parallel histograms + target-cpu=native"; `2db497a6` "cleargbm perf: trusted histogram fast path + fused pre-validation + rayon threshold"; `48f89591` "cleargbm perf: 8-wide unrolled histogram loop + #[inline] on trusted path". All four exist and their subjects match the change descriptions in the table. Batch E has no commit by the page's own account (build-side variance settle), so nothing anchors its 1.66s → 1.60s step.
[^5]: Code paths touched, verified present 2026-07-31: `libs/cleargbm_rs/src/histogram/mod.rs`, `libs/cleargbm_rs/src/tree/histograms.rs`, `libs/cleargbm_rs/src/tree/builder.rs`, `libs/cleargbm_rs/src/binning/feature_bins.rs`, `libs/cleargbm_rs/src/hooks.rs`, `libs/cleargbm_rs/src/training/train.rs`. The current state of the histogram hot path is described with line-level receipts in [[cleargbm-perf-simd-histogram-accumulator]].
[^6]: **In-session measurement, no committed artifact.** The fit-time deltas attached to individual experiments (the −4% / −5% / −16% / +23% / +15% style figures) were taken by running the benchmark during the 2026-07-21 session; only the phase-boundary snapshots were ever written to a manifest, and those files were later overwritten[^1][^2]. They are recorded here for the design decisions they drove and the direction they indicate, not as reproducible measurements. The same failure mode is documented for a different figure in [[cleargbm-f32-score-narrowing-reverted]] § "Why it was reverted", and the harness that would make such numbers reproducible is [[cleargbm-leaf-normalized-benchmarking]].
[^7]: Cross-wiki reference into `~/PROJECTS/tech-wiki/pages/` — outside this workspace root, so it cannot appear in `source_paths` and is not resolvable by this wiki's audit. Cited as provenance for where the hypothesis came from, not as a verified claim of this page.
[^8]: [[cleargbm-leaf-normalized-benchmarking]] — the rebuilt median-of-repeats harness and the authoritative 2026-07-24 re-measurement (cleargbm 1.2809s ± 0.0638 vs lightgbm 0.8981s ± 0.0719; raw 1.426×, per-leaf 0.937×), plus the protocol properties that explain why this session's numbers are not directly comparable.
[^9]: Verified 2026-07-31 — a grep for `min_sum_hessian` and `min_child_weight` across `libs/cleargbm_rs/src` and `libs/cleargbm/src` returns **no match**. The only leaf-size regularizers on `GradientBoostingConfig` are `min_samples_split` and `min_samples_leaf` (`libs/cleargbm_rs/src/training/config.rs:22,24`), both of which are exact sample counts — so an approximated `counts` array would degrade the only constraint enforcing leaf size. The `max_bins` sweep figures quoted in this block are in-session measurements with no committed artifact[^6].
