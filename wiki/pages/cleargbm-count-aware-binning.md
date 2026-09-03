---
title: ClearGBM count-aware binning — the LightGBM gaps close at the root
tags: [ml, cleargbm, binning, histogram, quality, roadmap-p6]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-hpc3-farm-and-rw-value]]"
source_paths:
  - libs/cleargbm_rs/src/binning/edges.rs
  - tools/hpc3/runs/sweep-cleargbm-p6-rung2.json
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-25_count_aware_binning.md
source_git_blobs:
  "libs/cleargbm_rs/src/binning/edges.rs": be806fdfb21bf1461b0ff1db9be7ace47e57fef0
  "tools/hpc3/runs/sweep-cleargbm-p6-rung2.json": 5c66991f2efd3634f79cb1f5bf4e527fd1899b86
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-25_count_aware_binning.md": 36471731297364d6294fb7e65ad9e40c4b2237e2
fact_checked: "2026-08-25"
confidence: high
hubs: [libs]
---

# ClearGBM count-aware binning — the LightGBM gaps close at the root

The P6 scoreboard's two adverse corpora (LightGBM led [[cleargbm-hpc3-farm-and-rw-value]]'s
weather_tmax and voc_match_quality) are closed by one root-cause fix in
`cleargbm_rs::binning::edges` (board task `1ad15eb6`).

## The mechanism

ClearGBM binned by quantile positions of the sorted value MULTISET,
deduplicated. Skewed features starve: weather's zero-inflated
`hot_excess` (2,359 distinct values, ~95% zeros) collapsed to **6 bins**
at a 64 budget because nearly every quantile position landed on the
zero; voc's `species` (41 codes) merged into 36 bins while 23 sat
unused. LightGBM's `GreedyFindBin` works on (distinct value, count)
pairs and does neither.

Two independent lines of evidence pinned it before the fix: a direct
bin-count probe on the loaded corpora (the table in the results doc),
and HPC3 rung 2 — 48 members over both corpora x a 24-point protocol
grid under OLD binning — showing the weather gap systematic across all
24 configs but SHRINKING with raw bin budget (+0.152% at 64 bins →
+0.108% at 255), and the voc gap flipping to a ClearGBM lead at 255,
exactly what quantile starvation predicts.

## The fix

Count-aware edges per LightGBM's shipped `GreedyFindBin` semantics
(bin.cpp, tech-wiki pin `lightgbm-bin-cpp.html`) with `min_data_in_bin`
fixed at 1 — NO new config knob: one bin per distinct value when they
fit the budget (midpoint edges, partition-exact under adjacent-double
collapse); otherwise greedy equal-count bins with heavy values taking a
bin of their own. Config surface and per-config determinism unchanged;
results change relative to old binning, so the whole standing board was
re-measured (the lightgbm/xgboost arms reproduce bit-identically as
anchors).

## The scoreboard after

- **weather_tmax: ClearGBM leads** — cleargbm@leaf_wise 2.068105 vs
  lightgbm 2.069525 (was 2.073458 trailing); wins 3/2.
- **voc_match_quality: both ClearGBM arms lead** — leaf_wise 18.172941,
  cleargbm 18.195941 vs lightgbm 18.316997 (a 0.7% deficit became a
  0.8% lead); wins 4/1.
- rw_value stays ClearGBM-led and improves (175,476); metab_confidence
  stays the weak-signal tie; financial_distress joins the registry
  harness as a near-tie (no old-binning baseline under this harness).
- us binary head-to-head: cleargbm@leaf_wise auc_roc 0.6881 — an exact
  four-decimal tie with LightGBM (was 0.6874) at better auc_pr; wall
  clock re-baselined at 1.312x raw / 0.863x per-leaf.
- **The honest cost — the rw_matches flagship lead was a binning
  artifact.** Re-run through the identical weighted grouped-5-fold
  protocol (LightGBM reproduces its standing 0.7299 ± 0.0749 EXACTLY,
  proving protocol identity), ClearGBM's 0.7492 became 0.7295: matched
  binning, matched quality. The old coarse binning suppressed
  tail-noise on rw_matches' zero-inflated counters (a regularizer by
  accident) while destroying tail-signal on weather/voc. One rule
  cannot maximize both — which names a `min_data_in_bin`-style
  coarseness knob (LightGBM ships 3; ours is fixed at 1) as a
  corpus-named candidate for a future landing.
- The scoreboard after all of it: ClearGBM leads weather_tmax,
  voc_match_quality and rw_value; ties rw_matches, metab_confidence
  and the us binary; financial_distress is a near-tie nominally led by
  xgboost. No material LightGBM lead anywhere.

Full numbers: `BENCHMARK_RESULTS_2026-08-25_count_aware_binning.md`;
manifests `BENCHMARK_MANIFEST_2026-08-25_binning_*` + rung-2 results in
`tools/hpc3/runs/results/p6-rung2/`.
