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
  four-decimal tie with LightGBM (was 0.6874) at better auc_pr.
- Remaining named work: the rw_matches binary flagship re-baseline
  (through radar's optimize pipeline) rides the next farm rung.

Full numbers: `BENCHMARK_RESULTS_2026-08-25_count_aware_binning.md`;
manifests `BENCHMARK_MANIFEST_2026-08-25_binning_*` + rung-2 results in
`tools/hpc3/runs/results/p6-rung2/`.
