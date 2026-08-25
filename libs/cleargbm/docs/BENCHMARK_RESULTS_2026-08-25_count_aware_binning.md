# Count-aware binning: the LightGBM gaps close (2026-08-25)

The P6 scoreboard carried two adverse corpora — LightGBM led weather_tmax
and voc_match_quality — and this landing closes both at the root, with
the mechanism named, reproduced, fixed, and re-measured across the whole
standing board.

## The mechanism: quantile-of-multiset starvation

ClearGBM's bin edges were quantile positions of the sorted value
MULTISET, deduplicated. On skewed features that starves resolution two
ways, measured on the real corpora before the fix:

| corpus | feature | distinct values | ClearGBM bins @ 64 budget |
|---|---|---:|---:|
| weather_tmax | hot_excess | 2,359 | **6** |
| weather_tmax | cold_excess | 1,605 | **4** |
| voc_match_quality | species | 41 | **36** |
| voc_match_quality | peaks_within_1min | 19 | **13** |

`hot_excess` is zero on ~95% of days, so nearly every quantile position
landed on the zero and dedup collapsed the tail — the feature carrying
the extreme-day signal was quantized to six levels. On voc, six pairs of
plant species were literally inseparable by the model while 23 bins sat
unused. LightGBM does neither: its `GreedyFindBin` works on (distinct
value, count) pairs.

## The fix

`cleargbm_rs::binning::edges` now bins count-aware, following the
shipped semantics of LightGBM's `GreedyFindBin` (src/io/bin.cpp,
archived in the tech-wiki as `lightgbm-bin-cpp.html`) with
`min_data_in_bin` fixed at 1 — no new config knob:

- distinct values fit the budget → one bin per distinct value, edges at
  midpoints between neighbours (guarded to stay partition-exact when
  adjacent doubles collapse);
- otherwise greedy equal-count bins, with any value heavier than the
  mean bin size taking a bin of its own and the budget re-spreading
  over the rest.

Documented divergences from the shipped code: no `min_data_in_bin` knob,
and the running mean is not refreshed once the rest-bin budget hits zero
(the shipped code divides by zero there). Config surface unchanged;
per-config determinism holds as always — this changes results relative
to the OLD binning, which is exactly what the quality gate below
measures.

## Corroboration from the farm (rung 2, OLD binning)

Before the fix landed, a 48-member HPC3 rung
(`sweep-cleargbm-p6-rung2.json`: both adverse corpora x lr
{0.03,0.05,0.1} x bins {64,255} x depth/leaves {6/31,8/63} x min-leaf
{5,20}, four arms, seeds 42-46 each — 960 arm-seed fits) characterized
the gap across the protocol space, all 48 members green:

- **weather_tmax**: best-ClearGBM trailed LightGBM in ALL 24 configs —
  the gap was systematic, not protocol luck — but shrank with raw bin
  budget (mean +0.152% at 64 bins → +0.108% at 255), exactly what
  quantile starvation predicts (255 multiset-quantiles still mostly
  land on the zeros).
- **voc_match_quality**: the gap tracked bins even more strongly — dead
  even at 64 (ClearGBM ahead in 7/12), ClearGBM ahead in 9/10 at 255
  (mean -0.311%), where the 255 budget restores per-species resolution
  that count-aware binning provides at 64.

Manifests: `tools/hpc3/runs/results/p6-rung2/` (48 files, per-job logs
on the cluster).

## The quality gate: the whole standing board, before → after

Same registry harness, same P1 protocol, same seeds 42-46; the
lightgbm/xgboost arms are bit-identical across old and new manifests
(their libraries were untouched), which anchors the comparison.

**weather_tmax** (mean test RMSE, grouped by station) — **gap closed,
ClearGBM leads**:

| arm | old | new |
|---|---|---|
| cleargbm@leaf_wise | 2.073458 | **2.068105** |
| lightgbm | 2.069525 | 2.069525 |
| xgboost | 2.070401 | 2.070401 |
| cleargbm | 2.075148 | 2.071119 |

Per-seed wins: was lightgbm 4 / xgboost 1; now **cleargbm@leaf_wise 3 /
lightgbm 2**.

**voc_match_quality** (mean test RMSE, grouped by site) — **gap closed,
both ClearGBM arms lead**:

| arm | old | new |
|---|---|---|
| cleargbm@leaf_wise | 18.480583 | **18.172941** |
| cleargbm | 18.442302 | 18.195941 |
| lightgbm | 18.316997 | 18.316997 |
| xgboost | 18.564598 | 18.564598 |

A 0.7% deficit became a 0.8% lead. Per-seed wins: was lightgbm 3; now
**cleargbm@leaf_wise 3 / cleargbm 1 / lightgbm 1**.

**No regressions on the rest of the board:**

- rw_value: cleargbm keeps the lead and improves (175,525 → 175,476 vs
  lightgbm 176,091); leaf_wise slips slightly (175,819 → 176,232) but
  the corpus stays ClearGBM-led.
- metab_confidence: still the weak-signal statistical tie (cleargbm
  0.129497 → 0.129414; full spread 0.05% of RMSE).
- financial_distress joins the registry harness for the first time
  (no old-binning baseline exists under THIS harness; the P1-era
  number came from a different split protocol): near-tie, xgboost
  1.803805, lightgbm 1.809619, cleargbm@leaf_wise 1.810075, cleargbm
  1.810951.
- us binary head-to-head (the canonical binary harness, 200 trees,
  seeds 42-44): cleargbm@leaf_wise auc_roc 0.6874 → **0.6881 — now
  exactly tying LightGBM at four decimals** — with the better auc_pr
  (0.1410 vs 0.1366); depth-wise 0.6824 → 0.6832. Wall clock
  re-baselined at the standing protocol (repeats 5, warmups 2): raw
  ratio 1.312x vs LightGBM, per-leaf 0.863x
  (`BENCHMARK_MANIFEST_2026-08-25_binning_us_binary_timed.json`).

## The honest cost: the rw_matches flagship lead was a binning artifact

The standing scoreboard's headline — ClearGBM 0.7492 vs LightGBM 0.7299
mean held-out AUC on rw_matches grouped 5-fold CV — did NOT survive the
fix, and the reason is instructive. Re-run through the identical
protocol (radar's `scripts.cv_external`, production config, weighted,
569,561 rows / 99 groups / 5 folds / seed 42; the LightGBM arm
reproduces its standing number EXACTLY, 0.7299 ± 0.0749, proving
protocol identity):

| arm | old binning | new binning |
|---|---|---|
| cleargbm | **0.7492 ± 0.0754** | 0.7295 ± 0.0728 |
| lightgbm | 0.7299 ± 0.0749 | 0.7299 ± 0.0749 |

Matched binning produced matched quality: the 2-point lead was
substantially a REGULARIZATION artifact of the old coarse
quantile-of-multiset binning. rw_matches' zero-inflated counter
features carry noise in their tails at 99-group correlation, and the
old rule's accidental coarseness suppressed it; weather and voc carry
SIGNAL in their tails, and the same coarseness destroyed it. One
binning rule cannot maximize both, which NAMES the missing dial: a
`min_data_in_bin`-style coarseness knob (LightGBM itself ships
min_data_in_bin=3; ours is fixed at 1, slightly finer than LightGBM).
That knob is now a corpus-named candidate for a future landing under
the config-honesty rules — not silently defaulted, and not added in
this one.

## Rung 3: the bankruptcy family re-baselined (16 members, new binning)

The full rung-1 grid (4 datasets x 4 feature presets, 20 Optuna trials
each through radar's `scripts.optimize`) re-ran on HPC3 under the new
wheel — 320 trials, 0 pruned, 0 failed. Best validation AUC per member:

| dataset | none | log_only | ratios_only | full |
|---|---|---|---|---|
| taiwan | 0.9481 | 0.9557 | 0.9538 | 0.9412 |
| us | 0.8146 | 0.8146 | 0.8372 | 0.8438 |
| polish | 0.9622 | 0.9622 | 0.9547 | 0.9483 |
| kaggle_give_me_credit | 0.8691 | 0.8701 | 0.8687 | 0.8691 |

At the per-dataset tuned best — the number the standing tracks — the
family is a statistical wash vs rung 1's old-binning grid: taiwan
0.9575 → 0.9557, us 0.8457 → 0.8438, polish 0.9593 → **0.9622**,
kaggle 0.8701 → 0.8701 exactly. Two slightly down, one up, one tied,
at deltas comparable to 20-trial Optuna noise. No systematic
regression; "tie on the bankruptcy family" stands.

One structural change worth recording: `log_only` no longer ties
`none` to four decimals (taiwan 0.9557 vs 0.9481). The old at-value
edges made histogram training AND prediction invariant to per-feature
monotone transforms; the new midpoint edges keep training partitions
invariant but place thresholds differently for UNSEEN values (a raw
midpoint is a log-space geometric mean), so validation routing — and
hence early stopping and the tuner's path — can differ. The preset is
no longer provably decorative for tree backends; it is merely usually
irrelevant.

## The standing after everything

ClearGBM leads weather_tmax, voc_match_quality and rw_value;
statistically ties rw_matches (0.7295 vs 0.7299), metab_confidence,
the us binary head-to-head, and the bankruptcy family;
financial_distress is a near-tie nominally led by xgboost (full
four-arm spread 0.4% of RMSE). No corpus shows a material LightGBM
lead anywhere on the board. Wall clock 1.312x raw, 0.863x per leaf.

Manifests: `BENCHMARK_MANIFEST_2026-08-25_binning_*_quality.json`
(weather_tmax, voc_match_quality, rw_value, metab_confidence,
financial_distress, us_binary).

## Gates at landing

- cleargbm_rs: full gate green — clippy, 1,540+ tests, 100.00% segment
  coverage. The new binning carries hand-computed golden tests: exact
  per-value bins, the zero-inflated 90/10 shape (old rule: 2 bins; new:
  8), heavy-value carve-outs, rest-budget exhaustion, and the
  adjacent-double midpoint collapse (an odd-mantissa pair forces
  round-half-to-even onto the upper value).
- cleargbm (python) and covenant_ml gates: green at landing (see the
  commit).
