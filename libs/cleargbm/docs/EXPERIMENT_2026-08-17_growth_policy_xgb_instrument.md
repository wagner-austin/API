# Growth-policy experiment: leaf-wise measured via XGBoost before building it — 2026-08-17

**Question:** what would `growth_strategy: leaf_wise` (board task `453c9234`) actually buy
ClearGBM on the authoritative benchmark workload?

**Instrument:** XGBoost 3.1.2 implements both growth policies, so the leaf-wise effect can
be measured with everything else held constant — same code, same splits, same constraint
semantics — before any Rust is written. Script:
`libs/cleargbm/scripts/experiment_growth_policy_xgb_instrument.py`.

## Protocol

- Dataset: `libs/covenant_ml/tests/data/american_bankruptcy.csv` — 78,682 × 18, 6.63% positive.
- Split: company-disjoint 70/30 re-permuted per seed; seeds 42/43/44; median-of-3 fit time with 1 warmup; `n_jobs=1`.
- Shared config: 200 estimators, lr 0.05, max_bin 64, min-leaf 20, reg_alpha 0, reg_lambda 0.
- Arms: xgb `depthwise` max_depth=6 · xgb `lossguide` max_leaves=31 · xgb `lossguide` max_leaves=47 (ClearGBM's measured mean) · LightGBM num_leaves=31 anchor · ClearGBM max_depth=6 anchor.

## Result (means over 3 seeds)

| arm | fit s | AUC-ROC | AUC-PR | log-loss | mean leaves |
|---|---|---|---|---|---|
| xgb depthwise d6 | 0.772 | **0.7005** | **0.1670** | **0.2340** | 22.8 |
| xgb lossguide L31 | 0.956 | 0.6962 | 0.1612 | 0.2367 | 31.0 |
| xgb lossguide L47 | 1.183 | 0.6937 | 0.1586 | 0.2396 | 47.0 |
| lgb leafwise L31 | 0.943 | 0.6994 | 0.1632 | 0.2349 | 31.0 |
| cleargbm depthwise d6 | 1.706 | 0.7004 | 0.1626 | 0.2356 | ~47.15 (from BENCHMARK_MANIFEST_2026-07-24) |

## Reading

1. **Within the instrument, leaf-wise monotonically hurts quality on this dataset.**
   Depthwise (22.8 leaves) > lossguide-31 > lossguide-47 on every quality metric. On a
   6.6%-positive workload, added tree capacity is overfit, not signal.
2. **The prize for a leaf-wise ClearGBM is therefore work-at-tied-quality, not quality.**
   The quality ceiling is reachable at ~23 leaves; ClearGBM currently spends ~47
   depth-wise leaves to tie it. Gain-ordered growth stopping at ~23-31 leaves plausibly
   ties quality at roughly half the tree-building work — which is the residual 1.84×
   wall-clock gap (see `cleargbm-leaf-normalized-benchmarking` in the api wiki: ClearGBM
   is already ~0.94× LightGBM per leaf).
3. **Acceptance framing for the leaf-wise arm** (feeds task `453c9234`): judge
   `cleargbm@leaf_wise` primarily on fit-time at statistically tied quality vs the
   depth-wise baseline, with quality regression as the guarded downside — NOT on quality
   improvement, which this measurement says not to expect here.

## Confounds, stated

- XGBoost's `min_child_weight` is a **hessian sum**; at 6.6% positive with p near the
  base rate, 20 hessian units is roughly ≥320 samples per leaf, so the depthwise arm is
  more heavily regularized than LightGBM/ClearGBM's count-based min-leaf 20. The
  within-XGBoost contrast is unaffected (all three arms share the constraint), but the
  22.8-leaf figure is partly that constraint at work, and cross-library absolute
  comparisons here are looser than the within-instrument ones.
- One dataset. The direction (small trees win) is a property of this workload's class
  balance and size, not of leaf-wise growth in general; Shi 2007 and the LightGBM paper
  measured the opposite on other workloads.
- Not run through the covenant_ml benchmarking harness (no alternating-order protocol,
  no manifest); repeat under the harness once variant arms exist. Timing context: on
  this machine tonight the three-way baseline was cleargbm 1.634 / lgb 0.889 / xgb 0.779.

## Dataset-variety follow-up (same night): the lever does not engage on small data

The same three arms were run on two additional on-disk datasets
(`scripts/experiment_growth_policy_multi_dataset.py`, stratified random 70/30, seeds
42/43/44):

| dataset | shape | positive | depthwise d6 | lossguide L31 | lossguide L47 | mean leaves |
|---|---|---|---|---|---|---|
| taiwan-bankruptcy (`data/external/kaggle_taiwan_bankruptcy`) | 6,819 × 95 | 3.23% | AUC 0.9516 | 0.9516 | 0.9516 | 4.4 in ALL arms |
| german-credit (`data/external/german_credit`) | 1,000 × 20 | 30.0% | AUC 0.7664 | 0.7664 | 0.7664 | 4.6 in ALL arms |

**All three arms are identical to four decimals on both datasets, with identical mean
leaf counts.** The `min_child_weight=20` (hessian-sum) constraint stops tree growth at
~4.5 leaves before either the depth budget or the leaf budget binds, so growth policy
never gets to make a decision. This is a null, not a contradiction: growth policy is a
live lever only when data volume lets trees grow to where the budgets bind (the 78k-row
American dataset), and it is moot under this regularization on small datasets. For the
leaf-wise ClearGBM arm this narrows the claim: the work-at-tied-quality prize is
specific to large-n workloads; small-n workloads will not distinguish the variants at
all under standard min-leaf regularization.
