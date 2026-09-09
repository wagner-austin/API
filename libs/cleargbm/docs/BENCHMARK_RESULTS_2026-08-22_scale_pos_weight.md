# 2026-08-22 — scale_pos_weight implemented: the decorative knob made real, +1.3 AUC points on rw_matches

Agent-board task `cad8d8e6`, the follow-up to the config-honoring fix
(996bf364). The covenant_ml cleargbm backend had computed, logged and
reported a class weight the core could not apply — every
ClearGBM-vs-LightGBM comparison on imbalanced data ran weighted LightGBM
against unweighted ClearGBM.

## What was implemented

Weighted binary log loss end to end in cleargbm_rs, positive samples scaled
by `scale_pos_weight` (finite, > 0, required with no default — the same
no-silent-absence contract as `growth_strategy`):

- gradients `w * (p - 1)` and hessians `w * p(1-p)` for positives in the
  boosting loop;
- the base score from the weighted prevalence
  `log((w * n_pos) / (w * n_pos + n_neg))` — boost-from-average under
  weights;
- the early-stopping validation loss as the weighted mean, so the stopping
  criterion tracks the objective actually trained;
- config field validated at construction, serialized into the model JSON,
  parsed at the pyo3 boundary; threaded through cleargbm's Python config
  and every construction site in the workspace.

The covenant_ml backend now passes its auto-computed weight instead of
logging it into the void. The benchmark, growth-policy and optimizer arms
pin 1.0 explicitly (matching their unweighted LightGBM/XGBoost
comparators). New tests: weighted-vs-unweighted training differs
(knob-sensitivity), exact weighted loss and base-score values, validation
rejections, and a covenant_ml integration test asserting the SAVED MODEL's
config records the computed imbalance ratio — the dataflow proof this bug
class evades.

## Equivalence gate: w = 1.0 is bit-identical

Every weighted formula reduces to the historical operation sequence at
weight 1.0 (multiplies by exactly 1.0; integer-valued f64 count sums).
Verified by measurement, not just construction: the four-arm benchmark
rerun under the weighted crate reproduces the 2026-08-21 single-pass
manifest's quality metrics and leaf counts byte-for-byte on every cleargbm
arm and seed. All recorded manifests remain valid.

## The measurement: rw_matches grouped 5-fold CV

569,561 rows, 99 match groups, production config (300 trees, depth 5,
early stopping). The auto-computed weight on this corpus's imbalance is
now applied:

| fold | reg-fixed, unweighted | weighted |
|---|---|---|
| 0 | 0.8343 | 0.8362 |
| 1 | 0.6111 | 0.6153 |
| 2 | 0.7356 | **0.7901** |
| 3 | 0.7009 | 0.7270 |
| 4 | 0.8005 | 0.7776 |
| **mean** | **0.7365 ± 0.0783** | **0.7492 ± 0.0754** |

Four of five folds improve, the mean gains 1.3 AUC points, the spread
tightens. LightGBM's weighted result on the identical protocol is
0.7299 ± 0.0749: with both engines finally weighting the same way,
ClearGBM leads by ~2 points of mean held-out AUC on the production corpus.

### Power: neither number above is resolved by this design

Added by the power audit, board `1e4ab572`, 2026-09-09. **The ± figures in
the table are the ACROSS-FOLD spread within each arm, which is not the
dispersion either comparison needs.** Both arms run on the same five folds,
so the fold-to-fold movement is common-mode and cancels under pairing. The
paired instrument, from the per-fold values published above:

| quantity | value |
|---|---|
| mean paired difference | +0.01276 (the "1.3 AUC points") |
| paired sd of the differences | 0.02908 |
| across-fold sd, as tabled | 0.0754 / 0.0783 — **2.7× larger** |
| MDE, `t_crit(4)·sd/√n` | **0.03611** |
| paired t | 0.981 on 4 df (`t_crit` 2.776) |

**The observed effect is 0.35× its own detection floor: NOT TESTED.** Fold 4
moves −0.0229 against fold 2's +0.0545, and at n=5 that spread swamps the
mean. "Four of five folds improve" is a 4-of-5 sign split, which alone is
p = 0.375.

**The ~2-point lead over LightGBM cannot be tested at all from this page**,
because LightGBM's per-fold values are not published — only its mean and
across-fold sd. `platform_core.minimum_detectable_effect` takes actual
per-replicate differences and refuses a summary statistic, by design.

**What this does NOT undermine.** Shipping weighted log-loss was correct
independently of the effect size: the prior comparison ran weighted LightGBM
against unweighted ClearGBM, which is not a like-for-like comparison at any
n. The fix removed a real asymmetry. What is unsupported is the MAGNITUDE —
"+1.3 points", and the "1.5 points combined" in the ledger below that
inherits from it. **Stated threshold of practical interest: 0.013 AUC**, the
size of this claim itself, since it is the smallest difference this program
has acted on.

**The cheapest repair is one column.** Publishing LightGBM's five per-fold
values, and the per-fold values for every other rw_matches CV claim, makes
all of them checkable — this is the only one of 24 `BENCHMARK_RESULTS_*.md`
documents that publishes a per-fold table at all, which is why it is the
only one this audit could test.

## The day's bug-class ledger

Three same-class defects (config value silently not reaching training)
found and dispositioned on 2026-08-22: reg_lambda et al hardcoded
(996bf364, fixed, +0.2 pts), scale_pos_weight decorative (this change,
+1.3 pts), max_features/track_contributions dropped at the Rust boundary
(documented no-ops, still open as an operator decision). Combined measured
cost of the two fixed bugs on rw_matches: 1.5 points of held-out AUC that
the engine was silently leaving on the table. The detector that catches
this class is knob-sensitivity testing — types, coverage and
completion-asserting tests all pass while a value goes nowhere.
