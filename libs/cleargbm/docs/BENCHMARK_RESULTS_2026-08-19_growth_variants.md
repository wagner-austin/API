# Benchmark 2026-08-19 — depth-wise vs leaf-wise vs LightGBM

First measurement of the `cleargbm@leaf_wise` arm. Manifest:
[`BENCHMARK_MANIFEST_2026-08-19_variants.json`](BENCHMARK_MANIFEST_2026-08-19_variants.json)
(schema 2).

Reproduce with:

```
cd libs/covenant_ml
poetry run python scripts/benchmark_cleargbm_vs_lightgbm.py --variants \
    --out ../cleargbm/docs/BENCHMARK_MANIFEST_2026-08-19_variants.json
```

## Setup

American bankruptcy, 78,682 rows x 18 features,
sha256 `cff2c899a97ecd62…`. 200 trees, `max_depth` 6, 64 bins,
`num_leaves` 31, `n_jobs` 1. Seeds 42/43/44, 2 warm-ups + 5 timed fits per arm
per seed, median per seed. Company-disjoint splits; arm order rotates one slot
per seed, so across three seeds each arm occupies each slot exactly once.

## Results

| arm | fit (mean of per-seed medians) | leaves/tree | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| `cleargbm` (depth-wise) | 1.0131s ± 0.0188 | 47.06 | 0.6824 | 0.1416 |
| `cleargbm@leaf_wise` | 0.9451s ± 0.0509 | 30.98 | 0.6874 | 0.1383 |
| `lightgbm` | 0.5147s ± 0.0140 | 30.96 | 0.6881 | 0.1366 |

Against LightGBM: raw 1.968x, leaf 1.520x, **per-leaf 1.295x**.

## What the leaf-wise arm shows

**Fit time is ~6.7% lower at 34% fewer leaves, with quality tied.** AUC-ROC
moves +0.0050 and AUC-PR −0.0033; both differences are smaller than the spread
across the three arms (0.0057 and 0.0050 respectively), so this is a tie, not
an improvement.

**The speed-up is entirely from building fewer leaves, not from being
cheaper.** Per leaf, leaf-wise costs *more*: 0.0305 s/leaf against depth-wise's
0.0215 s/leaf, i.e. **42% more expensive per leaf**. Best-first picks the
highest-gain leaf, which is typically one of the largest remaining nodes, so
its splits are concentrated on expensive nodes; depth-wise spends much of its
budget on cheap nodes deep in the tree. Anyone reading the 6.7% as "leaf-wise
made the builder faster" has it backwards.

**Do not treat 6.7% as established.** The leaf-wise arm's own per-seed spread
(±0.0509) is comparable to the gap (0.068s), and n = 3 seeds. The honest
reading is that leaf-wise reaches the same quality with a third fewer leaves
and is not slower.

This is consistent with the prediction recorded in
[`EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md`](EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md):
judge the arm on fit time at statistically tied quality, and do not expect a
quality gain. Using XGBoost as an instrument, leaf-wise growth *hurt* AUC on
this workload; here it does not hurt it, but it does not help either.

## The measurement defect this run exposed

The first two attempts at this benchmark reported `cleargbm` at 7.41s and
`lightgbm` at 3.37s — roughly 7x the
[2026-07-30 manifest](BENCHMARK_MANIFEST_2026-07-30.json) (0.905s / 0.481s) on
the identical dataset, config and leaf counts. LightGBM is untouched
third-party code, so the inflation could not be attributable to ClearGBM.

Cause: **Windows applies EcoQoS power throttling to the process a few seconds
in.** The identical LightGBM fit, repeated in one process:

```
fit 0  0.547s     fit 3  3.794s
fit 1  0.536s     fit 4  6.496s
fit 2  0.540s     fit 5  7.108s     <- 13x, and it does not recover
-- opted out of power throttling --
fit 6  0.540s     fit 8  0.491s
fit 7  0.521s     fit 9  0.503s
```

RSS (233 MB) and thread count (75) were flat throughout, and 90 seconds of
idle did not restore speed, ruling out both a leak and thermal recovery.

This matters beyond one slow run: the demotion is a **one-way step change
part-way through a run**, so the rotation protocol cannot cancel it. Rotation
neutralises a symmetric order effect; here the arms measured before the step
keep the fast regime and the rest never see it again, and each arm's median
straddles the step differently. The first two attempts disagreed with each
other by 9% on the leaf-wise arm for exactly this reason.

`covenant_ml.benchmarking.power` now opts the process out before any fit is
timed, and a refusal aborts the run rather than producing a number nobody can
attribute. The effect on measurement quality is visible in the variance:
per-arm standard deviation fell from ±0.72s / ±0.54s / ±0.18s to
±0.019s / ±0.051s / ±0.014s.

**Any fit-time figure produced by this harness before 2026-08-19 should be
treated as suspect** unless it was measured in a foreground interactive shell.
The 2026-07-30 numbers look unaffected — they match what this machine produces
once throttling is disabled — but that is an inference from agreement, not a
property anyone verified at the time.

## Not covered here

No XGBoost arm. The three-way figures quoted in the 2026-08-17 experiment came
from `covenant_ml.growth_policy`, a separate harness that reaches XGBoost as a
measurement instrument; XGBoost is not yet a `BenchmarkModelName`, so it
cannot appear in a manifest from this runner.
