# Benchmark 2026-08-21 — first four-arm manifest: XGBoost joins the baseline set

First measurement with the `xgboost` arm under the harness protocol. This
closes the gap the
[2026-08-19 run](BENCHMARK_RESULTS_2026-08-19_growth_variants.md) recorded
under "Not covered here": XGBoost had only ever been measured by the
scratchpad instrument harness behind
[`EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md`](EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md),
which produced no manifest. It is now a `BenchmarkModelName`, measured under
the same per-seed slot rotation and EcoQoS power-throttle opt-out as every
other arm, in one manifest:
[`BENCHMARK_MANIFEST_2026-08-21_four_arm.json`](BENCHMARK_MANIFEST_2026-08-21_four_arm.json)
(schema 2).

Reproduce with:

```
cd libs/covenant_ml
poetry run python scripts/benchmark_cleargbm_vs_lightgbm.py --variants \
    --out ../cleargbm/docs/BENCHMARK_MANIFEST_2026-08-21_four_arm.json
```

## Setup

Identical to 2026-08-19: American bankruptcy, 78,682 rows x 18 features,
sha256 `cff2c899a97ecd62…`. 200 trees, `max_depth` 6, 64 bins, `num_leaves`
31, `n_jobs` 1. Seeds 42/43/44, 2 warm-ups + 5 timed fits per arm per seed,
median per seed. Company-disjoint splits; four arms rotate one slot per seed.
(With four arms and three seeds the rotation no longer visits every slot per
arm — a full-coverage run needs four seeds. The residual order effect is
bounded by the warm-ups; noted rather than hidden.)

## Results

| arm | fit (mean of per-seed medians) | leaves/tree | AUC-ROC | AUC-PR |
|---|---|---|---|---|
| `cleargbm` (depth-wise) | 0.8888s ± 0.0036 | 47.06 | 0.6824 | 0.1416 |
| `cleargbm@leaf_wise` | 0.8375s ± 0.0103 | 30.98 | 0.6874 | 0.1383 |
| `lightgbm` | 0.4928s ± 0.0045 | 30.96 | 0.6881 | 0.1366 |
| `xgboost` | 0.4913s ± 0.0177 | 22.08 | 0.6863 | 0.1367 |

Against LightGBM: raw 1.804x, leaf 1.520x, per-leaf 1.187x.

## What this run establishes

**Determinism across runs, demonstrated.** Every quality metric and leaf
count reproduces the 2026-08-19 manifest bit for bit — AUC-ROC
0.6824 / 0.6874 / 0.6881, leaves 47.06 / 30.98 / 30.96 — two days apart, on
the same seeds and splits. This is the property the variant-trialing state
exists to provide: quality differences between arms are attributable to the
arms, never to the run.

**XGBoost lands on LightGBM's wall clock by a different route.** 0.4913s vs
0.4928s — indistinguishable — but XGBoost builds 22.1 leaves to LightGBM's
31.0, i.e. it is ~40% slower per leaf and simply builds fewer. Quality is a
four-way statistical tie (AUC-ROC spread 0.0057, the same spread the
2026-08-19 three-arm run showed). This reproduces the 2026-08-17 instrument
experiment's shape — depth-wise XGBoost reaching the quality ceiling at ~22
leaves — now under the manifest protocol instead of a scratchpad script.

**Fit times are not comparable across dated runs; ratios mostly are.** Every
arm ran faster than on 2026-08-19 (cleargbm 0.889s vs 1.013s, lightgbm 0.493s
vs 0.515s) — different day, different machine state, same relative story. The
per-leaf ratio moved from 1.295x to 1.187x between the two runs; treat the
per-leaf gap as "roughly 1.2–1.3x", not as either number. Cross-run
comparisons should use within-run ratios only.

**The leaf-wise verdict from 2026-08-19 stands.** Leaf-wise reaches tied
quality at a third fewer leaves and is not slower; the ~6% wall-clock edge
over depth-wise persists (0.8375s vs 0.8888s) but per-leaf it remains the
more expensive builder. Nothing here changes the honest reading.
