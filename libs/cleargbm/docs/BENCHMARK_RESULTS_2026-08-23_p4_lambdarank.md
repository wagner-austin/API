# P4 Landing B — LambdaMART ranking (2026-08-23)

The `lambdarank` objective landed end to end, closing P4: per-query pair
lambdas as the gradient source (Burges 2010 as implemented by LightGBM's
`LambdarankNDCG`, both pinned in the tech-wiki at commit 3ec5b99b), query
groups as DATA (a new training entry takes them beside the rows; the
generic entry rejects the objective naming the right one), NDCG as the
evaluation metric (early stopping minimizes `1 - mean NDCG@k`), and the
raw score as the ranking key — prediction reuses the existing single-score
`predict_raw`, and `predict_proba` refuses ranking models.

## Deliberate, documented divergences from LightGBM

- **Exact sigmoid, no lookup table**: LightGBM caches
  `1/(1+e^(sigma x))` in a million-entry table; the table is a speed cache
  with quantization error, not semantics, so cleargbm evaluates it
  exactly.
- **Sigma fixed at 1.0** (LightGBM's default) with no config knob —
  adding one later is a stated knob, never a hidden default.
- **Lambda normalization always on** (LightGBM's `lambdarank_norm`
  default): the `0.01 + |delta score|` division under non-degenerate
  scores and the `log2(1 + sum)/sum` row rescale are this objective's one
  stated behavior.
- Everything else is parity: `2^label - 1` gains capped at label 31,
  `1/log2(rank+2)` discounts, counting-sort max DCG, the
  truncation-bounded pair loop skipping equal labels, per-row weights
  multiplying lambda AND hessian after the query scan, and stable
  score-descending sorts.

## Serde break (round-4 artifact retrain)

Config field 21, `lambdarank_truncation_level`: required-with-null —
`Some(k >= 1)` iff the objective is `lambdarank` (it bounds the pair loop
and the max-DCG normalizer), null otherwise; `scale_pos_weight` and
`n_classes` must be null under `lambdarank`. Stored artifacts predating
the field refuse to load, so all three service artifacts were retrained
and reproduced their recorded numbers exactly:

| artifact | expected | reproduced |
|---|---|---|
| rw_matches `active_cgbm.json` | val 0.7790 / test 0.7142 / best 16 / spw 1.655 | identical |
| taiwan `taiwan_cleargbm_model.json` | val 0.9451 / 98 trees / spw 29.994 | identical |
| us `us_cleargbm_model.json` | val 0.7848 / 14 trees / spw 14.077 | identical |

## Identity gate — the single-score path is bit-unchanged

The four-arm benchmark (cleargbm, cleargbm@leaf_wise, lightgbm, xgboost x
seeds 42-45) reproduces the 2026-08-22 knob-identity manifest **112/112
non-timing values byte-for-byte** through the fourth-objective refactor.

Manifest: `BENCHMARK_MANIFEST_2026-08-23_p4_ranking_identity.json`.

## Ranking quality vs LightGBM's ranker

New harness: `covenant_ml.benchmarking.ranking_quality` +
`scripts/benchmark_cleargbm_ranking.py`. Deterministic synthetic corpus
(400 queries x 20 documents x 8 features per seed; grades 0-3 by
within-query utility quartile over a noisy linear signal), the final
quarter of queries held out, matched hyperparameters (100 rounds, depth
4, lr 0.1, 64 bins, min 20 rows/leaf, truncation 10, no subsampling,
single thread), seeds 42-45.

| seed | cleargbm NDCG@10 | lightgbm NDCG@10 |
|---|---|---|
| 42 | 0.948514 | **0.950267** |
| 43 | **0.952755** | 0.951968 |
| 44 | 0.947605 | **0.947741** |
| 45 | **0.949923** | 0.949570 |

A 2-2 split by seed, every gap under 0.002 NDCG: the lambda formulation
is quality-competitive with LightGBM's production ranker, not merely
wired.

Manifest: `BENCHMARK_MANIFEST_2026-08-23_p4_ranking_quality.json`.

## Gates at landing

- cleargbm_rs: 1584 tests, 100.00% segment coverage, clippy `-D warnings`
  clean, all files <= 600 lines. (The guard's Rust test rules also ran
  this landing and 37 accumulated test-signature violations from earlier
  landings were closed: every `#[test]` now returns `Result`.)
- cleargbm: 251 tests, 100.00% coverage (`ensemble_ranking.py` new;
  `_types_model.py`'s categorical decode tests split out at the 600-line
  ceiling).
- covenant_ml: 2471 tests, 100.00% coverage (`compute_ndcg_at_k` metric,
  the ranking quality harness, config sweep).
- covenant-radar-api: 2588 tests, 100.00% coverage, zero source changes.
