# P5 Landings A + B — continued training and GOSS (2026-08-23)

## Landing A — continued training (no serde change)

The single-score boosting rounds moved to `training/single_score_rounds.rs`,
shared verbatim by the fresh trainer and the new continuation entry.
`continue_gradient_boosting` trains additional rounds under the model's OWN
embedded config and APPENDS the new trees — one self-contained artifact
whose config states the combined budget — deliberately inverting LightGBM's
`init_model` shape, where the returned booster is a delta model excluding
its own baseline (both behaviors pinned in the tech-wiki from
`engine.py`/`basic.py` @ 3ec5b99b).

- **Split training is exact**: 3 rounds + a 3-round continuation on the
  same data reproduces a fresh 6-round run bit for bit, for binary and
  regression, held by tests at the Rust core and the Python surface.
- Scope stated: multiclass and ranking continuation refused by name; bin
  edges recomputed from the continuation data.
- Identity through the loop extraction: **112/112** byte-for-byte
  (`BENCHMARK_MANIFEST_2026-08-23_p5_continuation_identity.json`).

## Landing B — GOSS (config fields 22-23, round-5 artifact retrain)

`goss_top_rate` + `goss_other_rate` land as required-with-null paired
fields: both-or-neither, each in (0, 1) exclusive, summing to at most 1,
excluding `subsample < 1` (GOSS replaces row subsampling), single-score
objectives only (multiclass/ranking refuse by name). The sampler ships
LightGBM's semantics (`goss.hpp` @ 3ec5b99b): rank by |gradient x
hessian| (the shipped divergence from the paper's |g|), skip sampling
while `round < 1/learning_rate`, keep the top outright, stream-sample the
rest with the adaptive `rest_need/rest_all` draw, and multiply
`(cnt - top_k)/other_k` into gradient AND hessian. One stated divergence
of our own: `other_k` floors at 1 where LightGBM's expression can divide
by zero. Deterministic per config; GOSS off is bit-identical history.

### Round-5 artifact retrain (config fields 22-23)

| artifact | expected | reproduced |
|---|---|---|
| rw_matches `active_cgbm.json` | val 0.7790 / test 0.7142 / best 16 / spw 1.655 | identical |
| taiwan `taiwan_cleargbm_model.json` | val 0.9451 / 98 trees | identical |
| us `us_cleargbm_model.json` | val 0.7848 / 14 trees | identical |

### Identity gate

Four arms x seeds 42-45 reproduce the knob-identity manifest **112/112**
byte-for-byte with GOSS off
(`BENCHMARK_MANIFEST_2026-08-23_p5_goss_identity.json`).

### GOSS quality vs LightGBM's GOSS

New harness: `covenant_ml.benchmarking.goss_quality` +
`scripts/benchmark_cleargbm_goss.py`. Deterministic noisy-logistic binary
corpus (20000 rows x 8 features per seed, irreducible noise), held-out
quarter, four arms per seed (each library full and GOSS at top 0.2 /
other 0.1), 200 rounds depth 4.

| seed | cleargbm full AUC | cleargbm GOSS AUC | lightgbm full AUC | lightgbm GOSS AUC |
|---|---|---|---|---|
| 42 | 0.795201 | 0.788343 | 0.796829 | 0.787572 |
| 43 | 0.787940 | 0.781519 | 0.787530 | 0.781295 |
| 44 | 0.795337 | 0.788131 | 0.795461 | 0.789797 |
| 45 | 0.799012 | 0.790690 | 0.799182 | 0.790963 |

The number that matters is the within-library sampling cost: ClearGBM's
mean AUC gap is **-0.0072** and LightGBM's is **-0.0073** — the sampler
pays the same quality price as the production implementation, for the
same ~70% row reduction after warmup.

Manifest: `BENCHMARK_MANIFEST_2026-08-23_p5_goss_quality.json`.

## Gates at landing

- cleargbm_rs: 1610 tests, 100.00% segment coverage, clippy clean.
- cleargbm: 259 tests, 100.00% (config fields + decode validation).
- covenant_ml: 2481 tests, 100.00% (GOSS quality harness + field sweep).
- covenant-radar-api: 2588 tests, 100.00%, zero source changes.
