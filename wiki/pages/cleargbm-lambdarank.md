---
title: ClearGBM LambdaMART — pair lambdas by query, NDCG as the gate
tags: [ml, cleargbm, ranking, lambdarank, ndcg, roadmap-p4]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-multiclass-softmax]]"
  - "[[cleargbm-sample-weights]]"
source_paths:
  - libs/cleargbm_rs/src/losses/lambdarank.rs
  - libs/cleargbm_rs/src/training/train_ranking.rs
  - libs/cleargbm/src/cleargbm/ensemble_ranking.py
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p4_lambdarank.md
fact_checked: "2026-08-23"
confidence: high
hubs: [libs]
---

# ClearGBM LambdaMART — pair lambdas by query, NDCG as the gate

P4 Landing B of the [[cleargbm-program-charter]] (board task `1fedf1e3`),
closing the phase. The `lambdarank` objective implements Burges 2010's
LambdaMART as LightGBM's `LambdarankNDCG` spells it (both primary sources
pinned in the tech-wiki at commit 3ec5b99b): per query, documents sort by
running score, a truncation-bounded pair scan accumulates
`lambda = -|dNDCG| / (1 + e^(delta score))` per misordered pair into the
per-row gradients and the `p(1-p)`-scaled pair weight into the hessians,
and the unchanged tree builder consumes both. Quality: NDCG@10 parity
with LightGBM's LGBMRanker (2-2 by seed, gaps < 0.002); identity: the
single-score path reproduced knob-identity 112/112 through the refactor.

## Groups are data; the truncation level is config

Query group sizes travel WITH the rows through a dedicated entry
(`train_gradient_boosting_ranking`; the generic entry rejects the
objective naming it) — the same data-not-config rule P2 set for weights.
The one new config field is `lambdarank_truncation_level` (Rust serde
field 21, required-with-null): `Some(k >= 1)` iff the objective is
`lambdarank`, because it changes the fitted model (it bounds the pair
loop and the max-DCG normalizer). `scale_pos_weight` and `n_classes` are
null under ranking. Validation is all-three-or-none (features, labels,
val groups); there is NO validation-weight argument — NDCG is per-query,
and a per-document eval weight has no defined meaning for it.

## Divergences from LightGBM, each deliberate

- **Exact sigmoid** instead of the million-entry lookup table (a speed
  cache with quantization error, not semantics).
- **Sigma fixed at 1.0** (LightGBM's default), no knob — a later knob is
  a stated addition, never a hidden default.
- **Lambda normalization always on** (LightGBM's default behavior): the
  `0.01 + |delta score|` division when scores are non-degenerate, then
  the `log2(1 + sum)/sum` row rescale.

Parity elsewhere: `2^label - 1` gains, labels capped at 31, queries
capped at 10000 documents, `1/log2(rank+2)` discounts, counting-sort max
DCG (`O(n + k)`), equal-label pairs skipped, weights multiplying lambda
AND hessian after the scan, stable sorts. All-zero-label queries
contribute zero lambdas and score NDCG 1.0.

## Model and prediction

Base score 0.0 (scores are relative within a query — no global offset),
stored in the existing scalar arm: NO model serde break this landing.
`predict_raw` IS ranking inference — the raw score is the key, documents
sort by it descending; `predict_proba` refuses ranking models. Early
stopping minimizes `1 - mean NDCG@truncation` over validation queries.

## Where things live

Rust: `losses/lambdarank.rs` (tables, validation, max DCG, the pair
scan, NDCG), `training/train_ranking.rs` (the loop; groups + labels +
weights travel as `RankingTrainingData`). Python:
`cleargbm.ensemble_ranking` (the training entry). The quality harness is
`covenant_ml.benchmarking.ranking_quality` +
`scripts/benchmark_cleargbm_ranking.py`, scored by the new
`compute_ndcg_at_k` metric. rw_matches carries no graded relevance or
query structure today, so the measured corpus is synthetic (seeded,
deterministic); wiring a real in-house ranking corpus is future work
under P6's experiment farm.
