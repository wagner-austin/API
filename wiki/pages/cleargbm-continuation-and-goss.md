---
title: ClearGBM continued training and GOSS — append trees, sample honestly
tags: [ml, cleargbm, goss, continued-training, roadmap-p5]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-sample-weights]]"
  - "[[cleargbm-lambdarank]]"
source_paths:
  - libs/cleargbm_rs/src/training/continue_training.rs
  - libs/cleargbm_rs/src/training/goss.rs
  - libs/cleargbm_rs/src/training/single_score_rounds.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p5_continuation_goss.md
source_git_blobs:
  "libs/cleargbm_rs/src/training/continue_training.rs": 4be58ac4fa08bac251058925688003eac0a984ec
  "libs/cleargbm_rs/src/training/goss.rs": f2ad49e26863678fe99e7e689ba9e0b356fb24ff
  "libs/cleargbm_rs/src/training/single_score_rounds.rs": 7fb0f35e74dd69934978bdd4ea51f640ba47bc50
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p5_continuation_goss.md": 747f42be1be3da957738a6e9a90edb4178ea634c
fact_checked: "2026-08-23"
confidence: high
hubs: [libs]
---

# ClearGBM continued training and GOSS — append trees, sample honestly

P5 Landings A and B of the [[cleargbm-program-charter]] (board task
`0c320137`). Both ride the single-score boosting loop, now extracted to
`single_score_rounds.rs` and shared verbatim between the fresh trainer,
continuation, and the GOSS branch — identity 112/112 reproduced after
each landing.

## Continued training inverts LightGBM's delta-model shape

LightGBM's `init_model` bakes the old model's raw predictions into the
dataset as an init score and returns a booster holding ONLY the new
trees — an artifact that excludes its own baseline (pinned in the
tech-wiki from `engine.py`/`basic.py` @ 3ec5b99b).
`continue_gradient_boosting` starts from the same math (scores
initialized from the existing model's predictions) but APPENDS the new
trees, returning one self-contained model whose embedded config states
the combined budget (`n_estimators = existing trees + additional`).

Split training is EXACT: n rounds plus a k-round continuation on the
same data reproduces a fresh (n+k)-round run bit for bit under
deterministic knobs — the continuation's starting scores are precisely
the running scores the fresh run had at round n. Held by tests for
binary and regression at both language layers. Continuation trains under
the model's OWN config (the caller states only data + additional
rounds); per-tree feature-mask seeds continue from the existing tree
count; bin edges are recomputed from the continuation data (the model
stores none — stated, not hidden). Multiclass/ranking continuation:
refused by name, future scope. No serde change.

## GOSS ships the shipped semantics, as one honest knob pair

`goss_top_rate`/`goss_other_rate` (config serde fields 22-23,
required-with-null): both-or-neither, each in (0,1), sum <= 1, exclusive
with `subsample < 1` (GOSS replaces row subsampling), single-score
objectives only. The pass is LightGBM's `goss.hpp` @ 3ec5b99b: rank rows
by |gradient x hessian| (the shipped code's divergence from the paper's
|g|), skip sampling during the first `1/learning_rate` rounds (bit-
identical to GOSS off — tested), keep the top-k outright via a partial
selection threshold, stream-sample the rest at `rest_need/rest_all`, and
multiply `(cnt - top_k)/other_k` into gradient AND hessian of sampled
rows. One stated divergence: `other_k` floors at 1 where LightGBM's
expression can divide by zero. RNG: the run's row-sampling stream (GOSS
is its only consumer when active), so runs are deterministic per config.

Quality gate: on the 20000-row noisy-logistic corpus, ClearGBM's GOSS
costs a mean AUC gap of -0.0072 vs its own full training; LightGBM's
GOSS costs -0.0073 vs its own — the same price for the same ~70% row
reduction (`BENCHMARK_MANIFEST_2026-08-23_p5_goss_quality.json`).

## Where things live

Rust: `training/continue_training.rs`, `training/goss.rs`,
`training/single_score_rounds.rs` (the shared loop). Python:
`cleargbm.ensemble_continued` (binary + regression continuation);
GOSS is config-only (the existing train entries carry it). Harnesses:
`covenant_ml.benchmarking.goss_quality` + `scripts/benchmark_cleargbm_goss.py`.
Artifacts retrained round 5 (fields 22-23), all numbers exact.
Remaining P5 scope: quantized training (the crown) — and EFB, which this
phase recommends EXCLUDING: its habitat is sparse one-hot data the
dataset registry lacks, and its hardcoded 1/10000 conflict budget would
need a knob the constitution refuses to invent without a corpus that
names it.
