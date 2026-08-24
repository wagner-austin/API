---
title: ClearGBM on HPC3 — the experiment farm runs, and rw_value joins the board
tags: [ml, cleargbm, hpc3, slurm, rw-value, roadmap-p6]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-quantized-training]]"
source_paths:
  - tools/hpc3/runs/hpc3.json
  - tools/hpc3/runs/sweep-cleargbm-p6-rung1.json
  - libs/covenant_ml/scripts/derive_rw_value.py
  - libs/covenant_ml/src/covenant_ml/benchmarking/regression_quality.py
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-24_p6_farm_and_rw_value.md
fact_checked: "2026-08-24"
confidence: high
hubs: [libs]
---

# ClearGBM on HPC3 — the experiment farm runs, and rw_value joins the board

P6 Landings A and B1 of the [[cleargbm-program-charter]] (board task
`1ad15eb6`).

## The farm (Landing A)

ClearGBM experiments run on UCI HPC3's `free` CPU partition through
`tools/hpc3`. Onboarding was, as the HPC3 survey session promised, one
project block in the workspace (`tools/hpc3/runs/hpc3.json`): 4 cpus,
16 GB, 60 minutes, `gpu: null`, `deterministic: true`, with
numpy/lightgbm/xgboost pinned to the exact local benchmark versions and
enforced by preflight against the cluster env
(`/pub/wagnera3/envs/cleargbm`, conda-forge py311, `cleargbm_rs` wheel
built on-cluster under rustup stable).

First rung: 16 members (4 datasets x 4 feature presets), 20 Optuna TPE
trials each via radar's `scripts.optimize` — 320 trials, 0 pruned, 0
failed, `hpc3-triage` 0 findings. Findings worth keeping: the winning
preset tracks dataset WIDTH (full-engineering wins on 18-feature us,
drowns on 95-feature taiwan at this budget), and `log_only` ties `none`
to four decimals on three datasets because histogram split finding is
invariant to per-feature monotone transforms — a log-duplicate can
never change a tree. Per-member numbers and per-job logs:
`BENCHMARK_RESULTS_2026-08-24_p6_farm_and_rw_value.md` +
`tools/hpc3/runs/results/p6-rung1/`.

Fanning a single-machine CLI across 16 nodes surfaced two shared-writer
hazards, both fixed at the root rather than accepted: the shared
optimization-history JSONL (BeeGFS gives no cross-node append
atomicity) and the per-dataset optimal-config files (last-writer-wins
across preset members). Both writers now embed `HPC3_JOB_NAME` — which
the hpc3 batch script always exports — so farm members never share a
file; local behavior is unchanged. A third fix on the way: three radar
streaming tests asserted an exact count of asynchronous rdkafka
connection events (`poll(0.0) == 0`) and began flaking once a live
broker answered localhost:9092; they now pin the deterministic contract
(`flush(0.0) == 0`, the message-queue length) and survived three
consecutive full gates with the broker up.

## rw_value (Landing B1)

The standing benchmark's second regression corpus: 569,561 rows across
99 matches, derived deterministically from rw_matches
(`scripts/derive_rw_value.py` in covenant_ml). Target
`frames_remaining` — time to verdict from mid-match state; `won` and
`verdict` never reach the file, `match` is the group column, and the
new registry-driven regression harness splits by WHOLE MATCH when a
config declares grouping (the regression types/loader/registry gained
group support this landing). Four arms under the P1 protocol, seeds
42-46: ClearGBM leads mean test RMSE (175,525 vs LightGBM 176,091 vs
XGBoost 177,091) on a genuinely hard corpus (R² ~0.2; seed 44's fold is
negative for every arm — grouped splits are unforgiving, which is the
point).

## Remaining P6 scope

Weather needs a GHCN-D fetch and McKinnon-style construction (the radar
weather domain is inference-side only). Metabolomics/BVOC data is real
and located (Emily 23,134 x 58; ten VOC field sites in corvis
`research_*`) but joins the registry only behind an honest target
design with the operator's science — blank-vs-real peak classification
leads for Emily. Both are later landings under the open board task.
