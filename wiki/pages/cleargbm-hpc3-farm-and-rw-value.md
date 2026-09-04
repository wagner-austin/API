---
title: ClearGBM on HPC3 — the farm runs; four new-domain corpora join the board
tags: [ml, cleargbm, hpc3, slurm, rw-value, metabolomics, voc, roadmap-p6]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-quantized-training]]"
source_paths:
  - tools/hpc3/runs/hpc3.json
  - tools/hpc3/runs/sweep-cleargbm-p6-rung1.json
  - libs/covenant_ml/scripts/derive_rw_value.py
  - libs/covenant_ml/scripts/build_metab_corpus.py
  - libs/covenant_ml/scripts/build_voc_corpus.py
  - libs/covenant_ml/src/covenant_ml/datasets/xlsx_reader.py
  - libs/covenant_ml/src/covenant_ml/benchmarking/regression_quality.py
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-24_p6_farm_and_rw_value.md
source_git_blobs:
  "tools/hpc3/runs/hpc3.json": b9c7f77d605417ca0b1afb0a99c75d70f35e3d1c
  "tools/hpc3/runs/sweep-cleargbm-p6-rung1.json": 3a10d80f404ce7dce29abb754f309178052ad8ff
  "libs/covenant_ml/scripts/derive_rw_value.py": ea009dd9bbe34ff6786df7b9af24e9e4a51469e1
  "libs/covenant_ml/scripts/build_metab_corpus.py": 3b7eb28d5db3e72dded3d3092fc96e590416e2f7
  "libs/covenant_ml/scripts/build_voc_corpus.py": fbc511ae75148199c2ed706d154a2d67bb4ee8a3
  "libs/covenant_ml/src/covenant_ml/datasets/xlsx_reader.py": f490a898b0b7b7b37312aab6e90d82febff69408
  "libs/covenant_ml/src/covenant_ml/benchmarking/regression_quality.py": c7eead0886cb87eb9d5a7de355add58db819aa3a
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-24_p6_farm_and_rw_value.md": a8aaf35a888cfda68e7331b0132fe09a19eb06e4
fact_checked: "2026-08-24"
confidence: high
hubs: [libs]
---

# ClearGBM on HPC3 — the farm runs; five new-domain corpora join the board

P6 Landings A and B1-B5 of the [[cleargbm-program-charter]] (board task
`1ad15eb6`): the experiment farm plus rw_value, weather_tmax,
metab_confidence, voc_match_quality and metab_blank.

## The farm (Landing A)

ClearGBM experiments run on UCI HPC3's `free` CPU partition through
`tools/hpc3`. Onboarding was, as the HPC3 survey session promised, one
project block in the workspace (`tools/hpc3/runs/hpc3.json`): 4 cpus,
16 GB, 60 minutes, `gpu: null`, `deterministic: true`, with
numpy/lightgbm/xgboost pinned to the exact local benchmark versions and
enforced by preflight against the cluster env
(`/pub/wagnera3/envs/cleargbm`, conda-forge py311 — and since 2026-08-25
the Anaconda default-channel ToS gate in conda 25.11.1 is cleared too:
the operator directed acceptance for `pkgs/main`/`pkgs/r` under
`wagnera3`, so either channel family works; `cleargbm_rs` wheel
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

## weather_tmax (Landing B2)

The weather corpus exists, built THROUGH the deployed feature path:
`fetch_ghcnd_weather.py` vendors 24 GHCN-Daily stations under a
mechanical inventory rule (sha256-pinned manifest), and radar's
`scripts/build_weather_corpus.py` fits per-station Fourier/threshold
state on 1950-1989 ONLY, then emits 1990-2024 summer rows whose features
come from the radar `WeatherFeatureExtractor` itself (plus day-of-year
and three lagged anomalies); target = next day's anomaly. A data-driven
completeness gate refused 9 of 24 stations by name (the inventory's
year-span says nothing about continuity). Result: 43,132 rows across 15
stations, target mean +0.091 C — the warming signal where physics puts
it. The scoreboard entry is honest and adverse: **LightGBM leads**
(mean test RMSE 2.0695 vs xgboost 2.0704, cleargbm@leaf_wise 2.0735,
cleargbm 2.0751; grouped by station, seeds 42-46, R² ~0.44-0.50).
ClearGBM's first losing corpus is a named target now, and it is what
makes the wins elsewhere credible.

## metab_confidence (Landing B3)

The metabolomics corpus, from the operator's artcal campaign (corvis
dataset 15's provenance chain, mzMine -> DIA MGF -> SIRIUS 6.3.12) —
NOT the Emily/ProGenesis table, which provenance says never to join.
The SIRIUS merged summaries are the target source; ZODIAC was
deliberately not run in the campaign (empty `ZodiacScore` column), so
no ZODIAC data exists to include. One row per rank-1 CSI:FingerID
structure call; target = COSMIC `ConfidenceScoreExact`; predictors are
PRE-ANNOTATION measurables only (precursor m/z, RT, MS1 height and
correlated-peak count, MS2 spectral shape, detection statistics over
the 23 biological samples — the pooled `combine.mzML` column is
excluded, and annotation outputs never become features). 795
`-Infinity`-confidence rows dropped by counted rule; result **17,611
rows across 138 rt bins** (0.1-minute co-elution windows are the group
column — adducts/fragments of one molecule co-elute). Built
byte-deterministically by `covenant_ml/scripts/build_metab_corpus.py`
against three sha256-pinned sources (the MGF pin matches the corvis
provenance pin exactly); `data.csv` is committed.

The scoreboard entry: four arms are effectively TIED — mean test RMSE
xgboost 0.129348, lightgbm 0.129391, cleargbm@leaf_wise 0.129428,
cleargbm 0.129497 (spread 0.12%, far under the seed-to-seed spread).
The real finding is scientific: R² is 0.004-0.025 for every arm —
structure confidence is barely predictable from how well a feature was
measured; it is dominated by what the compound is. A weak-signal
benchmark entry, recorded exactly as measured.

## voc_match_quality (Landing B4)

The BVOC corpus, from the Faiola Lab's aggregated GC-MS peak table
(ten California reserve site sheets; tree-bot's sha256-pinned lab
snapshot, and the builder's manifest pin reproduces the snapshot's own
pin). The GC-MS twin of metab_confidence's question: one row per
chromatogram peak, target = the top NIST library match's quality
(`Match1.Quality`, 1-99, verbatim), predictors PRE-ANNOTATION
measurables only — species (41 codes including `blk` blanks), RT, and
chromatogram-context statistics (run peak count, RT rank fraction, gap
to previous peak, ±0.1/±1-min crowding, run RT span; a no-hit peak
still crowds its neighbours). `MatchScore` was verified equal to the
target on all 6,238 source rows and is excluded with the rest of the
library/curation outputs; `site` is the group column. Drops counted:
350 misfiled no-species rows (provenance-flagged), 35 no-quality, 1
no-RT, 2 impossible qualities (944/994). Result: **5,850 rows across
10 sites** (228 chromatograms), built byte-deterministically through a
new stdlib XLSX reader (`covenant_ml.datasets.xlsx_reader` — no new
dependency).

The scoreboard entry has REAL signal (R² 0.37-0.48, against
metab_confidence's 0.004-0.025) and is honestly adverse: **LightGBM
leads** (mean test RMSE 18.3170 vs cleargbm 18.4423, cleargbm@leaf_wise
18.4806, xgboost 18.5646; per-seed wins lightgbm 3 / cleargbm 1 /
xgboost 1). ClearGBM is second at a 0.7% gap — the second corpus on
the named-target list beside weather_tmax.

## metab_blank (Landing B5)

The Emily/ProGenesis table as a blank-vs-real peak classifier — the
dashboard's own leading design, landed 2026-08-25. The lab's standard
3x blank-filter rule IS the label (all individual blanks together,
pooled combine excluded, sidestepping the documented leaf-vs-root
assignment dispute); features are physicochemical measurables ONLY
(m/z, charge, RT, peak width, m/z + Kendrick/CH2 mass defects — the
contaminant-homolog signatures), never anything intensity-derived,
because the intensities define the label. 19,064 rows across 127
retention windows, positive ratio 0.641, byte-deterministic. The
corpus forced one instrument extension: mixed-label co-elution groups
make stratified group labels undefined, so covenant_ml gained plain
`group_kfold_split` and cv_external selects the instrument by a
stated data property (uniform → stratified exactly as before, anchor
reproduced; mixed → plain grouped k-fold, announced). Result: a
statistical three-way tie (lightgbm 0.8710, cleargbm 0.8691,
leaf-wise 0.8690, spread inside the fold spread) and a scientific
finding — contamination is physicochemically distinguishable at AUC
~0.87 with no intensity information, so a blank-filter prior exists
even where blanks are missing or disputed.

## Rung 5 — tuned vs tuned, the capstone

Each dataset's best preset, both engines tuned over their OWN Optuna
spaces at matched budgets (100 trials; 40 on us-full): ClearGBM takes
taiwan 0.9640 vs 0.9581 and kaggle 0.8708 vs 0.8699; LightGBM takes
polish by 0.0012 and us by four points — measured to be a
tuning-surface asymmetry (ClearGBM's space pins
max_features/colsample/reg_lambda, knobs the engine has), the named
successor: search-space parity. Three of four ClearGBM winners tuned a
coarseness floor; taiwan's winner chose none — a dial, not a default.
P6 closed with this rung
(`BENCHMARK_RESULTS_2026-08-25_p6_rung5_tuned_vs_tuned.md`).
