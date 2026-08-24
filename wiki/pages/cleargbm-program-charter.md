---
title: ClearGBM program charter — AI-first tabular learning, built to outpredict
tags: [ml, cleargbm, roadmap, charter, ai-first]
related:
  - "[[cleargbm-decorative-knob-class]]"
  - "[[cleargbm-leaf-normalized-benchmarking]]"
  - "[[cleargbm-perf-leaf-wise-growth]]"
source_paths:
  - libs/cleargbm_rs/src/training/train.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-22_knob_closure.md
fact_checked: "2026-08-22"
confidence: high
hubs: [libs]
---

# ClearGBM program charter — AI-first tabular learning, built to outpredict

Set by the operator 2026-08-22: grow ClearGBM from a binary-classification
engine into the tabular learner of first resort — for covenant modeling,
weather prediction, RustedWarfare match/value models, metabolomics and BVOC
data, and anything else tabular — with the ambition of beating every other
model on the corpora we care about. Resources: this workspace, the tech-wiki
research pipeline, corvis, HPC3, and effectively unbounded AI time.

## The constitution (what is never traded)

1. **Config honesty.** Every field a config can state is honored by training
   or does not exist — defended by knob-sensitivity tests. This invariant
   was bought with five bugs' worth of pain ([[cleargbm-decorative-knob-class]])
   and is the reason our quality numbers can be believed.
2. **Determinism per config.** Same config + data = same model, bit for bit.
   New features may change results relative to *other* configs, never
   relative to themselves.
3. **Safe Rust, 100% coverage, everything documented.** `unsafe_code =
   "forbid"` stands; its measured price (~20% wall clock vs LightGBM, per
   the 2026-08-22 ledger) is accepted. SIMD, if ever, comes via safe crates
   with an explicit re-baseline.
4. **Measured, or it didn't happen.** Every feature lands with its gate
   declared first — bit-identity for pure mechanics, the quality gate for
   anything that changes results — benchmarked against the dataset registry,
   with the manifest committed.

Quality-affecting capabilities (quantized training, GOSS, sampling) are NOT
transparency trades: as explicit opt-in knobs they preserve all four rules.
The one rejected feature shape is arbitrary user-code objectives crossing
the FFI; the objective menu is built-in Rust, always inspectable.

## AI-first, concretely

The primary operator of this library is an AI session. Therefore:

- One JSON config in, one JSON artifact out; the artifact embeds the full
  config it trained under (already true) plus, as the program lands it,
  dataset fingerprint and training-report metrics — an artifact should
  answer every question about itself.
- Machine-readable everywhere: manifests, metrics, sweep results. Pretty
  tables are for humans and crash on cp1252; JSON is for operators.
- Deterministic reruns and honest configs make every experiment a fact a
  later session can build on without re-verification.
- Errors name the violated invariant and the fix, not just the symptom.
- The wiki and `libs/cleargbm/docs/` are the memory; the agent board is the
  coordination surface; every phase below is a board task.

## The roadmap (phases, dependency-ordered)

- **P1 — Objective abstraction + regression.** DONE 2026-08-22 — see
  [[cleargbm-objective-seam]]. The `Objective` enum (binary log loss,
  squared error) landed behind one seam; the binary path proved
  byte-identical (56/56 manifest values), and ClearGBM's first regression
  benchmark entry LEADS both opponents on financial_distress (RMSE 1.8023
  vs xgboost 1.8291, lightgbm 1.8637) at the fastest wall clock.
  Quantile/Huber remain future menu additions on the same seam.
- **P2 — Per-row sample weights.** DONE 2026-08-23 — see
  [[cleargbm-sample-weights]]. Both objectives take optional per-row
  weights (data, not config; no artifact break); `scale_pos_weight` is
  now provably the derived special case, bit for bit; four-arm identity
  reproduced 56/56. GOSS (P5) and ranking (P4) are unblocked.
- **P3 — Data realism.** DONE 2026-08-23, in two landings. Landing A
  (2026-08-22) — see [[cleargbm-nan-direction-and-colsample]]: the
  missing-value direction was found ALREADY learned per split (the
  spec's "fixed policy" premise was wrong; pinned by a
  stump-discriminator test), and `colsample_bytree` landed as a
  required-with-null (0,1)-exclusive per-round mask composed with
  `max_features`. Landing B (2026-08-23) — see
  [[cleargbm-categorical-splits]]: native many-vs-many categorical
  splits by gradient-sorted prefix scan, set-membership nodes, SHAP
  refusal. Both landings: identity 112/112, all five artifacts retrained
  with exactly reproduced numbers.
- **P4 — More objectives.** DONE 2026-08-23, in two landings. Landing A
  (multiclass softmax) — see [[cleargbm-multiclass-softmax]]: K trees per
  round on the P1 seam, class-major buffers, `n_classes`
  required-with-null (config field 20) + per-class base scores (model
  field 6), log loss below LightGBM on all four quality seeds. Landing B
  (LambdaMART ranking) — see [[cleargbm-lambdarank]]: pair lambdas by
  query on P1+P2 (groups are data; `lambdarank_truncation_level` is
  config field 21), exact sigmoid, NDCG-gated early stopping, NDCG@10
  parity with LGBMRanker (2-2 by seed, gaps < 0.002). Both landings:
  identity 112/112, all three artifacts retrained with exactly reproduced
  numbers. The lambda math is pinned in the tech-wiki (Burges 2010 +
  LightGBM rank/DCG pages @ 3ec5b99b).
- **P5 — Accelerators through the quality gate.** DONE 2026-08-23, in
  three landings. Landing A (continued training) — see
  [[cleargbm-continuation-and-goss]]: trees append to a self-contained
  artifact (inverting LightGBM's delta-model init_model shape), split
  training exact to the bit. Landing B (GOSS): the honest rate pair
  (config fields 22-23), shipped-code semantics, quality cost matching
  LightGBM's own (-0.0072 vs -0.0073 mean AUC gap). Landing C
  (quantized training) — see [[cleargbm-quantized-training]]:
  `quantized_gradient_bins` (config field 24), packed 16/32-bit integer
  histograms per Shi 2022's shipped form, split training exact under
  quantization via pure-function rounding randoms. The measurement
  refuted the marketing: at our single-threaded 20k-200k scales the
  knob IMPROVES quality (+0.0029 mean AUC, stochastic-rounding
  regularization) but costs ~1.1-1.5x wall clock — and LightGBM's own
  quantized arm only breaks even there too; the 2x lives at
  many-million-row, many-thread scale. EFB is EXCLUDED: its habitat is
  sparse one-hot data the registry lacks, and its hardcoded 1/10000
  conflict budget would need a knob no corpus yet names.
- **P6 — Scale.** IN PROGRESS 2026-08-24 — see
  [[cleargbm-hpc3-farm-and-rw-value]]. Landing A (the farm) DONE: ClearGBM
  runs on HPC3's free CPU partition through tools/hpc3; the first rung put
  320 Optuna trials (16 members, 4 datasets x 4 presets) through compute
  nodes with zero failures and zero triage findings, and the fan-out's two
  shared-writer hazards were fixed at the root (per-job filenames under
  HPC3_JOB_NAME). Landing B1 (rw_value) DONE: the time-to-verdict
  regression corpus (569,561 rows, 99 match groups) joined the registry
  with grouped splitting, and ClearGBM leads its mean test RMSE. Landing B2
  (weather_tmax) DONE: 43,132 GHCN-D summer rows across 15 stations,
  state fitted on 1950-1989 only, features through the deployed radar
  extractor, day-ahead anomaly target — and the entry is honestly
  adverse: LightGBM leads it (2.0695 vs ClearGBM 2.0735/2.0751 mean
  RMSE), ClearGBM's first losing corpus and now a named target.
  Landing B3 (metab_confidence) DONE: the artcal campaign's SIRIUS
  rank-1 structure calls as a corpus — 17,611 rows across 138
  co-elution windows, target COSMIC ConfidenceScoreExact, predictors
  pre-annotation measurables only; the four arms tie (spread 0.12% of
  RMSE) and the honest finding is scientific: R² 0.004-0.025 — a
  weak-signal entry where learner differences are noise. Landing B4
  (voc_match_quality) DONE: the BVOC corpus from the Faiola Lab's
  ten-site GC-MS peak table — 5,850 rows grouped by site, target NIST
  Match1.Quality, pre-annotation predictors (species, RT, chromatogram
  context) through a new stdlib XLSX reader; real signal (R²
  0.37-0.48) and honestly adverse: LightGBM leads, ClearGBM second at
  0.7%. Remaining: larger farm rungs, and closing the weather_tmax +
  voc_match_quality gaps. Distributed/GPU training only when
  single-node ceilings measurably bind.

Out of scope until a real need names them: sparse-matrix input, external
memory, DART, in-library CV (covenant_ml owns CV).

## How "better than any other model" gets decided

Not by assertion. The dataset registry (rw_matches, taiwan, us, polish,
the kaggle credit family, and the weather/metabolomics corpora as they are
onboarded) is the standing benchmark; LightGBM and XGBoost under identical
protocols are the standing opponents; the leaf-normalized and
weighted-symmetric disciplines from this week are the rules. Current
standing, 2026-08-22: quality lead on rw_matches (0.7492 vs 0.7299), tie
on the bankruptcy family, ~1.25x wall clock. The program wins when the
quality column leads everywhere it matters and nobody has to squint at an
asterisk.
