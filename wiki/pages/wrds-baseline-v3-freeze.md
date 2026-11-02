---
title: CRSP version-3 licensed study freeze
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]", "[[wrds-baseline-v2-freeze]]", "[[wrds-protocol-revision]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/freeze-report.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evaluation.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/freeze-report.json": "66ac30dcc341698ef31b3c144a633303fe10025d"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evaluation.json": "cc3375f54d1b5ced0efc206cc9e42f13d019fd49"
fact_checked: 2026-11-02
confidence: medium
---

# CRSP version-3 licensed study freeze

This is an explicit accounting/runtime amendment of the stopped version-2
study. The candidate matrix, estimator settings, seed aggregation, tie rules
and resource ceilings are unchanged. Fifteen requests were admitted before
the first queue submission; this freeze is not an evaluation result.[^freeze]

## Frozen design

Training covers 2000–2014, validation 2015–2021 and the sealed final period
2022–2024. The current amendment retains rejected-entry cash and actual
traded-notional costs, and adds `exact-missing-price-spans`. Universe filters,
calendar and terminal rules remain in [[crsp-monthly-study-design]].[^design]

| Control | Frozen value |
|---|---|
| Candidate features | momentum + liquidity; momentum only; liquidity only |
| Seeds per candidate | 11, 23, 37, 53, 71 |
| Estimator | 200 trees, depth 6, 31 leaves |
| Selection | LightGBM minus momentum at 10 bps; average seeds within month, then 82 required months |
| Tie | Fewer leaves, then lexicographic candidate digest |
| Final seed | 11 |
| Resources per request | 8 CPUs, 128 GiB, 7,200-second payload timeout; 121-minute allocation, no GPU |
| Uncertainty | 95% confidence; 12-month moving blocks, 10,000 draws, bootstrap seed 7 |

Selection and estimator settings are inherited unchanged from the prior
freeze. The exact request digests and candidate identities are in the
program-authored matrix. Policy digest is
`2a3ae28753525fc7bd8818add1d369d2ff224f71557cb30cc49dd26b91025c06`.[^freeze][^design]

## Qualification and execution boundary

Native job `58748788` used source `087f67b4c` and the frozen image to evaluate
the exact prior failed training span. It completed in 76 seconds; native
inspection bound all sixteen input digests, its result and shared RunRecord.
The check establishes the linked two-session return is applied once, within
`1e-12`, and normal-exit plaintext removal. It does not qualify every interval
or prove cleanup after a different allocation is killed.[^preparation]

The first request reached native job `58749373` through shared queue
`766bc4c8`; the other fourteen wait for its verified result. No candidate is selected and no final
ciphertext is staged or released. [[wrds-baseline-readiness]] carries current
status; this page records the fixed design.[^preparation]

[^freeze]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/freeze-report.json`, `old_policy`, `new_policy`, `policy_fields_changed`, `design_fields_changed`, `selection_unchanged`, `estimator_unchanged` and `matrix`; unchanged selection contract from [[wrds-baseline-v2-freeze]].
[^design]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evaluation.json`, `design`, `stopping_months`, `fold_months` and `confidence`.
[^preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `qualification`, `freeze`, `queue_submission` and `limits`; qualification is separate from a complete licensed study.
