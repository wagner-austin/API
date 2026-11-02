---
title: Corrected CRSP baseline freeze
tags: [wrds, crsp, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]", "[[wrds-baseline-20261101]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v2-freeze-20261102/evidence.json"
  - "libs/covenant_ml/src/covenant_ml/markets/research_ranking.py"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v2-freeze-20261102/evidence.json": "a9e2d556798ae3b7426ab4cf0bf764d4e4d0e0b8"
  "libs/covenant_ml/src/covenant_ml/markets/research_ranking.py": "ca12a77ffe77312243c3f2d8b7671245587fd00c"
provenance:
  - "Board task 09113184-4de9-4a8b-8c77-d7ba3f24b92c; preparation measured November 2, 2026 UTC."
fact_checked: 2026-11-01
confidence: high
---

# Corrected CRSP baseline freeze

Policy `f3ce57f6d520` starts a new journal after the stopped version-1 study.
At this freeze, fifteen requests were admitted with no new licensed evaluator
job and no final grant released. The original failed journal was unchanged.
Preparation is distinct from the executed synthetic rehearsal in
[[wrds-operational-rehearsals]].[^freeze]

## Fixed comparisons and selection

The three candidates use momentum plus liquidity, momentum alone, and
liquidity alone. Each retains the original 200-tree, depth-6, 31-leaf estimator
settings and runs seeds **11, 23, 37, 53, 71**. The candidate identities are
unchanged; the baseline ceiling is reduced from twenty to these three before
any new validation results. Later agent experiments need a separately frozen
policy through the same controllers and shared queue.[^freeze]

Ranking averages paired LightGBM-minus-momentum net returns across the five
seeds for each month, then across all frozen validation months at **10 bps**.
Exact ties choose fewer leaves, then lexicographically smaller candidate digest.
The final evaluation uses seed **11**, never the best observed seed.
Intervals remain descriptive and do not choose the winner.[^rank][^freeze]

## Explicit amendments

The frozen amendment records these changes from the stopped study.[^freeze]

| Item | New frozen value and reason |
|---|---|
| Accounting rules | Protocol version 2: rejected entry orders retain cash and costs follow actual traded notional. |
| Encrypted validation | Bundle `ea669b73`; existing staged observation ciphertext is reused with the audited protocol revision. |
| Runtime | Installed source `8ef7e0f17`, image `e1560c68`, and the inspected version-2 evaluator. |
| Resource request | 8 CPUs, 128 GiB, 7,200 evaluator seconds and 121 scheduler minutes; the former four-CPU ceiling did not cover the actual allocation. |
| Candidate ceiling | Three fixed baselines, reduced from twenty; estimator and seed choices are unchanged. |

These are the complete policy-field changes recorded in `amendment.json`.
The historical universe, calendar, split dates, costs, uncertainty settings,
selection window and tie rules remain fixed. The final holdout stays separate.
All input documents and fifteen request/recipe digests are retained.[^freeze]

## Measured preparation

The existing preparation reporter read all fifteen admitted rows with empty
job, selection and final-release fields. Native preflight admitted the eight-CPU
request on the free partition. Controller inspection verified 623 installed
files against seven frozen wheels. These measurements establish prepared
inputs and feasibility; they do not establish successful model fitting.[^freeze]

[^freeze]: `tools/hpc3/artifacts/wrds-baseline-v2-freeze-20261102/evidence.json`, `prepared_state`, `request_resources`, `native_preflight`, `selection_rule`, `candidate_budget`, `service_reports`, `old_journal_unchanged` and `files`; pinned sibling documents include the old/new policies, evaluation, request matrix and program-written reports.
[^rank]: `libs/covenant_ml/src/covenant_ml/markets/research_ranking.py`, `candidate_means` and `winning_attempt`; required seeds and months must be complete before ranking.
