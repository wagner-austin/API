---
title: WRDS equity research extends the existing experiment and coordination systems
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[model-trainer-run-record-provenance]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py"
  - "libs/covenant_ml/src/covenant_ml/markets/equity_study.py"
  - "libs/platform_core/src/platform_core/run_record.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py": "fd06501a73c52a71e1b2e7734285bde3381fffca"
  "libs/covenant_ml/src/covenant_ml/markets/equity_study.py": "e4e44e46f0031df3686b09d5067a1cea49033eb8"
  "libs/platform_core/src/platform_core/run_record.py": "7d3ca55d81674311342b179cbf476a3d61b14dd6"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "976da4e9ae76c81cf3357e797886d57092422666"
provenance:
  - "Operator request 2026-10-28: bounded extraction and replay, define substantive evaluation before expansion, then integrate agent-driven research into existing systems; board 35895aac-1b12-4cf5-b868-516196914789"
  - "Corvis live wiki and board discovery 2026-10-28: mcps-codebase agent-task-board-service, non-claude-agent-board-access and project-research; research room and WRDS task threads read before choosing the integration"
fact_checked: 2026-10-28
confidence: medium
---

# WRDS equity research extends the existing systems

The implementation path is a bounded extension: Corvis owns task claims and review;
the research libraries own evaluation rules and evidence; WRDS and HPC3 own jobs.
There is no evidence here requiring a replacement coordinator or a second research
framework. This page defines the next study and integration boundary; the monthly
study and restricted agent interface below are **designed, not implemented**.[^request]

## What is already reusable

The existing WRDS adapter verifies the exact query and CSV, builds chronological
splits with an embargo, and predicts next-observation return sign from one prior
return. That is the engineering pilot, not the monthly investment study below;
its acquisition timestamp does not prove historical vendor availability.[^adapter]
Existing equity-study code already binds a snapshot, calendar, cohort, weights
and costs. Existing market research verifies a frozen protocol before acquisition.
Reuse those invariants and shared validation/provenance primitives; do not force
CRSP return/delisting semantics into a provider-adjusted-price record.[^equity][^market]
Keep the shared RunRecord and its comparison functions as the numerical evidence
contract; a completed coordination task alone does not establish reproducibility
or out-of-sample performance.[^records]

## Study design v1, before any expanded extraction

These are proposed fixed rules for implementation, not measured facts or claims
of profitability. Freeze their machine-readable form and tests before scoring;
any change creates a new protocol version and preserves earlier attempts.[^request]

| Decision | Rule |
|---|---|
| Question and target | Can lagged return/volume features improve monthly stock selection over fixed momentum? Target each eligible stock's total return from the close of the first trading session of the next month to the corresponding close one month later, minus the eligible universe's equal-weight return. |
| Information and execution | Decide after month-end close using only observations dated through that close; execute at the next month's first trading-session close. Features cannot use that execution session's price or volume. Include distributions and delisting outcomes in targets. No fill at the same close that supplies the signal. |
| Historical universe | At each decision date, common shares with effective share codes 10/11 and exchange codes 1/2/3, absolute closing price at least $5, and at least 252 prior sessions. Rank by trailing 60-session median absolute-price-times-volume; take at most 1,000. Resolve ties by PERMNO. Use effective-dated membership, including later-delisted securities; never today's constituent list. |
| Historical availability | Preserve extraction vintage and source mappings. Effective dates govern membership. A current CRSP history is a revised historical backtest, not a proven as-published archive. Compustat stays outside v1; later financial features require report/publication availability and revision handling, not fiscal-period dates alone. |
| Calendar and missingness | Establish an independent exchange-session calendar. No forward-filled return labels or silent dropping of held positions. Missing required prices, returns or delisting outcomes make the affected evaluation incomplete; log the failure. Eligibility excludes insufficient pre-decision history with counted reasons. |
| Development and final period | Train 2000–2014; tune only on 2015–2022 through expanding chronological folds. Seal January 2023–December 2025 decisions as the final period, acquiring through the first trading-session close of February 2026 for the last exit. Purge training labels whose exit overlaps the next split and exclude the monthly training decision immediately before that split. Earlier data are feature warm-up only. |
| Baselines and model family | Equal-weight eligible universe and fixed 12-to-2-month momentum (compound months t-12 through t-2, then select the top decile). Candidate: fixed boosted-tree family using lagged returns, volatility and volume; compare ClearGBM/LightGBM on identical rows and seeds. No Compustat or LLM text signals in v1. |
| Portfolio and costs | Long-only, no leverage, equal-weight top decile, deterministic PERMNO ties, monthly rebalance. Primary all-in cost is 10 basis points times absolute traded notional, including entry/exit and turnover; report fixed 5/25-bp sensitivities. Costs are declared assumptions, not calibrated capacity or actual execution claims. |
| Primary measurement | Mean monthly net-return difference versus momentum on the same dates. Report annualized return, volatility, drawdown and turnover secondarily. Use paired 12-month moving-block bootstrap, 10,000 draws, seed 7, with intervals and sample length; 36 final decisions limit inference. |
| Trial accounting | At most 20 distinct validation candidates under v1, five fixed seeds 11/23/37/53/71 each, identical candidate budget. Select one candidate on primary validation score, tie by fewer leaves then candidate id. Evaluate the selected candidate once on the sealed final period. Seeds are repetitions, not extra hypotheses or independent market histories. |

Before acquisition expands, resolve the exact subscribed tables/columns, return
and delisting conventions, calendar and retrospective-availability limitations.
If required evidence cannot be represented, amend the protocol before results;
do not quietly substitute fields or dates. Historical data may already be known
to model pretraining or prior research: a sealed retrospective period limits
this search process, but is not a pristine prospective test. Any claim of a new
edge needs a later prospective frozen follow-up.[^request]

## Agent integration after the pilot

Add a typed proposal/admission boundary to the existing research runner: a proposal
names its hypothesis, parent, protocol digest, allowed features, model parameters
and resource budget. Admission records the attempt **before** extraction or job
submission. Link proposal → snapshot/query → code/image → scheduler attempt →
RunRecord → review, including refusal, crash, timeout, null and negative results.
One candidate with five seeds is one hypothesis with five tracked jobs.[^request]

The coordinator can assign proposer, runner and reviewer work using its current
tasks. The evaluator enforces dates, costs, permitted changes and trial limits.
The proposer receives development/validation evidence only. Final-period data
and scoring need a restricted evaluator identity and filesystem/tool permissions;
a prompt telling an agent with unrestricted SSH to avoid a directory is not
isolation. That capability boundary is a focused integration requirement, not
an implemented guarantee today. Do not autonomously launch this research loop
until admission, holdout isolation and append-only attempt recording are tested.[^request]

[^request]: Board task 35895aac-1b12-4cf5-b868-516196914789, § "OPERATOR", submitted 2026-10-28T16:38:53Z; the study rules are this session's proposed design, introduced in API commit 3358191792a4a655825b5647694fc0430cf7bd33 and calendar-corrected in 78b009119, not quotations or empirical findings.
[^adapter]: `libs/covenant_ml/src/covenant_ml/evaluation/wrds.py:186` `load_wrds_evaluation`.
[^equity]: `libs/covenant_ml/src/covenant_ml/markets/equity_study.py:55` `decode_equity_study` and `:113` `load_equity_study`.
[^market]: `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Evidence and arithmetic" and preceding command/plan contract.
[^records]: `libs/platform_core/src/platform_core/run_record.py:309` `agree_across_runs`.
