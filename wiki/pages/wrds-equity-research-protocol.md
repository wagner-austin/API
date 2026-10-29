---
title: WRDS equity research extends the existing experiment and coordination systems
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[model-trainer-run-record-provenance]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py"
  - "libs/covenant_ml/src/covenant_ml/markets/equity_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/paired.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/splits.py"
  - "libs/platform_core/src/platform_core/run_record.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/platform_core/src/platform_core/event_journal.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_journal.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_records.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_execution.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "tools/fleet/src/fleet/cli/research_queue.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "tools/fleet/src/fleet/core/research_hpc.py"
  - "tools/fleet/src/fleet/core/research_wrds.py"
  - "tools/fleet/src/fleet/cli/tick.py"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py": "fd06501a73c52a71e1b2e7734285bde3381fffca"
  "libs/covenant_ml/src/covenant_ml/markets/equity_study.py": "e4e44e46f0031df3686b09d5067a1cea49033eb8"
  "libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py": "2ff9370dd458fcc4cf3e5a7d21a490583f88ae05"
  "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py": "e72d1e11e542cb9353f24142a40ae3029698c9b0"
  "libs/covenant_ml/src/covenant_ml/evaluation/paired.py": "d039fbdb6910716d9f4b2b4eb7716083794e9049"
  "libs/covenant_ml/src/covenant_ml/evaluation/splits.py": "4efca7be5d2ce636c20510223eb3c5bfc8379516"
  "libs/platform_core/src/platform_core/run_record.py": "7d3ca55d81674311342b179cbf476a3d61b14dd6"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "4e4740d246c440f85c1a5ab12e89f383c5a56cab"
  "libs/platform_core/src/platform_core/event_journal.py": "b0660ebf4ebe52161aed9b5e7c425064eac6782e"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "ca09fde6b7e9615b802b58fd98345fa936ab809d"
  "libs/covenant_ml/src/covenant_ml/markets/research_records.py": "3a5069fbde15c85b942237313970187f8bff7c00"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "8aa230d8f7be88a7d5a968f8bd00d008c5cd2ac9"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "18737b9e2165b22b857b7a2221dd7f7f98733a61"
  "tools/fleet/src/fleet/cli/research_queue.py": "aced6395998134dd8d9532dc7243b87745eab0c6"
  "tools/fleet/src/fleet/core/research_control.py": "84af969117df4a38b47eaae30b25ddb14803d145"
  "tools/fleet/src/fleet/core/research_hpc.py": "ac4a658ffe4ca6beae15ce56df08f712e7bddca2"
  "tools/fleet/src/fleet/core/research_wrds.py": "2130dac9b040de079efcdb5b5b238c6cb6022927"
  "tools/fleet/src/fleet/cli/tick.py": "596da64a093028b74c9bda926b6035da5c64cb72"
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
framework. Frozen proposal admission, durable attempt recording and HPC3 submission
and reconciliation are implemented in the existing libraries. The monthly study
and restricted holdout identity below remain integration work.[^request][^admission][^execution]

## Shared queue integration

The fleet research lane has an implementation in the existing fleet package:
it claims shared dispatch rows, uses existing resource leases during bounded
controller operations, and polls native jobs through maintained controllers.
Four requests may remain active, with the oldest renewed receipt polled each
tick. Native scheduler waiting does not occupy a local execution slot.
Board task `3dff6fbf` verified deployed automatic reconciliation, and task
`f5e2befd` verified the queue contract. Their audits closed VERIFIED on
2026-10-29 UTC. The hpc3 wiki page `wrds-installed-controller-verification`
holds the runtime and live receipts; this page describes the code seams.[^queue]

The queue reference contains backend, exact private specification digest and
attempt. The native adapter checks tenant binding and input hashes on each
poll. HPC3 keeps proposal admission and its attempt journal; WRDS keeps held
submission and its durable receipt. Transport failures retain those identities.
Completion requires fresh native evidence and verified RunRecord metadata;
no research rows or credentials are placed on the board.[^queue][^controllers]

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

### Reuse boundaries checked against source

The existing implementations provide useful parts of the proposed study, but
their current contracts differ from that study in specific ways:[^equity][^window][^sampling][^splits]

| Existing seam | Verified behavior | Extension needed for the monthly study |
|---|---|---|
| `equity_study.load_equity_study` | Binds a verified snapshot, cohort, selected weights and independently supplied session calendar. | Bind effective-dated CRSP membership and decision/entry/exit dates, preserving its frozen-design checks. |
| `equity_outcomes.evaluate_equity_window` | Requires the entire frozen cohort and every declared session; scores adjusted-price endpoints and subtracts one fixed round-trip cost. | Carry holdings between rebalances, charge absolute traded notional against drifted pre-trade weights, and represent return/delisting events explicitly. A fixed cost per window is not turnover accounting. |
| `ChronologicalGroupedSplits` | Partitions whole distinct time keys; validation stopping rows come from training. | Freeze explicit date boundaries and purge labels by their actual exit/availability timestamps. The existing embargo counts keys, not elapsed time or label intervals. |
| `build_resampling` / `bootstrap_weights` | Forms fixed, non-overlapping blocks by integer division and resamples their counts. | Add explicit overlapping moving-block semantics to the shared evaluation package for the proposed monthly statistic; retain the current diagnostic's declared semantics. |
| `paired_mean_difference` | Uses Newey-West errors for block-unit means. | The proposed primary moving-block bootstrap interval needs a distinct measured method; selecting the existing block unit does not implement it. |

The shared admission journal, queue, native scheduler adapters and RunRecord
publisher already compose. Extending the numerical evaluator does not require
another coordinator or another attempt ledger. The diagnostic source selects
`validation_split`, but reads a manifest whose underlying snapshot still resides
under the shared account. Final-period enforcement therefore also needs an
execution identity that proposer tools cannot use to read the held-out files;
the split function alone is not that boundary.[^execution][^controllers][^splits]

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

The implemented typed proposal/admission boundary accepts a proposal that
names its hypothesis, parent, protocol digest, allowed features, model parameters
and resource budget. Admission records the attempt **before** extraction or job
submission. Link proposal → snapshot/query → code/image → scheduler attempt →
RunRecord → review, including refusal, crash, timeout, null and negative results.
One candidate with five seeds is one hypothesis with five tracked jobs.[^request]

`research_records` reuses the benchmark configuration codec; `research_journal`
binds the frozen policy to `platform_core.event_journal`, lifted from the existing
execution rehearsal and used by both consumers. Raw object proposals are recorded
before decoding. Policy refusals remain visible, and admitted candidate/seed pairs
stay consumed after failure. The lifecycle records submission intent, unique job
identity and an identity-checked shared RunRecord. `scripts.research_proposals`
exposes init, propose, status, submit and reconcile operations. Submission uses the
maintained cluster staging and job commands. The submission boundary compares typed
proposal values, preserving raw admission bytes while accepting JSON key reordering;
changed proposal values remain refused. Reconciliation requires native Slurm
accounting and verified results through the cluster evidence command. Transport
ambiguity remains unresolved until exactly one matching ledger row is found; no
automatic resubmission is allowed. The validation worker reuses the existing
trainers, metrics and RunRecord publisher, with early stopping drawn from training.
These are provenance checks over owner-writable evidence, not protection against
an adversary controlling the account.[^execution] This control layer refuses final-partition proposals
but does not implement filesystem isolation or the substantive monthly evaluator.
The CLI's policy is evaluator-owned configuration, not a field agents may replace
in a proposal.[^admission]

The coordinator can assign proposer, runner and reviewer work using its current
tasks. The evaluator enforces dates, costs, permitted changes and trial limits.
The proposer receives development/validation evidence only. Final-period data
and scoring need a restricted evaluator identity and filesystem/tool permissions;
a prompt telling an agent with unrestricted SSH to avoid a directory is not
isolation. That capability boundary is a focused integration requirement, not
an implemented guarantee today. Do not autonomously launch this research loop
until admission, holdout isolation and append-only attempt recording are tested.[^request]

[^request]: Board task 35895aac-1b12-4cf5-b868-516196914789, § "OPERATOR", submitted 2026-10-28T16:38:53Z; the study rules are this session's proposed design, introduced in API commit 3358191792a4a655825b5647694fc0430cf7bd33 and calendar-corrected in 78b009119, not quotations or empirical findings.
[^queue]: `tools/fleet/src/fleet/cli/research_queue.py` functions `tick` and `reconcile`; `tools/fleet/src/fleet/cli/tick.py` function `plan_tick`. Board task 3dff6fbf-e09a-430f-b13e-5dccb505ba80 tracks rollout separately from implementation.
[^window]: `libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py:67` `evaluate_equity_window`, complete-calendar check and `net_selected = gross_selected - cost` arithmetic.
[^sampling]: `libs/covenant_ml/src/covenant_ml/evaluation/resampling.py:53` `build_resampling` and `:135` `bootstrap_weights`; `libs/covenant_ml/src/covenant_ml/evaluation/paired.py` function `paired_mean_difference`. These sources were read during board task 8b19319c-1aa0-4e6a-bc83-0c36bd3b50bb; the extension column is design synthesis, not an implementation claim.
[^splits]: `libs/covenant_ml/src/covenant_ml/evaluation/splits.py` class `ChronologicalGroupedSplits`, methods `__call__` and `validation_split`.
[^controllers]: `tools/fleet/src/fleet/core/research_control.py` functions `read_specification` and `verified_input`; `tools/fleet/src/fleet/core/research_hpc.py` and `tools/fleet/src/fleet/core/research_wrds.py`, function `poll` in each.
[^adapter]: `libs/covenant_ml/src/covenant_ml/evaluation/wrds.py:186` `load_wrds_evaluation`.
[^equity]: `libs/covenant_ml/src/covenant_ml/markets/equity_study.py:55` `decode_equity_study` and `:113` `load_equity_study`.
[^market]: `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Evidence and arithmetic" and preceding command/plan contract.
[^records]: `libs/platform_core/src/platform_core/run_record.py:309` `agree_across_runs`.
[^admission]: `libs/covenant_ml/src/covenant_ml/markets/research_records.py` frozen contracts; `libs/covenant_ml/src/covenant_ml/markets/research_journal.py` admission and lifecycle reducers; `libs/platform_core/src/platform_core/event_journal.py` transactional persistence; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Board task 399048d8-b1d6-4e12-a4d3-a0ce2281c810.
[^execution]: `libs/covenant_ml/src/covenant_ml/markets/research_execution.py` submission and reconciliation; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py` validation diagnostic; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission", execution settings and limitations. Board task 6d2d086c-5242-482b-ac81-8a61dd8e354d.
