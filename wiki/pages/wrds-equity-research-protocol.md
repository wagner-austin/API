---
title: WRDS equity research extends the existing experiment and coordination systems
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[model-trainer-run-record-provenance]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/evaluation/evidence.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py"
  - "libs/covenant_ml/tests/markets/test_crsp_extract.py"
  - "libs/covenant_ml/tests/markets/test_crsp_panel.py"
  - "libs/covenant_ml/docs/crsp-monthly-v1.json"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py"
  - "libs/covenant_ml/tests/markets/test_crsp_protocol.py"
  - "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-date-coverage.json"
  - "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json"
  - "CODE_STANDARDS.md"
  - "libs/platform_core/src/platform_core/stage_manifest.py"
  - "libs/covenant_ml/tests/markets/test_research_execution.py"
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
  - "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "tools/fleet/src/fleet/cli/research_queue.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "tools/fleet/src/fleet/core/research_hpc.py"
  - "tools/fleet/src/fleet/core/research_wrds.py"
  - "tools/fleet/src/fleet/cli/tick.py"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/evaluation/evidence.py": "3035c22326bc69ff6d06c5e5f72d6660290e270b"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py": "b48812de06b8ffa1a4c095f26d45f44cfbc7a527"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py": "838e19bcd90c3a99173409378160b14035d12b5d"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py": "97d5165abfe06c0989c5d72653f79979ec104aca"
  "libs/covenant_ml/tests/markets/test_crsp_extract.py": "310c620a4669e5d80891e15e5788e8b9711cede8"
  "libs/covenant_ml/tests/markets/test_crsp_panel.py": "6d9f476a5dc05858e71e3f5350cf81cd0676018c"
  "libs/covenant_ml/docs/crsp-monthly-v1.json": "42137d1e107819d0030fcae10ca1ca560d6e8f6d"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py": "221b2b1c6d526a02786d78837a228b1bd5203e7f"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "8d3e0b6a7ebb84dd44d62bfb3c4acdc557678cb8"
  "libs/covenant_ml/tests/markets/test_crsp_protocol.py": "28789469447c10b67050d3fd4995471c29d85893"
  "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-date-coverage.json": "5f5d4ee9854c67bb76bf3894333c965b2add1763"
  "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json": "3aa4b35290d76b1a8f3c404521c96b0bddb96291"
  "CODE_STANDARDS.md": "474681549e2caa9dc26a049f53c793ab17eaf830"
  "libs/platform_core/src/platform_core/stage_manifest.py": "661ec21bf8846370ad93aefa4460371d1981547e"
  "libs/covenant_ml/tests/markets/test_research_execution.py": "6c83658eb2e0447a2a06fd12d23e6c3b138c1d30"
  "libs/covenant_ml/src/covenant_ml/evaluation/wrds.py": "bb7b6f6db20f501916ba08821c0e4efd2166bbe7"
  "libs/covenant_ml/src/covenant_ml/markets/equity_study.py": "e4e44e46f0031df3686b09d5067a1cea49033eb8"
  "libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py": "2ff9370dd458fcc4cf3e5a7d21a490583f88ae05"
  "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py": "6d95079f002b370fa9a8920c3d361a4f803e6136"
  "libs/covenant_ml/src/covenant_ml/evaluation/paired.py": "930ff0876202a68c51454352dc63c0b7d80160c8"
  "libs/covenant_ml/src/covenant_ml/evaluation/splits.py": "1a5b1a1a94e4666d039ea21557480b66eb95bbac"
  "libs/platform_core/src/platform_core/run_record.py": "7d3ca55d81674311342b179cbf476a3d61b14dd6"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "44dfcd58adde8408d1b34a558214ac70d36af563"
  "libs/platform_core/src/platform_core/event_journal.py": "b0660ebf4ebe52161aed9b5e7c425064eac6782e"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "ca09fde6b7e9615b802b58fd98345fa936ab809d"
  "libs/covenant_ml/src/covenant_ml/markets/research_records.py": "3a5069fbde15c85b942237313970187f8bff7c00"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "8aa230d8f7be88a7d5a968f8bd00d008c5cd2ac9"
  "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py": "bd2c9fd8c7810948c0702ac5f9d5a502c7054289"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "18737b9e2165b22b857b7a2221dd7f7f98733a61"
  "tools/fleet/src/fleet/cli/research_queue.py": "aced6395998134dd8d9532dc7243b87745eab0c6"
  "tools/fleet/src/fleet/core/research_control.py": "84af969117df4a38b47eaae30b25ddb14803d145"
  "tools/fleet/src/fleet/core/research_hpc.py": "ac4a658ffe4ca6beae15ce56df08f712e7bddca2"
  "tools/fleet/src/fleet/core/research_wrds.py": "2130dac9b040de079efcdb5b5b238c6cb6022927"
  "tools/fleet/src/fleet/cli/tick.py": "596da64a093028b74c9bda926b6035da5c64cb72"
provenance:
  - "Vendor semantics: tech-wiki pages crsp-ciz-daily-return-contract, crsp-ciz-daily-return-quality and crsp-ciz-aggregate-return-quality pin the July 2026 CRSP CIZ guide; these external references remain authoritative for field meanings. Board task 54b55cc7-ff40-4753-a04e-776bce1830d0."
  - "Ownership review 2026-10-29: MCPs source snapshot 06512f6450c960e52409448f306c1254ed633fb6 is external to this API wiki's source_paths; live queue, installed controller and board reads are identified separately in the review footnotes."
  - "Operator request 2026-10-28: bounded extraction and replay, define substantive evaluation before expansion, then integrate agent-driven research into existing systems; board 35895aac-1b12-4cf5-b868-516196914789"
  - "Corvis live wiki and board discovery 2026-10-28: mcps-codebase agent-task-board-service, non-claude-agent-board-access and project-research; research room and WRDS task threads read before choosing the integration"
fact_checked: 2026-10-29
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

## Ownership review and remaining boundaries

The 2026-10-29 review traced the research submission, recovery and result path,
including its failure tests. It found no compatibility shim, alternate backend
fallback, passthrough alias or new type alias in the reviewed integration.
This is a bounded source review, not a claim about every file in either repository.
The named guard rules remain `hook-fallback-dispatch`,
`shim-compatibility-marker`, `passthrough-alias`, `qr-legacy-validators`
and `typealias-forbidden`; tests also exercise identity and recovery behavior.[^review]

| Responsibility | Authoritative implementation reused |
|---|---|
| Tenant-scoped submission and deduplication | MCPs `submitDispatchJob`, its research routing functions and the existing dispatch table's immutable research reference. |
| Controller capacity and polling | `fleet.cli.research_queue` uses shared queue operations and resource leases; native scheduler waits retain their receipts. |
| Proposal configuration and attempt history | `research_records` uses the benchmark codec; `research_journal` uses `platform_core.event_journal`. |
| Native execution and collection | HPC3/WRDS adapters invoke the maintained `hpc3` commands. Resumed WRDS collection rechecks local bytes against freshly verified native evidence. |
| Staging identity and numerical evidence | Shared `StageManifest` and `RunRecord` codecs; no research-specific replacement records. |

The review found one concrete provenance defect: the bundle writer hashed UTF-8
text but wrote it in text mode, allowing Windows newline translation to change
the emitted bytes. `prepare_bundle` now writes, hashes and sizes the same byte
payload. The existing execution test suite reads those files back and verifies
their exact hashes, lengths and checksum content through the shared manifest
codec. The repair adds no helper or second writer. Its full fleet check passed 4,525 tests
with 100% coverage. Windows CI also passed 4,525 tests with 100% statement and
branch coverage; the independent staging-repair audit closed VERIFIED.[^bundle]

A fresh read of live queue job `fb62c111-f5c8-48f6-a790-899e68f76920` retained
native job `58586013` and RunRecord digest
`49b7a44e64b76b7d153a6a3fb1a3f1805ea328720fbc855ca6e0bc892d75af17`.
The installed controller still reports four completed trials and one refusal;
its canonical full-report digest remains
`727b765412d7ce3b19edfc71b45311135f06d72c1486e307a641f094f5bd9ab6`.
This read consumed no trial budget and submitted no allocation.[^live-review]

The earlier controller-path repair passed its Windows check and independent
audit. The native replay case also passed, but its parent board task
`8b19319c` remains waiting on the unrelated minutes-corpus root-cause task
`671c2919`; that parent audit is not complete. Label-overlap exclusion and
paired moving-block primitives now have implementations; their integration
into monthly return accounting and restricted holdout access remains work
described below.[^live-review][^sampling][^splits]

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
| `ChronologicalGroupedSplits` | Partitions whole distinct time keys and requires actual label availability strictly before stopping and scoring boundaries. Validation stopping rows come from training. | Bind the monthly study's fixed dates and actual outcome-availability timestamps. The separate embargo counts keys, not elapsed time. |
| `build_resampling` / `bootstrap_weights` / `moving_block_weights` | The first two retain fixed, non-overlapping blocks. The new primitive samples overlapping, noncircular blocks and truncates the last block to preserve the exact observation count. | Integrate monthly return differences and the frozen block length into the study worker. |
| `paired_mean_difference` / `compare_moving_block` | Newey-West errors remain available. The new paired comparison requires identical consecutive month identities and records a distinct moving-block percentile method. | Publish the monthly study's paired interval. Its uncentered bootstrap sign-tail probability is a diagnostic, not a calibrated null-hypothesis test. |

The shared admission journal, queue, native scheduler adapters and RunRecord
publisher already compose. Extending the numerical evaluator does not require
another coordinator or another attempt ledger. The diagnostic source selects
`validation_split`, but reads a manifest whose underlying snapshot still resides
under the shared account. Final-period enforcement therefore also needs an
execution identity that proposer tools cannot use to read the held-out files;
the split function alone is not that boundary.[^execution][^controllers][^splits]

## Study design v1, before any expanded extraction

The design is frozen in `libs/covenant_ml/docs/crsp-monthly-v1.json`. Its codec
and calendar, membership, execution-price and terminal-value validators are
implemented; holdings evaluation and holdout isolation remain integration work.
These are study assumptions, not profitability findings.[^frozen]

| Decision | Rule |
|---|---|
| Question and target | Can lagged return/volume features improve monthly stock selection over fixed momentum? Target each eligible stock's total return from the close of the first trading session of the next month to the corresponding close one month later, minus the eligible universe's equal-weight return. |
| Information and execution | Decide after month-end close using only observations dated through that close; execute at the next month's first trading-session close. Features cannot use that execution session's price or volume. Include distributions and delisting outcomes in targets. No fill at the same close that supplies the signal. |
| Historical universe | At each decision date, common shares with effective share codes 10/11 and exchange codes 1/2/3, absolute closing price at least $5, and at least 252 prior sessions. Rank by trailing 60-session median absolute-price-times-volume; take at most 1,000. Resolve ties by PERMNO. Use effective-dated membership, including later-delisted securities; never today's constituent list. |
| Historical availability | Preserve extraction vintage and source mappings. Effective dates govern membership. A current CRSP history is a revised historical backtest, not a proven as-published archive. Compustat stays outside v1; later financial features require report/publication availability and revision handling, not fiscal-period dates alone. |
| Calendar and missingness | Establish an independent exchange-session calendar. No forward-filled return labels or silent dropping of held positions. Missing required prices, returns or delisting outcomes make the affected evaluation incomplete; log the failure. Eligibility excludes insufficient pre-decision history with counted reasons. |
| Development and final period | Train 2000–2014; tune only on 2015–2021 through expanding chronological folds. Seal January 2022–December 2024 decisions as the final period, acquiring through February 2025 for the last exit. Purge training labels whose exit overlaps the next split and exclude the monthly training decision immediately before that split. Warm-up begins January 1998. |
| Baselines and model family | Equal-weight eligible universe and fixed 12-to-2-month momentum (compound months t-12 through t-2, then select the top decile). Candidate: fixed boosted-tree family using lagged returns, volatility and volume; compare ClearGBM/LightGBM on identical rows and seeds. No Compustat or LLM text signals in v1. |
| Portfolio and costs | Long-only, no leverage, equal-weight top decile, deterministic PERMNO ties, monthly rebalance. Primary all-in cost is 10 basis points times absolute traded notional, including entry/exit and turnover; report fixed 5/25-bp sensitivities. Costs are declared assumptions, not calibrated capacity or actual execution claims. |
| Primary measurement | Mean monthly net-return difference versus momentum on the same dates. Report annualized return, volatility, drawdown and turnover secondarily. Use paired 12-month moving-block bootstrap, 10,000 draws, seed 7, with intervals and sample length; 36 final decisions limit inference. |
| Trial accounting | At most 20 distinct validation candidates under v1, five fixed seeds 11/23/37/53/71 each, identical candidate budget. Select one candidate on primary validation score, tie by fewer leaves then candidate id. Evaluate the selected candidate once on the sealed final period. Seeds are repetitions, not extra hypotheses or independent market histories. |

The date change preceded outcome inspection: WRDS compute probes `40544368`
and `40544470` found legacy membership only through December 2024 and CIZ
returns/calendar through December 2025. The second probe also confirmed
terminal-event dates through 2025-12-30, with stored dates through 2025-12-31.
The previous February 2026 exit was unavailable.
Table extents do not prove individual-history completeness.[^coverage]
Historical data may already be known
to model pretraining or prior research: a sealed retrospective period limits
this search process, but is not a pristine prospective test. Any claim of a new
edge needs a later prospective frozen follow-up.[^request]

## CIZ data contract for the monthly implementation

The vendor definitions live in the tech wiki's
[daily return contract](https://github.com/wagner-austin/tech-wiki/blob/main/pages/crsp-ciz-daily-return-contract.md),
[daily quality contract](https://github.com/wagner-austin/tech-wiki/blob/main/pages/crsp-ciz-daily-return-quality.md)
and [aggregate quality contract](https://github.com/wagner-austin/tech-wiki/blob/main/pages/crsp-ciz-aggregate-return-quality.md).
This protocol owns the application choices below. These choices specify the next
adapter extension; they are not a claim that the existing `dsf` engineering pilot
implements CIZ monthly outcomes.[^ciz-contract][^adapter]

The read-only catalog probe established the subscribed `crsp_a_stock` relations
and candidate columns. It downloaded metadata only. Four exact ordered column
arrays are now frozen in the machine-readable protocol; its decoder refuses
omitted, additional, repeated or reordered columns. Column existence and type
do not prove field semantics, historical completeness or publication availability.[^ciz-schema][^frozen]

| Use | Selected input and constraint |
|---|---|
| Target return | `crsp_a_stock.stkdlysecuritydata`: `permno`, `dlycaldt`, `dlyret`, `dlyretmissflg`, `dlyretdurflg`, `dlydelflg`. Compound the held interval `(entry close, exit close]`; never substitute vendor calendar-month returns. |
| Feature inputs and price validation | The same relation's `dlyprc`, `dlyprcflg`, `dlyvol`, plus `dlyprevdt` for return-span validation. Ordinary entry/exit requires a finite positive `TR` closing trade. A selected security without an executable price makes the run incomplete; future fill prices cannot change pre-decision selection. |
| Ordinary daily completeness | Require finite return, missingness flag `NA`, nonterminal flag `N`, and duration in `D1,D2,D3,D4,DU`, with independent session and previous-date checks. An absent or unknown flag is an explicit validation error. |
| Terminal reconciliation | `stkdelists`: `permno,delistingdt,delret,delretmisstype,deldlydt,delamtdt,delnextdt`. Require finite return at least -1 and missingness flag `NA`. Value once at the latest stored/amount/present next-price date; retain a receivable until modeled reinvestment at the next rebalance strictly after that date. Missing terminal value makes the held outcome incomplete; a valuation date after the scored exit also fails. |
| Membership evidence | `dsenames`: `permno,namedt,nameendt,shrcd,exchcd`. Exactly one inclusive effective interval must apply at decision time. Preserve numeric codes 10/11 and 1/2/3 directly; do not infer equivalent mnemonic classifications or extrapolate after December 2024. Entry membership does not filter later held outcomes. |
| Session evidence | `metaexchangecalendar`: `caldt,exchangegroup,tradingflg`. Select group `CIZ`; retain every calendar day and its `Y`/`N` flag. Validate contiguous full months and construct windows from declared sessions, including historical Saturday sessions. Never infer the calendar from stock rows. |
| Incomplete observations | Exclude insufficient pre-decision history with counted reasons. A missing required held return or unresolved terminal event stops publication of a complete evaluation; no zero fill, forward fill or silent security removal. Aggregate flags serve diagnostics only and cannot override missing daily evidence. |
| Historical availability | Preserve source capture, query, extraction timestamp, checksums and effective-date joins. The resulting study is a revised-history retrospective backtest. These dates do not establish when a vendor first published a value or what a historical subscriber actually saw. |

The terminal rule is a declared retrospective valuation and settlement proxy.
Neither DelDlyDt nor the amount/next-price dates establish actual spendable-cash
availability. The frozen validators reject unresolved evidence; they do not
establish executable historical fills or a complete monthly study. Existing
staging, RunRecord, admission, queue and native controllers retain ownership.[^frozen][^ciz-contract][^controllers]

## Agent integration after the pilot

The input adapter now reads four relation CSVs and their exact SQL alongside
the frozen protocol, using the shared staging file identities and the evidence
reader lifted from the single-security diagnostic. It verifies consumed bytes,
refuses duplicate source keys and changed projections, and preserves explicit
missing values. A checksum binds bytes; it does not attest vendor authorship or
prove that a supplied query generated those bytes.[^panel]

Decision construction uses only observations through the decision close, counts
one first exclusion reason per rejected historical security, and keeps empty
decision months visible. Liquidity is the trailing 60-session median absolute
price times volume; equal liquidity sorts by PERMNO. Momentum skips the entire
decision month and compounds the preceding eleven calendar months, with every
required daily predecessor checked against the independent session calendar.
The 252-session history window ends at the decision close. Future fill prices
never filter this universe. Hand-computable fixture cases test −12% momentum,
invariance under changed future returns, prices, volumes and new listings, and
a nonconstant liquidity window whose median (2000) differs from its mean. A
reordered CSV header with a valid checksum is refused as a projection change.
This is input construction, not an executed monthly portfolio study; holdings,
cost accounting, worker integration and enforced holdout isolation remain.[^panel]

The controller's `python` executable and `workspace` configuration use absolute
native filesystem paths. HPC3 `root`, `entrypoint` and `image` use absolute POSIX
paths regardless of the controller's operating system. All five fields reject
parent-directory traversal. The execution command consumes the first two locally;
the frozen staging manifest and compute recipe consume the remote paths.[^paths]

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
[^review]: `CODE_STANDARDS.md`, named guard-rule table; API functions cited in [^queue], [^controllers], [^admission] and [^records]. External MCPs source snapshot 06512f6450c960e52409448f306c1254ed633fb6: `packages/db/src/fleet-dispatch/research.ts` functions `requireResearchRouting` and `existingResearchJob`, and `packages/db/migrations/877_fleet_dispatch_research_requests.sql` immutable reference constraints. Scope and findings: board task 79c9a370-763e-4fd0-a609-cd1a64080b19.
[^bundle]: `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py` function `prepare_bundle`; `libs/platform_core/src/platform_core/stage_manifest.py` codecs; `libs/covenant_ml/tests/markets/test_research_execution.py` case `test_bundle_manifest_identifies_exact_written_bytes`. API source commit 6f98b519968ea5435119aec0d21105f6e3cb2997 and regression commit 1eb02136baa6b86938b73dbad3395e0a41d9ce5c; fleet job 748608d6-faf0-4cb2-ac45-50fe75e9a977 passed at the latter tip. Windows GitHub check `check`, run 40820758612, passed 4,525 tests in 209.66 seconds with 100% coverage. Board repair db0ddd19-bd02-4941-ac09-f4a88a8361f0 closed VERIFIED through audit 0fa4a1c5-7a6f-4975-8682-28c33448e5e2 at 2026-10-29T05:05:50Z.
[^live-review]: Live `dispatch_get` and installed `research_proposals.py status` reads on 2026-10-29, preserved in board task 79c9a370 thread post 1a29cf1d-3ed9-4d47-a6a6-8a23c7c08bd3. `task_get 3261e090` records audit ed3e8e63 VERIFIED; `task_get 8b19319c` records its remaining wait on 671c2919. Earlier served replay case: MCPs `execution/tests/served/research-queue-replay.test.ts`, two submissions and three controller reads in served run c469a100-f103-4e26-a246-396d81b5ee04; the entire served suite was not green.
[^queue]: `tools/fleet/src/fleet/cli/research_queue.py` functions `tick` and `reconcile`; `tools/fleet/src/fleet/cli/tick.py` function `plan_tick`. Board task 3dff6fbf-e09a-430f-b13e-5dccb505ba80 tracks rollout separately from implementation.
[^window]: `libs/covenant_ml/src/covenant_ml/markets/equity_outcomes.py:67` `evaluate_equity_window`, complete-calendar check and `net_selected = gross_selected - cost` arithmetic.
[^sampling]: `libs/covenant_ml/src/covenant_ml/evaluation/resampling.py`, `build_resampling`, `bootstrap_weights` and `moving_block_weights`; `libs/covenant_ml/src/covenant_ml/evaluation/paired.py`, `paired_mean_difference` and `compare_moving_block`. The monthly integration column remains design work; the listed numerical primitives are implemented.
[^splits]: `libs/covenant_ml/src/covenant_ml/evaluation/splits.py` class `ChronologicalGroupedSplits`, methods `__call__` and `validation_split`.
[^controllers]: `tools/fleet/src/fleet/core/research_control.py` functions `read_specification` and `verified_input`; `tools/fleet/src/fleet/core/research_hpc.py` and `tools/fleet/src/fleet/core/research_wrds.py`, function `poll` in each.
[^adapter]: `libs/covenant_ml/src/covenant_ml/evaluation/wrds.py:129` `load_wrds_evaluation`.
[^panel]: `libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py`, `load_crsp_sources`; `libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py`, `decode_crsp_rows`; `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `build_crsp_panel`; `libs/covenant_ml/src/covenant_ml/evaluation/evidence.py`, `read_evidence`; `libs/covenant_ml/tests/markets/test_crsp_extract.py` and `libs/covenant_ml/tests/markets/test_crsp_panel.py`, real-file and independent arithmetic fixtures. Board task 869761ed-980a-4595-a45b-fc400aab371e tracks fleet verification separately from implementation.
[^ciz-schema]: `tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json`, candidate_fields, relation_column_counts and limits. The underlying metadata CSV, identified by that receipt's artifact_sha256, additionally confirms stkdlysecuritydata.dlyprevdt; read 2026-10-29, no security observations.
[^frozen]: `libs/covenant_ml/docs/crsp-monthly-v1.json`, complete frozen settings; `libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py`, `decode_crsp_protocol`; `libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py`, `crsp_windows`, `crsp_member_on`, `require_crsp_execution_price`, `crsp_terminal_value`; `libs/covenant_ml/tests/markets/test_crsp_protocol.py`, independent calendar examples and invalid-evidence refusals. Vendor guide pp. 7, 11–12, 25, 88, 91–92 supplies calendar, date, price and missingness definitions; reinvestment timing is a study assumption, not a vendor guarantee.
[^coverage]: `tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-date-coverage.json`, date_extents, query_sha256 and limits; read-only native job 40544470 on 2026-10-29, extending the three-relation probe 40544368 with stkdelists coverage. No security observations were returned or scored.
[^ciz-contract]: External tech-wiki pages `crsp-ciz-daily-return-contract`, `crsp-ciz-daily-return-quality`, `crsp-ciz-aggregate-return-quality`, field and flag dictionaries plus Application notes. Their single captured vendor guide is https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d, pp. 14-18, 49, 63, 81-82 and 90-92; text capture SHA-256 bb4792deb927b972a8bef1a656ba1fc5442b80783e10eecb3e633d4243334b2e. Application constraints here are proposed protocol rules, not vendor guarantees or implemented monthly behavior.
[^equity]: `libs/covenant_ml/src/covenant_ml/markets/equity_study.py:55` `decode_equity_study` and `:113` `load_equity_study`.
[^market]: `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Evidence and arithmetic" and preceding command/plan contract.
[^records]: `libs/platform_core/src/platform_core/run_record.py:309` `agree_across_runs`.
[^admission]: `libs/covenant_ml/src/covenant_ml/markets/research_records.py` frozen contracts; `libs/covenant_ml/src/covenant_ml/markets/research_journal.py` admission and lifecycle reducers; `libs/platform_core/src/platform_core/event_journal.py` transactional persistence; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Board task 399048d8-b1d6-4e12-a4d3-a0ce2281c810.
[^execution]: `libs/covenant_ml/src/covenant_ml/markets/research_execution.py` submission and reconciliation; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py` validation diagnostic; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission", execution settings and limitations. Board task 6d2d086c-5242-482b-ac81-8a61dd8e354d.
[^paths]: `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py:52` `decode_research_cluster` and `:101` `prepare_bundle`; `libs/covenant_ml/src/covenant_ml/markets/research_execution.py:70` `_command`. Board task 3261e090-1354-43ae-8703-a8cf5671f5f9 corrects the former use of POSIX validation for local controller paths.
