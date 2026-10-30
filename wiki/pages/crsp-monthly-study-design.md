---
title: Frozen CRSP monthly study design
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[wrds-proposer-isolation]]", "[[wrds-baseline-readiness]]"]
source_paths:
  - "libs/covenant_ml/docs/crsp-monthly-v1.json"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py"
  - "libs/covenant_ml/tests/markets/test_crsp_protocol.py"
  - "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-date-coverage.json"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py"
  - "libs/covenant_ml/tests/markets/test_crsp_holdings.py"
  - "libs/covenant_ml/tests/markets/test_crsp_study.py"
  - "libs/covenant_ml/tests/markets/test_research_trial.py"
source_git_blobs:
  "libs/covenant_ml/docs/crsp-monthly-v1.json": "42137d1e107819d0030fcae10ca1ca560d6e8f6d"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py": "221b2b1c6d526a02786d78837a228b1bd5203e7f"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "e721684f962c297aa60e8c1ebc80c1f3376da02d"
  "libs/covenant_ml/tests/markets/test_crsp_protocol.py": "28789469447c10b67050d3fd4995471c29d85893"
  "tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-date-coverage.json": "5f5d4ee9854c67bb76bf3894333c965b2add1763"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "8cfeae2c102164539246bd6ea99b93051e8ccb2b"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "c63fd95cfe672e9e198cd1b5cab895e97466dc77"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "47eaddf94090471d7dc84e188829de4d8cfaf1dd"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "b5081a606ba9f8d21673ef75c9a7da5a197c6dde"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "b797984c6422a9c99cfd4db6c09ef92a40c755f9"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "448c595406899be546fe79b550df921a6761dcd8"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "Vendor semantics: tech-wiki pages crsp-ciz-daily-return-contract, crsp-ciz-daily-return-quality and crsp-ciz-aggregate-return-quality pin the July 2026 CRSP CIZ guide; these external references remain authoritative for field meanings. Board task 54b55cc7-ff40-4753-a04e-776bce1830d0."
  - "Operator request 2026-10-28: bounded extraction and replay, define substantive evaluation before expansion, then integrate agent-driven research into existing systems; board 35895aac-1b12-4cf5-b868-516196914789"
fact_checked: 2026-10-30
confidence: medium
---

# Frozen CRSP monthly study design

- Research entry point: [[wrds-equity-research-protocol]].

The design is frozen in `libs/covenant_ml/docs/crsp-monthly-v1.json`. Its codec
and calendar, membership, execution-price and terminal-value validators are
implemented, and so are holdings evaluation and the validation study. The implemented process
boundary and deployment limits are in [[wrds-proposer-isolation]] and
[[wrds-baseline-readiness]]. These are study assumptions, not
profitability findings.[^frozen][^holdings]

| Decision | Rule |
|---|---|
| Question and target | Can lagged return/volume features improve monthly stock selection over fixed momentum? Target each eligible stock's total return from the close of the first trading session of the next month to the corresponding close one month later, minus the eligible universe's equal-weight return. |
| Information and execution | Decide after month-end close using only observations dated through that close; execute at the next month's first trading-session close. Features cannot use that execution session's price or volume. Include distributions and delisting outcomes in targets. No fill at the same close that supplies the signal. |
| Historical universe | At each decision date, common shares with effective share codes 10/11 and exchange codes 1/2/3, absolute closing price at least $5, and at least 252 prior sessions. Rank by trailing 60-session median absolute-price-times-volume; take at most 1,000. Resolve ties by PERMNO. Use effective-dated membership, including later-delisted securities; never today's constituent list. |
| Historical availability | Preserve extraction vintage and source mappings. Effective dates govern membership. A current CRSP history is a revised historical backtest, not a proven as-published archive. Compustat stays outside v1; later financial features require report/publication availability and revision handling, not fiscal-period dates alone. |
| Calendar and missingness | Establish an independent exchange-session calendar. No forward-filled return labels or silent dropping of held positions. Missing required prices, returns or delisting outcomes make the affected evaluation incomplete; log the failure. Eligibility excludes insufficient pre-decision history with counted reasons. |
| Development and final period | Train 2000–2014; tune on 2015–2021 through expanding chronological folds, retaining only windows whose exit is before January 2022 (decisions through October 2021). Seal January 2022–December 2024 decisions as the final period, acquiring through February 2025 for the last exit. Purge training labels whose exit overlaps the next split and exclude the monthly training decision immediately before that split. Warm-up begins January 1998. |
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

[^request]: Board task 35895aac-1b12-4cf5-b868-516196914789, § "OPERATOR", submitted 2026-10-28T16:38:53Z; the study rules are this session's proposed design, introduced in API commit 3358191792a4a655825b5647694fc0430cf7bd33 and calendar-corrected in 78b009119, not quotations or empirical findings.

[^frozen]: `libs/covenant_ml/docs/crsp-monthly-v1.json`, complete frozen settings; `libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py`, `decode_crsp_protocol`; `libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py`, `crsp_windows`, `crsp_member_on`, `require_crsp_execution_price`, `crsp_terminal_value`; `libs/covenant_ml/tests/markets/test_crsp_protocol.py`, independent calendar examples and invalid-evidence refusals. Vendor guide pp. 7, 11–12, 25, 88, 91–92 supplies calendar, date, price and missingness definitions; reinvestment timing is a study assumption, not a vendor guarantee.

[^coverage]: `tools/hpc3/artifacts/research-queue-fresh-20261029/crsp-date-coverage.json`, date_extents, query_sha256 and limits; read-only native job 40544470 on 2026-10-29, extending the three-relation probe 40544368 with stkdelists coverage. No security observations were returned or scored.

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. The substantive monthly CRSP study has not run.
