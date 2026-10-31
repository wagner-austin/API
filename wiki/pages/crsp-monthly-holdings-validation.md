---
title: CRSP monthly holdings and chronological validation
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py"
  - "libs/covenant_ml/tests/markets/test_crsp_holdings.py"
  - "libs/covenant_ml/tests/markets/test_crsp_study.py"
  - "libs/covenant_ml/tests/markets/test_research_trial.py"
source_git_blobs:
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "9686350a06789e4f05d869da140948b6a56d0d19"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "c63fd95cfe672e9e198cd1b5cab895e97466dc77"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "47eaddf94090471d7dc84e188829de4d8cfaf1dd"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "b5081a606ba9f8d21673ef75c9a7da5a197c6dde"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "b797984c6422a9c99cfd4db6c09ef92a40c755f9"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "448c595406899be546fe79b550df921a6761dcd8"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
fact_checked: 2026-10-30
confidence: medium
---

# CRSP monthly holdings and chronological validation

- Research entry point: [[wrds-equity-research-protocol]].

Monthly holdings start from one unit of cash. At each rebalance close the
investable value, held positions plus terminal values dated strictly before
that close, is split equally across the selection; traded notional is measured
against the drifted pre-trade positions and its frozen cost is paid from the
portfolio. Positions compound their own ordinary daily returns, checked against
the declared predecessor session. A position delisted inside a window stops at
its delisting date and becomes a receivable worth its `stkdelists` multiplier,
applied once and never combined with the daily delisting-flagged row; it is
reinvested only at a rebalance strictly after its valuation date. The last
window pays the cost of liquidating its positions. A missing or non-ordinary
held return, a missing or non-`TR` fill, an unresolved terminal event or a value
dated after the scored exit refuses the whole evaluation. Hand-calculated cases
check two-month drift, turnover and liquidation and a three-month delisting
whose value waits one month before reinvestment.[^holdings]

The validation study builds decisions only up to the day before the final
period. Labels are whether a security's window multiplier, terminal value
included, beats the eligible universe's equal-weight multiplier, available at
the exit close. Validation decisions are scored in expanding folds of declared
length; each fold fits the existing baseline trainers on earlier decisions,
uses the declared stopping months for early stopping, and drops every label not
available before the next boundary. Candidates take the top
`ceil(selection_fraction × eligible)` by predicted probability; fixed momentum
and the equal-weight universe run through the same accounting at all three
frozen costs, and every arm is paired with momentum by the frozen moving-block
bootstrap. On a seeded four-year fixture the tests check, by independent index
arithmetic, that each fold drops exactly the decision before each boundary, and
check independently selected momentum portfolios, raw-row label oracles and the
frozen-seed interval. Using binary labels because the trainers are classifiers is
a declared implementation choice, not part of the frozen design. Before labels
are attached, windows whose exits fall on or after `holdout_start` are omitted.
This corrects the earlier validation tail that consumed final-period outcomes;
the committed design now ends validation decisions in October 2021 and exits in
December 2021. Tests exclude an exit equal to the boundary and verify that
changing later returns and prices leaves validation output unchanged. This
temporal exclusion does not establish process or credential isolation.[^holdings][^boundary]

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. The substantive monthly CRSP study has not run.

[^boundary]: `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation`; `libs/covenant_ml/tests/markets/test_crsp_study.py`, `test_holdout_boundary_excludes_equal_and_later_exits` and `test_changed_holdout_outcomes_cannot_change_validation`; `libs/covenant_ml/tests/markets/test_research_trial.py`, `test_real_trainers_emit_the_admitted_shared_monthly_record`. Fleet job 9152ad13-99d3-4c5b-9686-6082ed67f8fb passed full make check at API cbe6d15998b6d8443ac89f46a8e72d367258fcf8: 4767 tests, 100% statement/branch coverage, strict typing and guards clean, 2026-10-30T02:34:54Z. Board task b1d24135-ad6d-4d32-a0d4-cef6abc83813 tracks independent review.
