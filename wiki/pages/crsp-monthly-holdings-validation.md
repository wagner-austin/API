---
title: CRSP monthly holdings and chronological validation
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "libs/covenant_ml/tests/markets/test_crsp_unfilled.py"
  - "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py"
  - "libs/covenant_ml/tests/markets/test_crsp_holdings.py"
  - "libs/covenant_ml/tests/markets/test_crsp_study.py"
  - "libs/covenant_ml/tests/markets/test_research_trial.py"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "28faea8bff58470c1a8b73230949f45dba82be9d"
  "libs/covenant_ml/tests/markets/test_crsp_unfilled.py": "7f848a9ca758b48c41f3dca9c891bc167b654fb0"
  "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py": "360c80d81411228346f695f26427b80ce15f6276"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "64bfcdcc9053f66511e8b6398de7b8bd44de19d4"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "13031c17a0e67e2f4fc75d2486d2b6689092536a"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "824b9bad6e08e7d4955dad45f2b8a48bfa4ab6a9"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "325bc527818cfa1c6464541208c1cb15baa5967f"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "2778e68c44c20c724ccd8403312888f168499350"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "b797984c6422a9c99cfd4db6c09ef92a40c755f9"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "50dcf2341e759b4805b719757302d34165f68fa7"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
fact_checked: 2026-11-01
confidence: medium
---

# CRSP monthly holdings and chronological validation

- Research entry point: [[wrds-equity-research-protocol]].

Monthly holdings start from one unit of cash. At each rebalance close the
investable value, cash and held positions plus terminal values dated strictly
before that close, is split equally across the decision-time selection.
Version 2 solves `slots × x + rate × turnover(x) = investable` on the linear
segments between prior position values. This charges actual buys and sells
against post-cost targets; the earlier pre-fee-target approximation is removed.
Positions compound their own ordinary daily returns, checked against
the declared predecessor session. A position delisted inside a window stops at
its delisting date and becomes a receivable worth its `stkdelists` multiplier,
applied once and never combined with the daily delisting-flagged row; it is
reinvested only at a rebalance strictly after its valuation date. The last
window pays the cost of liquidating its positions. A missing or non-ordinary
held return, an unverifiable fill, an unresolved owned terminal event or a value
dated after the scored exit refuses the whole evaluation. Hand-calculated cases
check two-month drift, turnover and liquidation and a three-month delisting
whose value waits one month before reinvestment.[^holdings]

Label construction uses the same order outcome as holdings. Exactly one event
at decision close, before entry, plus an entry-day terminal row with `DD`
duration and a non-`TR` price flag confirms a rejected purchase. Its equal
allocation remains zero-interest cash without replacement or redistribution;
its gross label multiplier is one and remains in the universe benchmark.
Missing rows/events and contradictory trade flags still refuse evaluation.
The monthly payload retains unfilled selections and cash weights alongside
held positions; weights describe the end value before final liquidation.[^unfilled]

Executed entry and ordinary exit must be closing trades. The terminal interval is
`[entry, exit)`: a delisting on entry is included, while a last ordinary trade
on exit can be sold before the subsequent terminal outcome. Tests cover these
boundaries and the pre-entry delisting found by the licensed baseline, without
filtering decision-time selection using later outcomes. The encrypted synthetic
study runs the real trainers and emits the rejected order and cash weight,
checks that a prior holding's receivable survives exactly once until its release,
then verifies private plaintext removal.[^execution][^unfilled]

[^unfilled]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_entry_filled` and `_rebalance_notional`; `libs/covenant_ml/tests/markets/test_crsp_unfilled.py`, hand-calculated mixed/all-rejected portfolios, contradictory evidence, held-loss refusals, encrypted study and old protocol rejection; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `_encode_series`.

[^execution]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_row`, `crsp_window_path` and `_require_fill`; `libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py`, terminal-boundary arithmetic and verified-extract cash labels with unchanged selection.

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

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. This was synthetic integration evidence; the later licensed attempt stopped before fitting, as recorded in [[wrds-baseline-20261101]].

[^boundary]: `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation`; `libs/covenant_ml/tests/markets/test_crsp_study.py`, `test_holdout_boundary_excludes_equal_and_later_exits` and `test_changed_holdout_outcomes_cannot_change_validation`; `libs/covenant_ml/tests/markets/test_research_trial.py`, `test_real_trainers_emit_the_admitted_shared_monthly_record`. Fleet job 9152ad13-99d3-4c5b-9686-6082ed67f8fb passed full make check at API cbe6d15998b6d8443ac89f46a8e72d367258fcf8: 4767 tests, 100% statement/branch coverage, strict typing and guards clean, 2026-10-30T02:34:54Z. Board task b1d24135-ad6d-4d32-a0d4-cef6abc83813 tracks independent review.
