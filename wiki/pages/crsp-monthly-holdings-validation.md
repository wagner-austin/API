---
title: CRSP monthly holdings and chronological validation
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "libs/covenant_ml/tests/markets/test_crsp_suspension_process.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py"
  - "libs/covenant_ml/tests/markets/test_crsp_return_path.py"
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
  "libs/covenant_ml/tests/markets/test_crsp_suspension_process.py": "fd513a076cdbc9475e4da1b7fe0c83459114f232"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "565c8ddad5836aaac5eb07d372cb3baaea746b15"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py": "f4cdd6c1ba7d1d86991bfdb2cb22edf362619c71"
  "libs/covenant_ml/tests/markets/test_crsp_return_path.py": "840edba442fe38ea109c34482522cccf422930a1"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "28faea8bff58470c1a8b73230949f45dba82be9d"
  "libs/covenant_ml/tests/markets/test_crsp_unfilled.py": "f315bb6ddce5fcc988221560818510d06e3da704"
  "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py": "560d6591acd3971a387552e3019ad8d6eb89d509"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "77b54879136c9ef85c8cd2d0ef9db6172ed59dd4"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "3b532fbe7a7b0102c89c7c62b6e2a35c6ba3bc5b"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "a6ad643fc3641382e127370155fddb7c713469f8"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "325bc527818cfa1c6464541208c1cb15baa5967f"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "2311912957c18245f3d5c0630c96c301fd13c012"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "ac6baa65e78f71d8bed5728e162e05773551b81d"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "50dcf2341e759b4805b719757302d34165f68fa7"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
fact_checked: 2026-11-03
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
Positions compound their own evidenced return intervals. Ordinary returns are
checked against the declared predecessor session. Prospective version 5, whose
held-return rule is version 4's, permits
explicit unpriced MP or SU rows with MP/MR returns followed by one available NA/P1-P9 return
whose previous-price date and duration cover exactly the intervening declared
sessions. No intermediate close is assigned a value. Missing rows, other
missingness, inconsistent spans and an unresolved tail still refuse the whole
evaluation; the same path supplies labels and holdings.[^spans]

The suspension extension changes the held-interval predicate, not entry fills,
exit trades, features or terminal timing. Its independent arithmetic tests cover
two through ten sessions, mixed MP/SU gaps, wrong predecessor/duration flags,
unknown corporate-action returns and an open tail. The real encrypted worker
test runs a synthetic nine-session suspension closed by P9 and an unresolved
suspension through exit; only the closed case can publish a score. Both require
private plaintext removal without a final key. These tests do not establish a
licensed outcome or native cluster qualification.[^suspension]

[^suspension]: `libs/covenant_ml/tests/markets/test_crsp_return_path.py`, `test_vendor_duration_must_cover_exactly_two_through_ten_sessions`, `test_mixed_suspension_and_missing_price_span_has_only_one_observed_value` and `test_real_extract_labels_use_the_same_span_without_future_universe_filtering`; `libs/covenant_ml/tests/markets/test_crsp_suspension_process.py`, `test_real_worker_scores_closed_suspension_or_refuses_and_cleans_an_open_tail`.

A position delisted inside a window, or on an exit close whose record is no
closing trade, stops at its delisting date and becomes a receivable worth its
`stkdelists` multiplier, applied once and never combined with the daily
delisting-flagged row; it is reinvested only at a rebalance strictly after its
valuation date, which version 5 allows to follow the exit. The last window pays
the cost of liquidating its positions. An unresolved held interval, an
unverifiable fill or an unresolved owned terminal event refuses the whole
evaluation. Hand-calculated cases check two-month drift, turnover and
liquidation, a three-month delisting whose value waits one month before
reinvestment, and the same wait for values dated one and two days after the
exit; [[crsp-v5-boundary-policy]] gives the reasons.[^holdings]

Current source shares `crsp_resolve_held` between consumers: it returns the
complete resolved spans or a refusal with no partial spans. Stock compounding
uses those spans through `crsp_held_spans`; an unresolved interval still raises.
Terminal valuation similarly uses `crsp_resolve_terminal`, with
`crsp_terminal_value` refusing unresolved records. These shared owners do not
establish which source is installed in an older frozen image; the exact study
runtime and observed outcome remain in [[wrds-baseline-readiness]].[^spans]

[^spans]: `libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py`, `crsp_resolve_held`, `crsp_held_spans`, `crsp_compound_held` and `_unpriced_held`; `libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py`, `crsp_return_span`, `crsp_resolve_terminal` and `crsp_terminal_value`; `libs/covenant_ml/tests/markets/test_crsp_return_path.py`, independent span, wealth, cost, label and boundary oracles, including first-refusal resolution without partial spans; `libs/covenant_ml/tests/markets/test_crsp_unfilled.py`, encrypted evaluator integration with a missing-price span. Vendor meanings remain in the tech wiki's `crsp-ciz-daily-return-quality`; this permission is a versioned study assumption, not permission to fill or discard unknown observations.

Label construction uses the same order outcome as holdings. Under version 5 a
new purchase fills only at a `TR` entry record; a present record with any other
price flag is an unfilled order. A terminal entry row is unfilled only when
exactly one event at decision close precedes entry, with `DD` duration and a
non-`TR` price flag. The equal allocation remains zero-interest cash without
replacement or redistribution; its gross label multiplier is one and remains in
the universe benchmark. Absent rows, contradictory terminal rows, a `TR` flag
without a positive price, and a held position with no trade at a rebalance
still refuse evaluation.
The monthly payload retains unfilled selections and cash weights alongside
held positions; weights describe the end value before final liquidation.[^unfilled]

Executed entry and ordinary exit must be closing trades. Current source reports
an absent or null execution price at a `TR` record as `CRSP-004`; malformed
numeric values still raise a type error. The boundary tests run quote,
suspended, halted, untracked and missing-price records through the real label
path: at an ordinary exit each refuses, at entry each is an unfilled order.
None of this updates a frozen image.[^execution]
The terminal interval is
`[entry, exit)`, plus the exit close itself when its record is no closing trade:
a delisting on entry is included, a last ordinary trade on exit is sold before
the subsequent terminal outcome, and a delisting on a quote-only exit ends the
old position with its terminal value. Tests cover these boundaries and the
pre-entry delisting found by the licensed baseline, without filtering
decision-time selection using later outcomes. The encrypted synthetic
study runs the real trainers and emits the rejected order and cash weight,
checks that a prior holding's receivable survives exactly once until its release,
then verifies private plaintext removal.[^execution][^unfilled]

[^unfilled]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_entry_filled` and `_rebalance_notional`; `libs/covenant_ml/tests/markets/test_crsp_unfilled.py`, hand-calculated mixed/all-rejected portfolios, contradictory evidence, held-loss refusals, encrypted study and old protocol rejection; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `_encode_series`.

[^execution]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_row`, `crsp_window_path` and `_require_fill`; `libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py`, `require_crsp_execution_price`; `libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py`, terminal-boundary arithmetic, missing entry/exit prices and verified-extract cash labels with unchanged selection.

The validation study builds decisions only up to the day before the final
period. Labels are whether a security's window multiplier, terminal value
included, beats the eligible universe's equal-weight multiplier. Every label of
a decision is available at the later of the exit close and the latest member's
terminal valuation date, because each depends on the whole universe. Validation decisions are scored in expanding folds of declared
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
