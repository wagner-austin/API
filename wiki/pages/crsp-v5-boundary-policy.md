---
title: CRSP version-5 execution, delisting-boundary and incomplete-outcome policy
tags: [wrds, crsp, qualification, missingness, protocol]
hubs: [libs]
related: ["[[crsp-outcome-qualification]]", "[[wrds-v3-refused-path-diagnosis]]", "[[wrds-crsp-action-case-evidence]]", "[[crsp-monthly-holdings-validation]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py"
  - "libs/covenant_ml/docs/crsp-monthly.json"
  - "libs/covenant_ml/tests/markets/test_crsp_unsold_exit.py"
  - "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py"
  - "libs/covenant_ml/tests/markets/test_crsp_holdings.py"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json"
  - "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/action-reconciliation.json"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_qualification.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_qualification_run.py"
  - "libs/covenant_ml/scripts/crsp_qualification.py"
  - "libs/covenant_ml/tests/markets/test_crsp_qualification.py"
  - "libs/covenant_ml/tests/markets/test_crsp_qualification_run.py"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "a17f46200e9c5c37d4eae0b7edfc4fc5a7840fa1"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "08da2ef4c43700b587bc2ece90aa9b99dd3d7a64"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py": "529faea3403d889659003b10db690354d1f7376a"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py": "0c6c56f05f23df1a239492e39798f935d47d6fdf"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "4e6afdb5f0fe6d1730185aa31f3ca13a39aad1dc"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py": "b8752083aca89e252059bfef5e39cb29c1bdbb57"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py": "b4cbda6c030daf2a1cbd0c53f3b25c0354e23fc5"
  "libs/covenant_ml/docs/crsp-monthly.json": "4b42124fa38490ee0ff9baceaf7f1e5fa7d5748c"
  "libs/covenant_ml/tests/markets/test_crsp_unsold_exit.py": "a5dea319619c9ca33035e8e3589b371a7ea24684"
  "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py": "560d6591acd3971a387552e3019ad8d6eb89d509"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "79e8c1a57c45e9014a3902d276dafb439efc69d7"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json": "5fca13ce388bd45adc74f8e52efa5d7d42af7775"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/action-reconciliation.json": "b39cb05dfa388c8f0285508518118fe77325ec78"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_qualification.py": "e243ecc4218c8024ed3de16136a18ea71061061c"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_qualification_run.py": "98f92e84a790249f06fcca59307cb0ade6dc3adf"
  "libs/covenant_ml/scripts/crsp_qualification.py": "7fd260c648cbeedc984cf639507dd4bf20403b62"
  "libs/covenant_ml/tests/markets/test_crsp_qualification.py": "e606cab000d562a1660868da40870d2bf2e91fc8"
  "libs/covenant_ml/tests/markets/test_crsp_qualification_run.py": "11e054f12ee9f4935ded10d976f660e4045a4a61"
provenance:
  - "Primary capture: tech-wiki/sources/crsp-ciz-guide-20260731.txt; SHA-256 bb4792deb927b972a8bef1a656ba1fc5442b80783e10eecb3e633d4243334b2e, re-hashed 2026-11-03; pp. 11-13 (StkDelists, StkDistributions), 88 (PC price flags), 90-92 (RD and RM flags)."
  - "Board task f963ee0b-7382-4d7a-a1db-9a7b4d5467a3; review of the version-4 suspension amendment is board post 43ba2e2b-c1d0-4a3c-ba3e-4fb75fc4123c."
fact_checked: 2026-11-03
confidence: medium
---

# CRSP version-5 execution, delisting-boundary and incomplete-outcome policy

This is the prospective protocol decision for every class of the 27 version-3
first refusals in [[wrds-v3-refused-path-diagnosis]]. Version 5 changes two
declared accounting rules and keeps the version-4 held-return rule. It reads
the same extraction columns and dates, so the existing encrypted extraction
can be revised by metadata alone. **Nothing here is installed or measured on
licensed data**: the controller and image remain the version-3 freeze, no
licensed version-5 revision exists, and the native full-panel scan that would
count the remaining refusals has not run.[^protocol][^revision]

## Decision by diagnosed class

The counts are version-3 first refusals. A path can hide a second problem
behind its first, so the "resolved" column states which rule removes the
first refusal, not that the path is complete. Only the native version-5 scan
can say that.[^cases]

| Class | Paths (ordinals) | Version-5 decision | Kind |
|---|---|---|---|
| Closed SU span | 1 (12343) | Exact P9 closure accepted by the version-4 held rule | Extension of an existing rule to a second unpriced flag; independently reviewed |
| BA new entry | 3 (10141, 106500, 136773) | Unfilled order kept as cash | Changed execution assumption (rule 1) |
| NT entry | 1 (84356) | Unfilled order kept as cash | Changed execution assumption (rule 1) |
| BA old-position exit | 3 (9165, 105499, 135749) | Delisting on an unsold exit ends the position with its terminal value; value dated 3, 1 and 1 days after exit | Changed boundary (rule 2) and label timing (rule 3) |
| Value dated after exit | 5 (18863, 27156, 34781, 44820, 187445) | Labels dated at the valuation; receivable rule unchanged | Changed label timing (rule 3) |
| NT held | 5 (54734, 62905, 83373, 84496, 92184) | Refused | No value exists |
| MV | 6 (10402, 59635 held; 134760, 173979, 179591, 191467 terminal) | Refused | Value flagged incomplete |
| DG | 1 (133712) | Refused | Delisting return missing |
| Open SU or MP tail | 2 (23610, 169697) | Refused | No closing span by exit |

So 13 first refusals have an explicit rule and 14 keep refusing. The 14
include 191467, a validation path.[^cases][^reconciliation]

## Rule 1: an entry without a closing trade is an unfilled order

The CIZ price flag distinguishes a closing trade (TR) from a bid/ask average
(BA), suspended (SU), halted (HA), not tracked (NT) and missing price (MP).[^pc]
Version 5 fills a new purchase only at a TR record. A present entry record with
any other flag is an unfilled order: its equal allocation stays in
zero-interest cash, and the security keeps a cash multiplier of one in both the
label benchmark and holdings. This generalizes version 3's confirmed pre-entry
delisting rule, which already did the same for one case.[^entry]

- **Not changed:** the decision-time universe, features and selection. No
  security is removed and no allocation is redistributed.
- **Still refused:** an absent entry record (no evidence), a TR record
  without a finite positive price, a contradictory terminal entry row, and a
  position already held that has no closing trade at a rebalance (it can be
  neither sold nor assumed held).
- **Direction of the assumption:** an investor might have traded near a quote;
  treating every non-TR entry as untradeable is the conservative execution
  assumption, and it never values the security.

## Rule 2: a delisting on an exit close with no trade to sell into

DelistingDt is the date of the security's last price on its exchange, DelRet
compares the delisting amount with the price on that last day, and the
delisting return is stored on the following trading day.[^delist] Version 3
placed an event on the exit close in the next window, which is right when the
exit record is a closing trade: the position is sold into it, and that is
unchanged. When the exit record is present but is no closing trade (the three
BA exits), there is nothing to sell into. Version 5 ends that position with
its terminal value: growth compounds through the exit close and then applies
one plus DelRet once, the same chain version 3 already used for every
delisting inside a window. A new order at that same quote in the next window
is unfilled under rule 1, so there is no second claim.[^exit]

## Rule 3: a label is available when its last value is dated

The receivable rule already kept a terminal value out of reinvestment until a
rebalance strictly after its valuation date. Version 3 also refused any value
dated after the exit, because every label was dated at the exit. Version 5
removes that refusal and dates **every** label of a decision at the latest of
the exit and every member's terminal valuation date, since every label
depends on the full-universe benchmark. The existing purge then drops any
training or stopping label not available before the fold boundary.[^labels]

**One convention to review explicitly.** The holdings evaluation still
attributes the terminal value to the window in which the delisting happened:
that month's end value and net return include the receivable at its later-dated
value. CRSP likewise stores a delisting return on DelDlyDt, the session after
the delisting date, rather than on its amount date, and its monthly series
flags whether a period's compounded return includes it.[^dlydel][^mthdel] No
decision reads the value early: the receivable is not reinvestable, and no
label using it is available, before its date.[^labels] But it does report a
monthly return containing a value not yet known at that month's exit. The
alternative, attributing it to the month in which it is dated, changes the
period returns the paired bootstrap compares; this policy does not adopt it.

## Leakage and boundary protections

- **Features:** decision-time only, unchanged.
- **Universe:** never filtered by execution or outcome availability.
- **Labels:** dated as in rule 3 and purged per fold; the final fit uses the
  same purge.
- **Final period:** extraction keeps a delisting only when its latest valuation
  date is on or before the partition's observation end, and the partition
  check re-verifies every dated field after decryption. A train-validation label
  therefore never uses a value dated in the final period.[^partition]
- **Partition-boundary omission (named, not repaired):** the same predicate
  drops a train-validation delisting whose valuation date crosses the boundary,
  while the daily series keeps its delisting return. CIZ stores that return
  on DelDlyDt, the session after the delisting date, flagged DlyDelFlg
  Y.[^dlydel] So the flag lies on a held session or, for an unsold exit, on
  the session after the exit. The window path refuses such a path as
  `delisting return without event` (CRSP-009) and never reads the value.
  Before this was named, the same paths refused under a span or exit key that
  hid the cause.[^partition][^eventless]
- **Who can meet it:** only a known decision, whose window exits before the
  2022-01-01 final period, with a member's event valued after 2021-12-31. The
  version-3 full-panel diagnosis ran under the same extraction predicate, and
  none of its 27 refusals has this shape: its five held-path refusals that end
  on a DA or DP record each have their event inside the window. Version 5 newly assigns unsold-exit
  events to the earlier window, so the native scan is what measures it.
  Whether such a decision should leave the known panel, because its labels
  are not knowable by the boundary, is a design question for review and is
  not decided here.[^cases]

## Why the 14 remaining refusals stay refusals

- **NT held (5):** no price and no return through exit. The action join found
  no delisting or distribution in the window or on exit, and the source
  comparison found the rows unchanged.[^reconciliation]
- **MV (6):** the vendor flags the corporate-action value missing (RM flag
  MV).[^rm] Each case has at least one distribution with a null amount; null is
  not zero. A number is present but the guide marks it incomplete.
- **DG (1):** the delisting price is more than ten periods from the delisting
  date and DelRet is null.[^rm]
- **Open tails (2):** no closing span by exit.

No imputation is adopted: zero, carry-forward and a literature delisting
haircut would each be an invented outcome, and the resolvers refuse instead.[^return-path]
Dropping securities or decision
months would select on future outcome availability. **So if the native
version-5 scan confirms any remaining refusal, the full-universe 2000-2020
target is not qualified under version 5, and the record must say so.**

Two routes could reduce the 14, and each needs its own review before use:

1. **Bounded MV source correction.** A distribution that delivers another
   CRSP security names it (DisPERMNO).[^dist] Its value could come from that
   security's own record on the ex-date. The capture deliberately did not
   expand to linked securities, so this needs a new licensed query.
2. **A declared incomplete-outcome rule with sensitivity bounds** for the
   cases no source can value (NT, DG, open tails). That would be a changed
   target, reported beside the unchanged one, not a repair.

## The qualification instrument

The native scan uses the production evaluator, not the archived version-3
audit script. `qualify_crsp_panel` walks the known panel in label order and
asks `crsp_resolve_window_path` for every path; that resolver is the one
`crsp_window_path` raises from, so the scan and the study cannot disagree
about which paths refuse. Each refused path keeps its one-based ordinal (the
same convention as the version-3 diagnosis), decision index, period, counting
key and entry and exit record digests, so the result joins to the 27 diagnosed
cases without publishing an identifier, date, price or return. The report also
counts the decisions, and the rows in them, that any refusal leaves
unlabelled. The entrypoint pins design, bundle, entrypoint source and image
before acquiring only the train-validation key, and publishes a shared run
record.[^instrument]

## Not established yet

- No licensed version-5 metadata revision, controller or image freeze.
- No native full-panel version-5 scan, so no measured remaining-refusal count.
- No candidate fit, queue admission or final release. The stopped version-3
  study keeps its one failed and fourteen unsubmitted attempts.
- The Bermudan Asian options study (board task 5bec93df) does not use the
  holdings path that rules 1 to 3 change. It shares the held-return resolver
  and the protocol decoder, so its version-3 run reproduces only at its own
  image.

[^protocol]: `libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py`, `decode_crsp_protocol`, version 5, `entry_rule` closing-trade-or-unfilled-cash, `held_return_rule` exact-unpriced-held-spans, `terminal_rule` unsold-exit-event-label-at-valuation; `libs/covenant_ml/docs/crsp-monthly.json`, the committed design with unchanged columns and dates.
[^revision]: `libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py`, `_inputs`, allowed changes now include `terminal_rule`; every extraction setting must stay identical.
[^cases]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json`, `results[0].refused_cases`: signature, entry and exit price flags, terminal-row day offsets and return-missing flags per path ordinal.
[^reconciliation]: `tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/action-reconciliation.json`, per-case event counts, null distribution amounts and delisting amount offsets from exit (3, 1, 1 for the BA exits).
[^pc]: CIZ guide p. 88, PC flag table: SU Suspended, BA Bid Ask Average, MP Missing Price, HA Halted, TR Closing Trade Price, NT Not Tracked.
[^entry]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_entry` and `evaluate_crsp_holdings`; `libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py`, `test_entry_without_a_closing_trade_is_an_unfilled_order_kept_in_cash`, `test_ordinary_exit_without_a_closing_trade_refuses_the_label_path` and `test_closing_trade_flag_without_a_positive_price_still_refuses_entry`.
[^delist]: CIZ guide pp. 11-12, StkDelists: DelistingDt, DelRet, DelAmtDt and DelDlyDt definitions.
[^exit]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_rows` and `crsp_resolve_window_path`; `libs/covenant_ml/tests/markets/test_crsp_unsold_exit.py`, `test_delisting_on_an_unsold_exit_is_counted_once_and_locked_past_the_exit`, `test_events_inside_and_on_an_unsold_exit_are_ambiguous` and `test_real_worker_publishes_the_late_receivable_and_the_unfilled_reentry`.
[^labels]: `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_study_rows` (`known`) and `crsp_fold` (`purge_label_overlap`); `libs/covenant_ml/tests/markets/test_crsp_unsold_exit.py`, `test_real_extract_dates_every_label_of_the_decision_by_its_latest_value`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_value_dated_after_exit_stays_locked_until_the_following_rebalance`.
[^partition]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `crsp_relation_sql`, delisting `valuation` predicate; `libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py`, `check_crsp_partition`, dated-field checks.
[^return-path]: `libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py`, `crsp_resolve_held`: an absent row or a row whose delisting flag is not N refuses the span.
[^rm]: CIZ guide pp. 91-92, RM flag table: DG delisting price more than 10 periods from the delisting date; MV missing corporate action value.
[^instrument]: `libs/covenant_ml/src/covenant_ml/markets/crsp_qualification.py`, `qualify_crsp_panel` and `crsp_qualification_counts`; `libs/covenant_ml/src/covenant_ml/markets/crsp_qualification_run.py`, `execute_crsp_qualification`; `libs/covenant_ml/scripts/crsp_qualification.py`; `libs/covenant_ml/tests/markets/test_crsp_qualification.py` and `libs/covenant_ml/tests/markets/test_crsp_qualification_run.py`, an open suspension and a DG delisting counted at independently located ordinals, and a real encrypted subprocess run without the final key. Fleet job 368a7a90-73ec-4a33-9359-3e19c185aefb passed make check for libs/covenant_ml at API f3de1bef42c79c92b5c87717d3c1100fc303d1d5, the published tip carrying the eventless-delisting refusal, on serendipity at 2026-11-03T08:06:41Z: guard, mypy and 5,646 tests, 100.00% statements and branches. This is synthetic evidence for the code, not a licensed qualification.
[^dlydel]: CIZ guide p. 12, StkDelists: DelDlyDt is the date the return is stored in StkDlySecurityData, by convention the trading date immediately after the delisting date; p. 17, StkDlySecurityPrimaryData: DlyDelFlg Y means a delisting return is present.
[^eventless]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_eventless_delisting`, `_unsold` and `crsp_resolve_window_path`; `libs/covenant_ml/tests/markets/test_crsp_qualification.py`, `test_a_delisting_the_partition_omits_is_named_rather_than_valued` (a real extract without security 8's event, one refusal at an independently located ordinal) and the held and unsold-exit cases of `test_each_refusal_has_its_counting_key_and_message`; `libs/covenant_ml/tests/markets/test_crsp_unsold_exit.py`, `test_a_sold_exit_needs_no_event_for_a_delisting_flagged_after_it`.
[^mthdel]: CIZ guide p. 20, StkMthSecurityData: MthDelFlg indicates if the returns from delists are included in the compounded period return.
[^dist]: CIZ guide p. 13, StkDistributions: DisPERMNO, the PERMNO of a security received in a spinoff, exchange, merger or other distribution.
