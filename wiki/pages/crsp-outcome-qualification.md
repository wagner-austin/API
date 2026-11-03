---
title: CRSP outcome qualification before a replacement study
tags: [wrds, crsp, qualification, missingness]
hubs: [libs]
related: ["[[wrds-v3-refused-path-diagnosis]]", "[[wrds-baseline-readiness]]", "[[crsp-monthly-holdings-validation]]", "[[crsp-v5-boundary-policy]]"]
source_paths:
  - "libs/covenant_ml/tests/markets/test_crsp_unfilled.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py"
  - tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/action-reconciliation.json
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py"
  - "libs/covenant_ml/tests/markets/test_crsp_queries.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/tests/markets/test_crsp_study.py"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py"
  - "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/protocol.json"
  - "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/benchmark-dependence.json"
  - "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/source-check-audits.json"
  - "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/corporate-action-catalog.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-v5-qualification-20261103/evidence.json"
source_git_blobs:
  "libs/covenant_ml/tests/markets/test_crsp_unfilled.py": "752574db15efc98e09420383a36345b98946bc66"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py": "529faea3403d889659003b10db690354d1f7376a"
  tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/action-reconciliation.json: b39cb05dfa388c8f0285508518118fe77325ec78
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "21b5d8b794f67352876ab9452d9d30858af533a3"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py": "b4cbda6c030daf2a1cbd0c53f3b25c0354e23fc5"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "690e01a7b1b6a0b1be22d35d376e4715094886e0"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "4e6afdb5f0fe6d1730185aa31f3ca13a39aad1dc"
  "libs/covenant_ml/tests/markets/test_crsp_queries.py": "cca05554450299e2c0e111c031dfd1f88a930ac4"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py": "b8752083aca89e252059bfef5e39cb29c1bdbb57"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py": "1a5c4f11e1e90bf50d29d800eeb03fc941721cc6"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "08da2ef4c43700b587bc2ece90aa9b99dd3d7a64"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "ac6baa65e78f71d8bed5728e162e05773551b81d"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json": "5fca13ce388bd45adc74f8e52efa5d7d42af7775"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py": "c9c37944ab5ca06f18953e9f9e3d4d21cb0e8438"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/protocol.json": "a095650070b230379e759563df3c20b3c2476ded"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/benchmark-dependence.json": "b2994cf9db46aff4a4714b987bd4f5c0729dd5b2"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/source-check-audits.json": "7c39757f5c5a238d9dabf24f0531f201adeba639"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/corporate-action-catalog.json": "cba4424c62478199f45a23b26df2ddd4b992c2ec"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-v5-qualification-20261103/evidence.json": "85b7e1763c9817abd7a82f29ffa93fdf27a89884"
provenance:
  - "Primary capture: tech-wiki/sources/crsp-ciz-guide-20260731.txt; SHA-256 bb4792deb927b972a8bef1a656ba1fc5442b80783e10eecb3e633d4243334b2e; corpus 0264d5cc-4dec-4715-b17b-45f3c8e5fbec; reread November 2."
  - "Board task f963ee0b-7382-4d7a-a1db-9a7b4d5467a3; this is qualification analysis, not an approved version-4 or version-5 freeze."
fact_checked: 2026-11-03
confidence: medium
---

# CRSP outcome qualification before a replacement study

The version-3 scan diagnosed every first refusal but did not identify complete
outcomes for every path. Its 27 refusals remain the starting evidence, not a
list of securities to remove. [[wrds-v3-refused-path-diagnosis]] retains the
ordinals, flags and native evidence without exposing licensed observations.[^cases]

## Source-backed distinctions

CRSP distinguishes closing trades from bid/ask averages, suspension from
not-tracked prices, and available returns from missing corporate-action value.
DelDlyDt is a storage convention; DelAmtDt dates a valuation, not observed
cash receipt. A finite number or a later date cannot erase these distinctions.[^vendor]

| Diagnosed class | Count | Qualification requirement |
|---|---:|---|
| BA on a rebalance delisting | 6 | Preserve three old-position exits separately from three new-order entries. A quote cannot satisfy a TR fill. A terminal boundary amendment must account for the old claim once and separately specify rejection of a new order. Version 5 proposes both (rules 1 and 2 of [[crsp-v5-boundary-policy]]). |
| NT | 6 | One entry and five held paths lack tracked evidence. An unfilled-new-order assumption supplies no value for an already owned security. Retain refusal unless bounded source review supplies the missing outcome. Version 5 treats the NT entry as an unfilled order and keeps the five held paths refusing. |
| MV | 6 | Two held paths and four terminal paths contain numbers flagged incomplete. Require complete corporate-action evidence; neither a numeric value nor monthly aggregation establishes completeness. |
| Valuation after exit | 5 | Four are one calendar day late and one is two days late. Later-known terminal value must not be booked as known wealth at the earlier exit. Extending maturity changes the target and split requirements. Version 5 dates the labels at the valuation and keeps the value unreinvestable until then; it still attributes the value to the delisting month's return, a convention [[crsp-v5-boundary-policy]] states for review. |
| SU | 2 | One exactly linked P9 span closes nine suspended sessions; the prospective version-4 resolver, kept in version 5, accepts that flag pattern, subject to exact span closure. Independent review passed (board post 43ba2e2b); licensed qualification remains pending. The other suspension remains open at exit and has no complete span. |
| DG | 1 | The return is missing and valuation is 69 calendar days after the event. The current flag-first resolver improves refusal reporting; it does not manufacture a return. |
| Open MP tail | 1 | No closing span is present by exit. Carrying the last mark or assigning zero would introduce an economic assumption. |

Counts and boundaries come from the frozen native diagnostic. The requirements
are engineering interpretations for review, not vendor-imposed investment
rules or permission to change the stopped study.[^cases][^owners][^vendor]

## An incomplete member affects the whole label benchmark

The target is relative to the full decision-time universe's mean multiplier.
An unknown member therefore leaves that benchmark unverified, even for members
with complete individual paths. The count of 27 refused paths is not a count
of all affected labels. Dropping those rows and recomputing the mean changes
the target; assigning the old labels to the retained rows does not repair it.
The equal-weight comparison arm also holds every eligible member, so a small
top-ranked portfolio cannot make the required universe comparison complete.[^labels]

The synthetic regression keeps one stock's multiplier at 1.125 and all
decision-time features unchanged. Changing a different member's multiplier
from 1.25 to 4 changes the first stock's label from one to zero. Removing that
peer's held observation preserves the decision-time universe and causes label
construction to refuse. This is an evaluator contract check, not a measured
count of affected months in the licensed dataset.[^labels]

The frozen metadata permits a separate derivation: 262 decisions contain
262,000 paths, and each decision is capped at 1,000 members. Every decision
therefore has exactly 1,000 members. The audit records consecutive one-based
path ordinals, so `(ordinal - 1) // 1000` identifies its zero-based decision.
The 27 refusals occupy 23 training and two validation decisions. Thus 25,000
rows depend on an unverified monthly benchmark. This is not a count of wrong
or flipped labels, selected holdings, or realized losses; no new data was
decrypted to derive it.[^dependence]

## Qualification design

The bounded suspension amendment extends the existing held-return owner as
`exact-unpriced-held-spans`, introduced in version 4 and kept in version 5;
version-3 requests and encrypted source metadata cannot silently run under that
codec. It assigns no intermediate prices and, by itself, changes no execution or
terminal rule. An independent review (board post 43ba2e2b) found it preserves
interval arithmetic, assigns no intermediate value, keeps execution boundaries
and adds no future-completeness selection. Synthetic
regressions and the encrypted real-process test are described in
[[crsp-monthly-holdings-validation]]. This is a reviewable implementation,
not a replacement freeze or proof that the original 27 refusals are repaired.[^suspension]

[[crsp-v5-boundary-policy]] records the prospective version-5 decision for
every class: an entry without a closing trade is an unfilled cash order, a
delisting on an exit close with no trade ends the old position with its
terminal value, and a value dated after the exit dates every label of its
decision. That gives 13 first refusals an explicit rule and keeps 14 (NT held,
MV, DG and both open tails) refusing. An independent peer review accepted it,
and the native full-panel scan under a separate version-5 runtime refused
exactly those 14 paths, with the same record digests as the version-3
diagnosis, in two runs that agree. The original full-universe target is
therefore not qualified under version 5.[^v5]

[^v5]: `tools/hpc3/artifacts/wrds-v5-qualification-20261103/evidence.json`, `attempts`, `reproduction`, `measurement`, `join_to_v3_diagnosis` and `conclusion`; review note by opus-mcps-1029-8f6e on board task f963ee0b at 2026-11-03T08:12:33Z.
[^suspension]: `libs/covenant_ml/src/covenant_ml/markets/crsp_protocol.py`, `decode_crsp_protocol`; `libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py`, `_unpriced_held` and `crsp_resolve_held`; `libs/covenant_ml/tests/markets/test_crsp_unfilled.py`, `test_old_rule_identity_cannot_admit_the_new_evaluator` and `test_old_encrypted_protocol_is_refused_without_reinterpreting_its_data`.

A replacement must distinguish selection, order execution, owned exposure and
label availability. Decision-time membership and features remain past-only.
An unavailable fill may be modeled as an unfilled order only under a frozen
execution policy; it must not remove a security from the earlier universe or
silently redistribute its allocation. Missing value on an owned position
remains a different problem.[^owners][^features]

A terminal-date extension requires explicit valuation and label-maturity dates,
purging through that maturity and compatible portfolio accounting. Through
version 4 the resolver refused valuation beyond exit, because relaxing that
comparison alone would book later information into an earlier month's labels.
Version 5 removes the refusal only together with dating every label of the
decision at its latest valuation, which the existing purge then respects; the
receivable rule already kept such a value out of reinvestment.[^owners]

The current extraction also filters entire delisting rows by their latest
valuation date at the partition boundary. Source review must distinguish an
event omitted by that predicate from a security with no terminal event; event
identity and unavailable value are different facts. This is a prospective
boundary check, not an established explanation of the 27 observed cases.
Any repair must preserve the partition's prohibition on future valuations.[^partition]
Version 5 refuses such a path by name (`delisting return without event`), and
the native scan found none in this extract.[^v5]

The required source check is bounded to the diagnosed paths and partition
edges, using the existing encrypted extraction and evaluator controls. Compare
the frozen vintage with newly captured vendor fields and retain exact SQL,
timestamps and digests. No correction is established until that comparison
measures one. If the source remains incomplete, qualification must report the
remaining refusal; complete-case deletion or invented fills cannot produce
the original full-universe baseline.[^cases][^partition]

[[wrds-encrypted-diagnostic-cohort]] documents the frozen-vintage subset
derivation used to prepare this comparison. That diagnostic preserves source
lineage and cannot pass the study loader.[^cohort]

The diagnostic query builder requires the exact train-validation scope
derived from the supplied protocol, including all date bounds. Renaming a
final scope or altering an individual bound is refused before SQL is returned.
This hardens future query admission; it supplies no missing outcome and does
not establish that a previous extraction crossed its boundary.[^query-scope]

The completed [[wrds-crsp-source-comparison]] measured all 81,920 queried
rows unchanged. It establishes no correction within the frozen four-relation
projection and validation bounds. The original full-universe target therefore
remains unqualified; repeating the same extraction is not an outcome repair.
The comparison and subsequent scalar-error privacy repair passed separate
independent audits. Neither review approves a changed economic rule or a new
study freeze.[^refresh][^source-checks]

## Additional evidence to qualify

The October 27 metadata inventory records direct access to
`crsp_a_stock.stkdistributions` and additional `stkdelists` columns. This is
historical schema/access evidence, not a fresh query or proof that a diagnosed
security has a usable payment. The guide distinguishes distribution ex-date,
payment date, amount and linked security from delisting amount date and stored
return date.[^catalog][^distribution]

The implemented [[wrds-crsp-corporate-action-capture]] captures these fields
for the authenticated cohort through the shared query, snapshot and encryption
owners. Fleet verification, native synthetic capture and bounded licensed
capture passed, and independent audit `2bfe3456` verified the capture milestone
on November 3 at 04:00 UTC. The licensed capture contains nineteen delisting events and
328 distribution events, all with admissible detail rows. These counts do not
establish usable consideration or repair the 27 refused paths. Linked
identifiers do not authorize recursive security expansion, and event presence
is recorded separately from details excluded by valuation/payment dates.
This investigation supplies no replacement return policy.[^actions]

The authenticated [[wrds-crsp-action-case-evidence]] now matches all 27 cases
to their original monthly windows. Eighteen have a captured action within the
window or on exit; this is measured coverage, not eighteen repaired outcomes.
Independent audit `ffd0a417` verified the native replay and licensed join on
November 3 at 05:08 UTC. The other nine cases do not acquire an outcome from
this join.[^casejoin]

The case-by-case reconciliation binds the action report to the original entry
digests, periods and exit offsets. All six NT cases, both SU cases and the open
MP tail have no captured event inside the window or on exit. All six MV cases
have relevant distributions, but each has at least one null distribution amount.
Null does not mean zero and does not establish the form of consideration.
For the three BA old-position exits, delisting amount dates are respectively
three, one and one calendar days beyond exit. Thus accepting the event on exit
alone would still leave a valuation-timing issue. The fourth entry-date
delisting is the MV terminal case 173979 with a TR entry; it must not be
grouped with the three rejected BA entry candidates.[^reconciliation]

Interpretation must answer whether cash or security consideration is complete,
when it is valued, and whether the daily total return already contains it.
Adding distributions to a total return without that reconciliation can double
count the same consideration. A populated payment field alone does not prove
an executable liquidation, an exit-date mark or contemporaneous vendor
availability. Unresolved NT, MV, DG and open tails continue to refuse; no
selection on future completeness or zero-valued imputation is approved.
Qualification still requires a reviewed policy and a full-panel native check
before new study admission.[^distribution][^vendor][^owners]

[^cases]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json`, `results[0].refused_cases`, source-window flags, relative date offsets and first-refusal counts.
[^vendor]: [CRSP CIZ guide](https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d), pp. 11–12, delisting date/value definitions; p. 88, PC; pp. 90–92, RD and RM; archived source and digest in frontmatter. Tech-wiki `crsp-ciz-aggregate-return-quality`, AR and DE definitions, cites the same guide pp. 49 and 63.
[^owners]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_rows`, `_entry`, `crsp_resolve_window_path`, `crsp_window_path` and `evaluate_crsp_holdings`; `crsp_return_path.py`, `_unpriced_held` and `crsp_resolve_held`; `crsp_calendar.py`, `crsp_resolve_terminal` and `crsp_executable`.
[^features]: `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `_decision` and `_feature`.
[^labels]: `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_study_rows`, the mean of every eligible member's multiplier, and `crsp_arms`, `EQUAL_WEIGHT` selection; `libs/covenant_ml/tests/markets/test_crsp_study.py`, `test_one_unavailable_outcome_invalidates_the_full_universe_label_benchmark`.
[^dependence]: `tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/benchmark-dependence.json`, inputs, ordinal derivation and period counts; sibling `protocol.json`, `max_universe`, exact bytes matching `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `revision.revision.protocol`; `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `_decision`, universe cap; `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py`, consecutive `checked` increment and `path_ordinal` recording; sibling `result.json`, `results[0].decisions`, `eligible_paths` and `refused_cases`.
[^partition]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `crsp_relation_sql`, delisting valuation predicate; `crsp_partition.py`, `check_crsp_partition`, dated-field checks.
[^cohort]: [[wrds-encrypted-diagnostic-cohort]], § "One transformation owner" and § "Admission is not permission to fit this subset".
[^query-scope]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `_universe` called by `crsp_relation_sql`, equality with `crsp_scopes(protocol)[0]`; `libs/covenant_ml/tests/markets/test_crsp_queries.py`, `test_renaming_final_scope_cannot_admit_its_observations_to_a_diagnostic`, `test_diagnostic_refuses_each_altered_protocol_boundary`, and unchanged valid-statement assertions.
[^refresh]: [[wrds-crsp-source-comparison]], § "Diagnostic contract" and § "Native evidence".
[^source-checks]: `tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/source-check-audits.json`, captured live board responses: comparison audit `dc59b228` completed VERIFIED at 2026-11-03T01:44:51.513Z; scalar-error privacy audit `9ccc6e3a` completed VERIFIED at 2026-11-03T01:55:18.663Z, each against its own acceptance criteria.
[^catalog]: `tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/corporate-action-catalog.json`, exact metadata rows from the October 27 inventory and SHA-256 identities of the source column/access CSV files, rechecked November 3 UTC.
[^actions]: [[wrds-crsp-corporate-action-capture]], § "Scope and privacy" and § "Verification status".
[^casejoin]: [[wrds-crsp-action-case-evidence]], § "Acceptance" and § "Licensed coverage and its limits".
[^reconciliation]: `tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/action-reconciliation.json`, exact source digests, method, case classes/counts, null-amount counts and delisting amount offsets from exit; original diagnosis `result.json`, path 173979 entry price flag and terminal event offset. No economic rule or return calculation is added by this metadata reconciliation.
[^distribution]: [CRSP CIZ guide](https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d), pp. 11–13, StkDelists and StkDistributions field definitions; pp. 17–18, DlyRet total-return definition. Archived primary source and digest in frontmatter. Interpretation and proposed investigation are this project's requirements, not a claim that those fields close the observed gaps.
