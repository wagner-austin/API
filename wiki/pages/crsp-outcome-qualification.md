---
title: CRSP outcome qualification before a replacement study
tags: [wrds, crsp, qualification, missingness]
hubs: [libs]
related: ["[[wrds-v3-refused-path-diagnosis]]", "[[wrds-baseline-readiness]]", "[[crsp-monthly-holdings-validation]]"]
source_paths:
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
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "8db552627c0a8cf8962514d5c2a49b8d7dcd1416"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py": "b0979d5cd4994793c83e3b15e743b4d7fcf21760"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "87201520de3acd3aa06b8f59d222e1bd53bbf667"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "13da81586926004b2f0ea07534af2dd295c5a1ea"
  "libs/covenant_ml/tests/markets/test_crsp_queries.py": "254aa595426041e71931fbe0b724d60844a8597b"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py": "b8752083aca89e252059bfef5e39cb29c1bdbb57"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py": "1a5c4f11e1e90bf50d29d800eeb03fc941721cc6"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "824b9bad6e08e7d4955dad45f2b8a48bfa4ab6a9"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "ac6baa65e78f71d8bed5728e162e05773551b81d"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json": "5fca13ce388bd45adc74f8e52efa5d7d42af7775"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py": "c9c37944ab5ca06f18953e9f9e3d4d21cb0e8438"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/protocol.json": "a095650070b230379e759563df3c20b3c2476ded"
  "tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/benchmark-dependence.json": "b2994cf9db46aff4a4714b987bd4f5c0729dd5b2"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
provenance:
  - "Primary capture: tech-wiki/sources/crsp-ciz-guide-20260731.txt; SHA-256 bb4792deb927b972a8bef1a656ba1fc5442b80783e10eecb3e633d4243334b2e; corpus 0264d5cc-4dec-4715-b17b-45f3c8e5fbec; reread November 2."
  - "Board task f963ee0b-7382-4d7a-a1db-9a7b4d5467a3; this is qualification analysis, not an approved version-4 freeze."
fact_checked: 2026-11-02
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
| BA on a rebalance delisting | 6 | Preserve three old-position exits separately from three new-order entries. A quote cannot satisfy a TR fill. A terminal boundary amendment must account for the old claim once and separately specify rejection of a new order. |
| NT | 6 | One entry and five held paths lack tracked evidence. An unfilled-new-order assumption supplies no value for an already owned security. Retain refusal unless bounded source review supplies the missing outcome. |
| MV | 6 | Two held paths and four terminal paths contain numbers flagged incomplete. Require complete corporate-action evidence; neither a numeric value nor monthly aggregation establishes completeness. |
| Valuation after exit | 5 | Four are one calendar day late and one is two days late. Later-known terminal value must not be booked as known wealth at the earlier exit. Extending maturity changes the target and split requirements. |
| SU | 2 | One exactly linked P9 span closes nine suspended sessions; extending the current MP-only predicate needs an explicit rule and adversarial tests. The other suspension remains open at exit and has no complete span. |
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

A replacement must distinguish selection, order execution, owned exposure and
label availability. Decision-time membership and features remain past-only.
An unavailable fill may be modeled as an unfilled order only under a frozen
execution policy; it must not remove a security from the earlier universe or
silently redistribute its allocation. Missing value on an owned position
remains a different problem.[^owners][^features]

A terminal-date extension requires explicit valuation and label-maturity dates,
purging through that maturity and compatible portfolio accounting. The current
resolver deliberately refuses valuation beyond exit. Relaxing that comparison
alone would book later information into an earlier month's wealth.[^owners]

The current extraction also filters entire delisting rows by their latest
valuation date at the partition boundary. Source review must distinguish an
event omitted by that predicate from a security with no terminal event; event
identity and unavailable value are different facts. This is a prospective
boundary check, not an established explanation of the 27 observed cases.
Any repair must preserve the partition's prohibition on future valuations.[^partition]

The required source check is bounded to the diagnosed paths and partition
edges, using the existing encrypted extraction and evaluator controls. Compare
the frozen vintage with newly captured vendor fields and retain exact SQL,
timestamps and digests. No correction is established until that comparison
measures one. If the source remains incomplete, qualification must report the
remaining refusal; complete-case deletion or invented fills cannot produce
the original full-universe baseline.[^cases][^partition]

[[wrds-encrypted-diagnostic-cohort]] documents the frozen-vintage subset
derivation used to prepare this comparison. That diagnostic preserves source
lineage and cannot pass the study loader; fresh vendor evidence is a separate
step.[^cohort]

The diagnostic query builder requires the exact train-validation scope
derived from the supplied protocol, including all date bounds. Renaming a
final scope or altering an individual bound is refused before SQL is returned.
This hardens future query admission; it supplies no missing outcome and does
not establish that a previous extraction crossed its boundary.[^query-scope]

[[wrds-crsp-source-comparison]] records the shared comparison owner, its
synthetic database rehearsals and the bounded vendor comparison. Its diagnostic
envelope is separate from a study dataset; unchanged source rows would not
establish complete economic outcomes.[^refresh]

[^cases]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json`, `results[0].refused_cases`, source-window flags, relative date offsets and first-refusal counts.
[^vendor]: [CRSP CIZ guide](https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d), pp. 11–12, delisting date/value definitions; p. 88, PC; pp. 90–92, RD and RM; archived source and digest in frontmatter. Tech-wiki `crsp-ciz-aggregate-return-quality`, AR and DE definitions, cites the same guide pp. 49 and 63.
[^owners]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_row`, `_entry_filled`, `crsp_window_path` and `evaluate_crsp_holdings`; `crsp_return_path.py`, `_missing_price` and `crsp_resolve_held`; `crsp_calendar.py`, `crsp_resolve_terminal` and `require_crsp_execution_price`.
[^features]: `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `_decision` and `_feature`.
[^labels]: `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_study_rows`, the mean of every eligible member's multiplier, and `crsp_arms`, `EQUAL_WEIGHT` selection; `libs/covenant_ml/tests/markets/test_crsp_study.py`, `test_one_unavailable_outcome_invalidates_the_full_universe_label_benchmark`.
[^dependence]: `tools/hpc3/artifacts/wrds-v3-outcome-qualification-20261102/benchmark-dependence.json`, inputs, ordinal derivation and period counts; sibling `protocol.json`, `max_universe`, exact bytes matching `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `revision.revision.protocol`; `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `_decision`, universe cap; `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/audit-execution.py`, consecutive `checked` increment and `path_ordinal` recording; sibling `result.json`, `results[0].decisions`, `eligible_paths` and `refused_cases`.
[^partition]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `crsp_relation_sql`, delisting valuation predicate; `crsp_partition.py`, `check_crsp_partition`, dated-field checks.
[^cohort]: [[wrds-encrypted-diagnostic-cohort]], § "One transformation owner" and § "Admission is not permission to fit this subset".
[^query-scope]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `crsp_relation_sql`, equality with `crsp_scopes(protocol)[0]`; `libs/covenant_ml/tests/markets/test_crsp_queries.py`, `test_renaming_final_scope_cannot_admit_its_observations_to_a_diagnostic`, `test_diagnostic_refuses_each_altered_protocol_boundary`, and unchanged valid-statement assertions.
[^refresh]: [[wrds-crsp-source-comparison]], § "Diagnostic contract" and § "Native evidence".
