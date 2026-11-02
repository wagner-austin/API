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
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "8db552627c0a8cf8962514d5c2a49b8d7dcd1416"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_return_path.py": "b0979d5cd4994793c83e3b15e743b4d7fcf21760"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py": "87201520de3acd3aa06b8f59d222e1bd53bbf667"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "d7fca763bfbbc141e0ea0bda85c53183f19ed10e"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_partition.py": "b8752083aca89e252059bfef5e39cb29c1bdbb57"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py": "1a5c4f11e1e90bf50d29d800eeb03fc941721cc6"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json": "5fca13ce388bd45adc74f8e52efa5d7d42af7775"
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

[^cases]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/result.json`, `results[0].refused_cases`, source-window flags, relative date offsets and first-refusal counts.
[^vendor]: [CRSP CIZ guide](https://indexes.morningstar.com/api/docs/6a71c517e35d02a71403136d), pp. 11–12, delisting date/value definitions; p. 88, PC; pp. 90–92, RD and RM; archived source and digest in frontmatter. Tech-wiki `crsp-ciz-aggregate-return-quality`, AR and DE definitions, cites the same guide pp. 49 and 63.
[^owners]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_row`, `_entry_filled`, `crsp_window_path` and `evaluate_crsp_holdings`; `crsp_return_path.py`, `_missing_price` and `crsp_resolve_held`; `crsp_calendar.py`, `crsp_resolve_terminal` and `require_crsp_execution_price`.
[^features]: `libs/covenant_ml/src/covenant_ml/markets/crsp_panel.py`, `_decision` and `_feature`.
[^partition]: `libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py`, `crsp_relation_sql`, delisting valuation predicate; `crsp_partition.py`, `check_crsp_partition`, dated-field checks.
