---
title: Bounded encrypted CRSP corporate-action evidence
tags: [wrds, crsp, encryption, diagnostics]
hubs: [libs]
related: ["[[crsp-outcome-qualification]]", "[[wrds-crsp-source-comparison]]", "[[wrds-encrypted-diagnostic-cohort]]"]
source_paths:
  - libs/covenant_ml/src/covenant_ml/markets/crsp_action_schema.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_action_records.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_diagnostic.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py
  - libs/covenant_ml/scripts/crsp_extraction.py
  - libs/covenant_ml/tests/markets/test_crsp_actions.py
  - libs/covenant_ml/tests/markets/test_crsp_actions_process.py
  - libs/covenant_ml/tests/markets/test_crsp_action_queries.py
  - libs/covenant_ml/tests/markets/test_crsp_refresh.py
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/markets/crsp_action_schema.py": "a62a7dd519ffcb861fa25c9318f54441f0dd8521"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_action_records.py": "667cc6fd8d1b6d216ac5bf6271e63779260acfc6"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py": "108f53efd42dd0d1de61297df3f1970ece097c33"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_diagnostic.py": "7f7ba67c59ff2294f0dbdcb1adca53866a2c4ab2"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "4e6afdb5f0fe6d1730185aa31f3ca13a39aad1dc"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py": "3c2f55409d11a293113bf42ef050a5e2fbff3de3"
  "libs/covenant_ml/scripts/crsp_extraction.py": "edb2cfb8dbd7f4646ea0c4ecf6da7b70d9b2cf1c"
  "libs/covenant_ml/tests/markets/test_crsp_actions.py": "677a639a0e779b79251b58cdc2377fa2f44c9c1b"
  "libs/covenant_ml/tests/markets/test_crsp_actions_process.py": "f62b54375902441dd35d75f5db5a500d2b6f9e10"
  "libs/covenant_ml/tests/markets/test_crsp_action_queries.py": "798b5d739e45026f2d9ef397a438ae9aa4a7b748"
  "libs/covenant_ml/tests/markets/test_crsp_refresh.py": "fdfe74bbc7918dce6853413d9622c81e983b47fa"
provenance:
  - "Implementation task fd32dc2e-b3c0-4e24-977c-8708667453e6; native capture and independent review pending."
  - "Fleet ff02f00e-b5ad-4295-8dbe-f121812392c3 checked API 9b076cba5eff905617afd9335a09d3a44f2b8843 on Lavender: 5534 passed, zero failed, 100.00% coverage, exit zero, 277/300 seconds; completed 2026-11-03T02:48:06Z."
fact_checked: 2026-11-03
confidence: medium
---

# Bounded encrypted CRSP corporate-action evidence

## Scope and privacy

The extraction CLI's `actions` command shares `capture_crsp_diagnostic` with
`refresh`: authenticated cohort opening, validation-key acquisition, private
scratch, complete-evidence byte bounds, encryption and cleanup have one owner.
No final key is acquired. Public receipts contain identities, timestamps,
aggregate counts and cleanup status; observations and exact SQL stay encrypted.
The operation supplies evidence for [[crsp-outcome-qualification]].[^capture]

One fixed projection registry defines four roles: delisting event identities,
delisting details, distribution event identities and distribution details.
Queries share the existing decision-time membership selection and require the
authenticated cohort's ordered securities and exact validation scope. Linked
security identifiers are retained as fields; the query never recursively
expands them. A limit-plus-one sentinel detects row truncation.[^queries]

Event projections contain only their source keys. Detail queries exclude rows
with any dated field after the partition end, while event queries retain
eligible event identities. Counts distinguish events without admissible
details. Post-query checks independently reject excess bytes or rows, foreign
securities, out-of-bound dates and details without matching events. Nulls remain
distinct from zero, including absent dates and linked identifiers.[^rows]

The encrypted `crsp-corporate-action-diagnostic-v1` record binds the cohort,
query artifacts, snapshot readings, source identity, timestamps and limits.
Its codec requires the exact file inventory and matching protocol/setup
digests. It is rejected by the study loader and actual evaluation path; it
does not add a study relation or change holdings accounting.[^record]

## Verification status

The full package fleet check passed 5,534 tests with zero failures and 100%
coverage, including lint, formatting, typing and guards. Tests cover encrypted
round trips, scalar-error privacy, missing final keys,
scope/limit refusals, real child-process success and nonzero failure, scratch
cleanup, and attempted study admission. Those process tests use controlled
CSV output; native PostgreSQL execution remains a separate rehearsal. Normal
cleanup does not establish forced-termination cleanup.[^tests][^fleet]

No licensed corporate-action capture or newly qualified economic outcome is
reported here. The previous four-relation native comparison remains in
[[wrds-crsp-source-comparison]].[^prior]

[^capture]: `libs/covenant_ml/scripts/crsp_extraction.py`, `main` diagnostic branch; `libs/covenant_ml/src/covenant_ml/markets/crsp_diagnostic.py`, `capture_crsp_diagnostic`; `crsp_actions.py`, `write_crsp_actions`.
[^queries]: `libs/covenant_ml/src/covenant_ml/markets/crsp_action_schema.py`, `CRSP_ACTIONS`; `crsp_queries.py`, `_universe`, `crsp_action_sql` and `crsp_driver_sql`; `libs/covenant_ml/tests/markets/test_crsp_action_queries.py`, exact-query, altered-scope and final-role refusals.
[^rows]: `libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py`, `check_crsp_action_rows`; `crsp_rows.py`, `decode_crsp_rows` and `_cell`; `libs/covenant_ml/tests/markets/test_crsp_actions.py`, post-query validation and null/zero round trip.
[^record]: `libs/covenant_ml/src/covenant_ml/markets/crsp_action_records.py`, `decode_crsp_action_evidence`; `libs/covenant_ml/tests/markets/test_crsp_refresh.py`, parameterized final-bundle and actual-trial refusals for both operations.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_actions.py`, capture, codec and refusal cases; `test_crsp_actions_process.py`, real child-process checks and privacy assertions.
[^fleet]: Fleet job `ff02f00e-b5ad-4295-8dbe-f121812392c3`, terminal FLEET-CHECK receipt and full `libs-covenant_ml-lavender-1793673769.log`; exact source identity, counts and completion time in provenance.
[^prior]: [[wrds-crsp-source-comparison]], § "Native evidence".
