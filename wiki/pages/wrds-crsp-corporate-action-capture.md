---
title: Bounded encrypted CRSP corporate-action evidence
tags: [wrds, crsp, encryption, diagnostics]
hubs: [libs]
related: ["[[crsp-outcome-qualification]]", "[[wrds-crsp-source-comparison]]", "[[wrds-encrypted-diagnostic-cohort]]"]
source_paths:
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/capture-audit.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/fixture-audit.json
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
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/source.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-expected.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-setup.sql
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-actions.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-completion.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-ciphertext-reading.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-actions.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-completion.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-ciphertext-reading.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-stage-inspection.json
  - tools/hpc3/artifacts/wrds-corporate-actions-20261103/staging-permission-correction.json
  - tools/hpc3/src/hpc3/core/stage.py
  - tools/hpc3/src/hpc3/core/remote.py
source_git_blobs:
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/capture-audit.json": "f062921fbf4784ce66199719cdde8ff49343e8cd"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/fixture-audit.json": "2358a8ef26c32e8561bc81c390d954976ac2af82"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_action_schema.py": "a62a7dd519ffcb861fa25c9318f54441f0dd8521"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_action_records.py": "667cc6fd8d1b6d216ac5bf6271e63779260acfc6"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py": "2cf03ae973b98bfd3a0a0edb9d7fabc7acf54663"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_diagnostic.py": "7f7ba67c59ff2294f0dbdcb1adca53866a2c4ab2"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_queries.py": "4e6afdb5f0fe6d1730185aa31f3ca13a39aad1dc"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py": "3c2f55409d11a293113bf42ef050a5e2fbff3de3"
  "libs/covenant_ml/scripts/crsp_extraction.py": "89ae6f053b812a620055a585ebb3d361331330cf"
  "libs/covenant_ml/tests/markets/test_crsp_actions.py": "b18dd3e4542ab12c26f862ac4696796e8e044176"
  "libs/covenant_ml/tests/markets/test_crsp_actions_process.py": "f62b54375902441dd35d75f5db5a500d2b6f9e10"
  "libs/covenant_ml/tests/markets/test_crsp_action_queries.py": "798b5d739e45026f2d9ef397a438ae9aa4a7b748"
  "libs/covenant_ml/tests/markets/test_crsp_refresh.py": "fdfe74bbc7918dce6853413d9622c81e983b47fa"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/source.json": "5f6ea00fe4e07dcb526513a3a2b076cbae77285d"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-expected.json": "e3b377b76883387645243ac085b6705c6ec10fee"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-setup.sql": "824d4fa7c335bd93e5d9a8b995abcbcb5f6bd2d4"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-actions.json": "89126a9b770b7eb56748996a023aad80d29b867b"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-completion.json": "5bbc48d896921ee2f1bd0723924cfdb179abdb52"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-ciphertext-reading.json": "98b0ba121c4cbadb6a8023d06875b01feb1b90a0"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-actions.json": "029d1492fe45296838e87ee259ca3e879a48e6f0"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-completion.json": "9d9c2f2f9379d08541f4189ed446cef134e3dabc"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-ciphertext-reading.json": "5776156b64f624c901e96f22577aeaa236ec970d"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-stage-inspection.json": "d71bc9b76d8514460031cb2fd0a01c2883b63095"
  "tools/hpc3/artifacts/wrds-corporate-actions-20261103/staging-permission-correction.json": "67d9b332f8d142b8d5b40e65be7f73ea3ea9f4a5"
  "tools/hpc3/src/hpc3/core/stage.py": "450870282e6df188076183fb5b65a24b795bc3f7"
  "tools/hpc3/src/hpc3/core/remote.py": "80dd48cea51b34dbd3746ae7c304c36c6ce51de1"
provenance:
  - "Implementation task fd32dc2e-b3c0-4e24-977c-8708667453e6 independently VERIFIED by audit 2bfe3456-5bc7-46d4-abeb-db0042405b2f at 2026-11-03T04:00:56.684Z. Native source remains c5baa6df3; reviewed published revision f10fd982cb7e includes the accepted test-fixture repair."
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

The per-case join and its verification limits are documented in [[wrds-crsp-action-case-inspection]].[^casejoin]

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

Independent audit `2bfe3456` verified all three capture criteria on November 3
at 04:00 UTC. Its checks include the native receipts, encrypted staging
readback and published wiki audits. The final controller check passed 2,205
tests at 100% coverage in 32 seconds; the evaluator check passed 5,534 at 100%
in 208 seconds, both within the unchanged 300-second budget. An earlier
evaluator run passed every test but exceeded that budget at 303 seconds.
Repair `4df253cc`, independently accepted before this audit, reduced repeated
cohort-fixture encryption while retaining real encryption and foreign-security
exclusion assertions. These runs used different fleet hosts and do not measure
a controlled speedup. The acceptance verifies capture, not economic
qualification.[^audit]

The full package fleet check passed 5,534 tests with zero failures and 100%
coverage, including lint, formatting, typing and guards. Tests cover encrypted
round trips, scalar-error privacy, missing final keys,
scope/limit refusals, real child-process success and nonzero failure, scratch
cleanup, and attempted study admission. Those process tests use controlled
CSV output; native PostgreSQL execution is recorded separately below. Normal
cleanup does not establish forced-termination cleanup.[^tests][^fleet]

Native synthetic job `40569021` ran the frozen source `c5baa6df3` against
PostgreSQL. Its predeclared oracle and collected receipt agree: two delisting
events and two distribution events, with one admissible detail and one event
without admissible details for each relation. The fixture includes foreign
securities, events outside both date bounds, nulls, zero amounts and
in-bound events with future-dated details. Collection verified all seventeen
outputs; the bundle inventory contains fourteen ciphertexts plus its metadata,
and the receipt confirms normal plaintext cleanup. The existing controller
retains the runtime fingerprint and published the native evidence.[^native]

Licensed job `40569098` captured nineteen delisting events and 328 distribution
events, with the same counts of admissible detail rows and no events excluded
from details by the partition date checks. Collection verified all sixteen
outputs, including thirteen ciphertexts plus bundle metadata. The public receipt
reports normal plaintext cleanup. Exact observations, SQL, snapshot readings
and source lineage remain encrypted.[^licensed]

HPC3 staging and separate complete-tree readback verified fourteen bundle files
and exactly fifteen inventory entries including the certificate. The first
inspection refused the newly created destination's mode `0755`; changing only
that directory to `0700` allowed the repeated inventory check to pass. The
generic stage command uses the account umask, so private destination creation
must precede transfer; digest verification alone does not establish private
permissions. No plaintext observations or keys were transferred.[^staging]

Matching event/detail counts do not establish complete consideration or usable
investment outcomes. The 27 economic refusals remain unresolved, and the
substantive study remains stopped. No fitting or final-scoring release occurred.
The previous four-relation comparison remains in
[[wrds-crsp-source-comparison]].[^licensed][^prior]

[^capture]: `libs/covenant_ml/scripts/crsp_extraction.py`, `main` diagnostic branch; `libs/covenant_ml/src/covenant_ml/markets/crsp_diagnostic.py`, `capture_crsp_diagnostic`; `crsp_actions.py`, `write_crsp_actions`.
[^queries]: `libs/covenant_ml/src/covenant_ml/markets/crsp_action_schema.py`, `CRSP_ACTIONS`; `crsp_queries.py`, `_universe`, `crsp_action_sql` and `crsp_driver_sql`; `libs/covenant_ml/tests/markets/test_crsp_action_queries.py`, exact-query, altered-scope and final-role refusals.
[^rows]: `libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py`, `check_crsp_action_rows`; `crsp_rows.py`, `decode_crsp_rows` and `_cell`; `libs/covenant_ml/tests/markets/test_crsp_actions.py`, post-query validation and null/zero round trip.
[^record]: `libs/covenant_ml/src/covenant_ml/markets/crsp_action_records.py`, `decode_crsp_action_evidence`; `libs/covenant_ml/tests/markets/test_crsp_refresh.py`, parameterized final-bundle and actual-trial refusals for both operations.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_actions.py`, capture, codec and refusal cases; `test_crsp_actions_process.py`, real child-process checks and privacy assertions.
[^fleet]: Fleet job `ff02f00e-b5ad-4295-8dbe-f121812392c3`, terminal FLEET-CHECK receipt and full `libs-covenant_ml-lavender-1793673769.log`; exact source identity, counts and completion time in provenance.
[^native]: `tools/hpc3/artifacts/wrds-corporate-actions-20261103/synthetic-expected.json`, `rows`; sibling `synthetic-setup.sql`, fixture CTEs; `synthetic-actions.json`, `rows` and `plaintext_removed`; `synthetic-completion.json`, `outputs`, `runtime` and `record`; `synthetic-ciphertext-reading.json`, exact ciphertext inventory and digests; `source.json`, frozen source and worker identities. Native service report `7565ed95-a6ba-4447-aaab-723aac830dca` binds completed job `40569021` to those output digests.
[^licensed]: `tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-actions.json`, `rows`, `bundle` and `plaintext_removed`; sibling `licensed-completion.json`, exact output digests and runtime; `licensed-ciphertext-reading.json`, checked inventory. Native service report `818a292e-791b-4174-b113-88ce2ede75d6` records scheduler completion and output identities. These are capture measurements, not economic qualification; [[crsp-outcome-qualification]], § "Additional evidence to qualify", states the remaining requirements.
[^staging]: `tools/hpc3/artifacts/wrds-corporate-actions-20261103/licensed-stage-inspection.json`, `manifest`, `verified_files`, `inventory.directories` and exact `inventory.files`; sibling `staging-permission-correction.json`, first refusal, measured modes and correction. Service report `a88d8a8d-9cfb-41a1-92eb-f803a0bde77a` records the independent readback. `tools/hpc3/src/hpc3/core/stage.py`, `stage_manifest` and `inspect_staged_tree`; `remote.py`, `make_directory`.
[^prior]: [[wrds-crsp-source-comparison]], § "Native evidence".
[^casejoin]: [[wrds-crsp-action-case-inspection]], § "Authentication and calendar ownership" and § "Verification boundary".
[^audit]: `tools/hpc3/artifacts/wrds-corporate-actions-20261103/capture-audit.json`, captured live `task_get` response: verdict, criteria, fleet checks `33ee64b6` and `b5cdaa51` and independent review evidence; sibling `fixture-audit.json`, accepted repair `4df253cc`, preserved budget failure and reviewer checks of the reduced fixture. [[wrds-encrypted-diagnostic-cohort]], § "Admission is not permission to fit this subset", describes the retained assertions.
