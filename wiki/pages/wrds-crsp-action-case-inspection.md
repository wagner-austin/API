---
title: CRSP action evidence matched to refused holding windows
tags: [wrds, crsp, diagnostics, provenance]
hubs: [libs]
related: ["[[wrds-crsp-corporate-action-capture]]", "[[crsp-outcome-qualification]]"]
source_paths:
  - libs/covenant_ml/src/covenant_ml/markets/crsp_cohort.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_action_cases.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py
  - libs/covenant_ml/scripts/crsp_extraction.py
  - libs/covenant_ml/tests/markets/test_crsp_action_cases.py
  - libs/covenant_ml/tests/markets/test_crsp_action_inspection.py
source_git_blobs:
  libs/covenant_ml/src/covenant_ml/markets/crsp_cohort.py: dac732909d85ff89967b8c345d7f7af769149ebd
  libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py: 2cf03ae973b98bfd3a0a0edb9d7fabc7acf54663
  libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py: 85ee73a0726c6e43bea6d5af991d2ed78a9a0252
  libs/covenant_ml/src/covenant_ml/markets/crsp_action_cases.py: b04d764ef7583264b012a00bba6a734d36343f8b
  libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py: 87201520de3acd3aa06b8f59d222e1bd53bbf667
  libs/covenant_ml/scripts/crsp_extraction.py: 89ae6f053b812a620055a585ebb3d361331330cf
  libs/covenant_ml/tests/markets/test_crsp_action_cases.py: 9fba1ef87831b64416145efb6f7382dae6c6b93c
  libs/covenant_ml/tests/markets/test_crsp_action_inspection.py: adfb9f478cc3fc2544175b733e6a272f61613f32
fact_checked: 2026-11-03
confidence: medium
---

# CRSP action evidence matched to refused holding windows

Implementation is awaiting fleet verification and native rehearsal under board
task `4fca73b7-9c36-47a9-a705-7541813c9338`. This diagnostic measures what the
captured evidence contains; it does not qualify outcomes or authorize the
baseline. The accepted capture is described in [[wrds-crsp-corporate-action-capture]].

## Authentication and calendar ownership

The inspector authenticates both encrypted bundle identities, requires the
capture's complete cohort record to match the original cohort, and checks the
combined plaintext budget before opening either bundle. Declared action files
must match their byte counts and digests; undeclared evidence and an oversized
capture are refused. Both contexts use the train-validation identity.[^inspection]

The existing cohort validator owns audit completeness and entry-row hashes.
The inspector resolves those anchors against the frozen observations, derives
monthly windows using the canonical calendar, and checks the original audit's
period and exit offset. Non-monthly entries and windows crossing into the final
holdout are refused. There is no second calendar or anchor implementation.[^windows]

## Meaning of the report

Each case retains its original ordinal, period, entry digest and relative exit
offset. For that security, every captured event remains visible, including events
before entry, on entry, within the holding window, on exit and afterward.
An event with no admitted detail remains present with a null detail.[^events]

Details expose row hashes, relative date offsets, field presence, and selected
null-versus-zero distinctions. Raw identifiers, absolute dates, flag strings
and financial amounts are not copied into the report. Linked-security presence
does not trigger a recursive expansion.[^events]

These fields do not establish complete consideration, actual settlement or
historical publication availability. The inspector computes no return, label or
fitted score and always returns `outcomes_qualified: false`. Economic acceptance
remains governed by [[crsp-outcome-qualification]].[^inspection]

## Verification boundary

The tests specify entry/exit and late-detail boundaries, event-only rows, altered
authenticated lineage, extra files, scope violations, combined plaintext limits
and corrupt ciphertext. Real child processes exercise success and failure;
successful inspection requires no final key. Execution results are still pending.[^tests]

The existing extraction CLI exposes `inspect-actions` and emits metadata plus
the native controller's metrics file. It refuses occupied result paths before
decryption. Its cleanup field concerns normal context unwinding; it is not a
forced-termination or process-isolation guarantee.[^cli]

[^inspection]: `libs/covenant_ml/src/covenant_ml/markets/crsp_action_cases.py`, `inspect_crsp_actions`; `libs/covenant_ml/src/covenant_ml/markets/crsp_extract.py`, `read_crsp_evidence`; `libs/covenant_ml/src/covenant_ml/markets/crsp_actions.py`, `check_crsp_action_rows`.
[^windows]: `libs/covenant_ml/src/covenant_ml/markets/crsp_cohort.py`, `crsp_audit_cases`, `resolve_crsp_anchors`; `libs/covenant_ml/src/covenant_ml/markets/crsp_action_cases.py`, `crsp_action_cases`; `libs/covenant_ml/src/covenant_ml/markets/crsp_calendar.py`, `crsp_windows`.
[^events]: `libs/covenant_ml/src/covenant_ml/markets/crsp_action_cases.py`, `_events` and `_detail`.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_action_cases.py`, boundary and calendar refusal tests; `libs/covenant_ml/tests/markets/test_crsp_action_inspection.py`, `test_real_encrypted_case_inspection_needs_no_final_key`, `test_authenticated_but_inconsistent_capture_refuses`, `test_real_inspection_child_emits_metrics_or_cleans_ciphertext_failure`.
[^cli]: `libs/covenant_ml/scripts/crsp_extraction.py`, `_inspect_actions`; `libs/covenant_ml/src/covenant_ml/markets/crsp_action_cases.py`, `inspect_crsp_actions` cleanup return.
