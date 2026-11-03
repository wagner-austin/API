---
title: Encrypted CRSP diagnostic cohort
tags: [wrds, crsp, encryption, provenance]
hubs: [libs]
related: ["[[crsp-outcome-qualification]]", "[[wrds-v3-refused-path-diagnosis]]", "[[wrds-baseline-readiness]]"]
source_paths:
  - libs/covenant_ml/src/covenant_ml/markets/crsp_cohort.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_cohort_records.py
  - libs/covenant_ml/tests/markets/test_crsp_cohort.py
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/request.json
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/preempted-input-evidence.json
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-preflight-report.json
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-submit.json
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/result.json
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-native-evidence.json
  - tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/ciphertext-collection.json
source_git_blobs:
  libs/covenant_ml/src/covenant_ml/markets/crsp_cohort.py: 872c57d646327f0eb2fd9a6dc31325276830b88a
  libs/covenant_ml/src/covenant_ml/markets/crsp_cohort_records.py: 9b342cd7bd1a30524af932d900ddb5bc04111f51
  libs/covenant_ml/tests/markets/test_crsp_cohort.py: 6717278e1b8fa58c8252d6edf1893338d86209e5
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/request.json: ba647e0bbb1fed9db6c64fc67d7989aa454f0310
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/preempted-input-evidence.json: 869b28d0615c22f5d5436317bf41f0c3fb3bac09
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-preflight-report.json: 92e625e80fbb93d22f4e81ace4e9dde26b560c4d
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-submit.json: def23228908a289bd9c600fecf514bdf118723b8
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/result.json: bf4b5280854c31a97d4b6fea1ac68d711c1eb782
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-native-evidence.json: bcd67522677a3c7cccdda8835f091c8669e77ace
  tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/ciphertext-collection.json: b9c632847496d1b6523645a5eb4411c910445b71
fact_checked: 2026-11-03
confidence: medium
---

# Encrypted CRSP diagnostic cohort

The cohort isolates source evidence for the 27 refused paths in
[[wrds-v3-refused-path-diagnosis]]. It derives observations from the already
frozen train-validation bundle; it is neither a fresh WRDS extraction nor a
replacement study dataset. The request pins the bundle, protocol and audit,
with limits of 27 cases, 40 million source rows and 100 million plaintext
bytes in the derived output.[^request][^owner]

## One transformation owner

`derive_crsp_cohort` authenticates the frozen bundle, verifies its partition,
resolves every audited entry-row digest inside private storage, and selects
the corresponding securities' membership, return and delisting rows. It keeps
the full calendar and exact original SQL/protocol evidence. The SQL describes
the original extraction, not a new query for the filtered files.[^owner]

The diagnostic envelope records the original and subset file identities,
unchanged source extraction time and a separate derivation time. It reuses
the shared row encoder, encryption, file hashing and evaluator key custody;
only the train-validation identity is acquired. Output directories are
exclusive, and metadata includes counts and digests without identifiers or
market observations.[^owner][^envelope]

## Admission is not permission to fit this subset

The distinct `crsp-diagnostic-cohort-v1` envelope is refused by the study
source loader. A regression passes an otherwise admitted encrypted cohort
to the actual `execute_trial` path and verifies refusal before an output
result exists. Other real-crypto tests cover changed identities/ciphertext,
final-partition refusal, budgets, existing destinations and plaintext removal
on normal exception unwinding.[^tests]

## Native evidence and limits

The first native attempt, `58790343`, verified all 17 inputs and was
`PREEMPTED` after 59 seconds. Its accounting exit field is `0:0`, but that
does not make its scheduler state successful. Input verification alone
establishes neither cohort completion nor forced-termination cleanup.[^preempted]

Explicit retry `58790450` uses the same pinned source `d37c6cd397`, image
`369ae415a20a`, command and free-partition request: 8 CPUs, 128 GB and 121
minutes. Its distinct attempt name preserves the failed attempt. Submission
and preflight are recorded.[^retry] It completed `0:0` in 573 seconds on
`hpc3-l18-01`; the existing inspector verified 17 inputs, result and RunRecord.
The result resolves 27 anchors to 23 securities: 95 membership rows, 72,981
return rows, 8,825 calendar rows and 19 terminal rows. It verifies all output
ciphertext and normal-exit plaintext removal. Bundle SHA-256 is
`182f9e58b6159750e39a469988bab5ea1256c2690559994ea2957effb35c6202`.[^completed]

Collection independently reread the exact twelve-file inventory against that
bundle pin: eleven ciphertext files totaling 5,047,538 bytes plus metadata.
The controller created no plaintext during collection.[^collection]

[[wrds-crsp-source-comparison]] records the completed bounded vendor check;
[[crsp-outcome-qualification]] retains the unresolved economic rules.
No result from this derivation permits removal
of the refused paths, a changed benchmark or release of the final holdout.[^owner][^tests]

[^request]: `tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/request.json`, `plan`, `source_commit` and source/program identities.
[^owner]: `libs/covenant_ml/src/covenant_ml/markets/crsp_cohort.py`, `_anchors`, `_securities`, `_write_subset` and `derive_crsp_cohort`.
[^envelope]: `libs/covenant_ml/src/covenant_ml/markets/crsp_cohort_records.py`, `decode_crsp_cohort` and `decode_crsp_cohort_plan`.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_cohort.py`, `test_actual_trial_refuses_a_cohort_even_when_its_bundle_is_admitted`, `test_real_encrypted_cohort_keeps_lineage_and_cannot_load_as_a_study`, and refusal regressions.
[^preempted]: `tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/preempted-input-evidence.json`, `accounting`, `inputs.compute_checksum_lines` and `inputs.verified_files`.
[^retry]: `tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-preflight-report.json`, `admissions[0].spec`; sibling `retry1-submit.json`, `job_id`, `experiment` and `image_digest`.
[^completed]: `tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/retry1-native-evidence.json`, `accounting`, `measurement` and `record`; sibling `result.json`, `results[0]`, input/code identities, derived counts, output bundle and cleanup assertions.
[^collection]: `tools/hpc3/artifacts/wrds-cohort-98379d28-20261102/ciphertext-collection.json`, `bundle_sha256`, `files_verified`, `ciphertext_bytes`, `plaintext_created` and `method`.
