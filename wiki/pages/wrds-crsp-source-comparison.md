---
title: Bounded encrypted CRSP source comparison
tags: [wrds, crsp, encryption, provenance]
hubs: [libs]
related: ["[[wrds-encrypted-diagnostic-cohort]]", "[[crsp-outcome-qualification]]", "[[wrds-baseline-readiness]]"]
source_paths:
  - libs/covenant_ml/src/covenant_ml/markets/crsp_refresh.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_refresh_records.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_snapshot.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_comparison.py
  - libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py
  - libs/covenant_ml/scripts/crsp_extraction.py
  - libs/covenant_ml/tests/markets/test_crsp_refresh.py
  - libs/covenant_ml/tests/markets/test_crsp_refresh_process.py
  - libs/covenant_ml/tests/markets/test_crsp_comparison.py
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/source.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/failed-first.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-comparison.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-completion.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-collection.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-mutations.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/changed-absent-comparison.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/changed-absent-completion.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/changed-absent-collection.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/requests.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-comparison.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-completion.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-collection.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-stage.json
  - tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-stage-inspection.json
source_git_blobs:
  libs/covenant_ml/src/covenant_ml/markets/crsp_refresh.py: 11e16545808e5aa1ec05c5ca1bdeb858eceff88c
  libs/covenant_ml/src/covenant_ml/markets/crsp_refresh_records.py: 8d8e94ed142d6b4badb23faf9b0fdd9ae48e8d68
  libs/covenant_ml/src/covenant_ml/markets/crsp_snapshot.py: 1e454f61a3b50a031c92803ebf3c54cc0343c4d1
  libs/covenant_ml/src/covenant_ml/markets/crsp_comparison.py: 2dc5d21a784e30c843e5e3857a615e03751f1141
  libs/covenant_ml/src/covenant_ml/markets/crsp_rows.py: b10994304c31e020bab370b7eb0f6fe2fd948b39
  libs/covenant_ml/scripts/crsp_extraction.py: 590e709a6a702ce5143fd84d692f9b36a1ad91af
  libs/covenant_ml/tests/markets/test_crsp_refresh.py: 37b284c4cdd68cd694544633f5f20a1527fed899
  libs/covenant_ml/tests/markets/test_crsp_refresh_process.py: 2afaf62b705e4f28b0807a2681381bb785a15a57
  libs/covenant_ml/tests/markets/test_crsp_comparison.py: eb978fb83e99e545de690e98d38f8f3e070bad96
  tools/hpc3/artifacts/wrds-source-refresh-20261102/source.json: 4f1d9f75bc21dbd8b725fa9436934d3925e6207f
  tools/hpc3/artifacts/wrds-source-refresh-20261102/failed-first.json: b2c92163502d0ba5dd44f01a4b999a77dd0ca026
  tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-comparison.json: d42193ba73edfe3cda505b7701cbd9fd9130ba19
  tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-completion.json: c2954438dcaeb235423e4c0608b163b4712775d5
  tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-collection.json: fedb72beea45d2c18e1f395e317c50c26362b59a
  tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-mutations.json: 63e5234c95018a15c1705bafef17ca69364dfd15
  tools/hpc3/artifacts/wrds-source-refresh-20261102/changed-absent-comparison.json: fde5bd57f249dad6e2b3852e1cc6c3378c87e42e
  tools/hpc3/artifacts/wrds-source-refresh-20261102/changed-absent-completion.json: 3688a1af5ef4f158ca0afeda69024864eb481615
  tools/hpc3/artifacts/wrds-source-refresh-20261102/changed-absent-collection.json: 7bc629e2c5853dce99ceb380269ad040451a04b5
  tools/hpc3/artifacts/wrds-source-refresh-20261102/requests.json: 389043de908378aa8a6434509d246f5a51848b16
  tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-comparison.json: e009d0150a6c6e4f3a84ce84890c7aae780206ef
  tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-completion.json: c04eb274699e84e429846f3cbf3a4e433d0d7e4c
  tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-collection.json: 85fec6da6bdeb1ec9c72fe33fa196496c5dcb3a0
  tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-stage.json: 0f28df6757d073fc7efc9f0612d0f33b872a6912
  tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-stage-inspection.json: 1670a63dc99b18eb429d28904083894b50c81f85
fact_checked: 2026-11-02
confidence: high
---

# Bounded encrypted CRSP source comparison

## Diagnostic contract

The existing extraction CLI's `refresh` command opens the authenticated
[[wrds-encrypted-diagnostic-cohort]] in private scheduler storage. It queries
only that cohort's securities within the frozen train-validation bounds and
shares query preparation, snapshot execution and manifest sealing with the
partition writer. Exact SQL, transaction readings, source rows and differences
are encrypted; public outputs contain identities, timestamps and counts.[^owner]

The comparison shares natural keys with the source decoder. It separates
unchanged, added, removed and changed rows, rejects duplicate keys and
preserves null versus zero and exact numeric differences regardless of row
order. A key edit is a removal and addition. The separate
`crsp-diagnostic-refresh-v1` envelope binds old and fresh evidence and cannot
load as a study dataset. No final identity is acquired.[^comparison][^envelope]

Tests exercise real encryption, changed/absent observations, malformed inputs,
actual client failure, file limits and timeout cleanup. The process tests use
controlled CSV output; native database execution is separate evidence below.
Normal-unwind cleanup does not prove cleanup after forced termination.[^tests]

## Native evidence

The first synthetic job, `40568113`, failed before database access. Its source
snapshot combined current `platform_core` with an older installed `platform_ml`
that imported a removed module. The replacement freezes five first-party
packages and the existing CLI (501 Python files) at API `88ad03e16960fe84e0b912b73c940245e74770f6`.
It preserves the failed attempt and uses a new request identity.[^snapshot]

Synthetic job `40568160` completed and its collected outputs verified. It
queried real PostgreSQL using synthetic inline relations: 1 membership row,
1,042 returns, 1,521 calendar rows and zero delistings matched the frozen
cohort exactly. Its public receipt reports normal plaintext cleanup; all 16
encrypted files were verified without local decryption. This is a plumbing
check, not a licensed-data or economic result.[^synthetic]

Synthetic job `40568175` changes one return to null and removes the next
observation. The native comparison records exactly one changed and one removed
return, 1,040 unchanged returns, and no other differences. Its 16 ciphertext
files and normal plaintext removal verified.[^changed]

Licensed job `40568218` completed the bounded vendor comparison. All 81,920
rows were unchanged: 95 membership rows, 72,981 returns, 8,825 calendar rows
and 19 delistings, with zero additions, removals or changed observations. The
15 ciphertext files verified; the receipt reports normal plaintext removal.
This establishes no fresh-vintage correction within the frozen projection and
bounds. The evidence covers only that projection and those bounds; the 27
economic refusals remain unresolved.[^licensed]

The existing HPC3 staging CLI verified all 15 ciphertext files and the bundle
manifest against the WRDS completion record. Its independent readback found
exactly those files plus the certification record under a mode-0700 directory,
with no extra entries. No plaintext or key was staged.[^stage]

The substantive study remains stopped. [[crsp-outcome-qualification]] owns the
unresolved economic requirements; the diagnostic does not authorize fitting,
new study admission or final-scoring release.[^envelope]

[^owner]: `libs/covenant_ml/scripts/crsp_extraction.py`, `main` refresh branch; `libs/covenant_ml/src/covenant_ml/markets/crsp_refresh.py`, `compare_crsp_vintages`, `_cohort_sources` and `_refresh`; `crsp_snapshot.py`, `prepare_crsp_queries`, `execute_crsp_snapshot` and `seal_crsp_manifest`.
[^comparison]: `libs/covenant_ml/src/covenant_ml/markets/crsp_comparison.py`, `compare_crsp_rows` and `_indexed`; `crsp_rows.py`, `crsp_row_key`; `libs/covenant_ml/tests/markets/test_crsp_comparison.py`, key, ordering, duplicate and empty-input cases.
[^envelope]: `libs/covenant_ml/src/covenant_ml/markets/crsp_refresh_records.py`, `decode_crsp_refresh`; `libs/covenant_ml/tests/markets/test_crsp_refresh.py`, encrypted differences, absent final key and actual-study rejection.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_refresh.py`, successful encryption and refusal cases; `test_crsp_refresh_process.py`, CLI observations and `test_real_client_failures_remove_authenticated_plaintext_without_leaking_sql`.
[^snapshot]: `tools/hpc3/artifacts/wrds-source-refresh-20261102/failed-first.json`, accounting, payload traceback and service-report identity; sibling `source.json`, commit, archive digest and package inventory.
[^synthetic]: `tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-comparison.json`, row counts, bundle digest and `plaintext_removed`; sibling `synthetic-completion.json`, native job, request, runtime, output digests and RunRecord observations. Sibling `synthetic-collection.json` records collection and ciphertext-only verification. Program-authored board reading `4c7e854b-fd54-4cc2-83bc-b3e778bc207d` records terminal success and attestation checks.
[^changed]: `tools/hpc3/artifacts/wrds-source-refresh-20261102/synthetic-mutations.json`, exact fixture digests and expected counts; sibling `changed-absent-comparison.json`, observed counts and cleanup, `changed-absent-completion.json`, native identity and output hashes, and `changed-absent-collection.json`, ciphertext verification.
[^licensed]: `tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-comparison.json`, all relation counts, bundle digest and cleanup; sibling `licensed-completion.json`, native request and attested outputs, `licensed-collection.json`, fifteen verified ciphertext files, and `requests.json`, resource, input and output declarations. Program-authored board reading `2f34aa0d-03ba-4c28-8693-2aa8fd415b20` records terminal success and attestation checks.
[^stage]: `tools/hpc3/artifacts/wrds-source-refresh-20261102/licensed-stage.json`, `manifest`, `expected_sha256` and `verified_files`; sibling `licensed-stage-inspection.json`, complete `inventory` and certificate digest.
