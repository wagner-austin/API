---
title: Encrypted CRSP protocol metadata revision
tags: [wrds, research, encryption, provenance]
hubs: [libs]
related: ["[[wrds-licensed-data-evidence]]", "[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "libs/covenant_ml/tests/markets/test_crsp_revision.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py"
  - "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py"
  - "tools/hpc3/artifacts/wrds-protocol-revision-20261102/evidence.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "libs/covenant_ml/tests/markets/test_crsp_revision.py": "3db03bce47865acba6d75416a1308e2710726d49"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py": "0c6c56f05f23df1a239492e39798f935d47d6fdf"
  "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py": "7e9ebdafddf540613819fc1310f9aaf0dd05602d"
  "tools/hpc3/artifacts/wrds-protocol-revision-20261102/evidence.json": "bd6ac340e20ce528c5a4ad2571714d69cd2fad34"
fact_checked: 2026-11-03
confidence: high
---

# Encrypted CRSP protocol metadata revision

Both licensed partitions have separately verified version-3 successors. All
sixteen CSV/SQL ciphertext files retain their exact original identities;
only protocol and manifest metadata was replaced using public encryption
recipients. No private key was acquired and no observation was opened during revision.[^v3-preparation]

## Contract

The existing extraction CLI exposes `revise-protocol` and `verify-revision`.
Both pin the original extraction receipt and protocol, require identical
extraction settings, reconstruct the exact original manifests, and verify
original and revised ciphertext. Only version, entry, cost, held-return and terminal rules may differ.
The revision is explicit; the evaluator does not accept the old protocol.[^revision][^measured]

The revision validator admits explicit entry, held-return and terminal rule
changes, shared by creation and independent verification; the terminal rule was
added for prospective version 5 ([[crsp-v5-boundary-policy]]), which changes no
extraction column or date. Current real-file regressions exercise the version-3
to prospective version-5 amendment, all four rules at once, without acquiring
keys, preserve all sixteen relation/SQL ciphertext identities and reject changed
extraction settings through both entry points. The separate licensed version-3
measurement below now establishes metadata lineage; runtime identity remains a
different measurement. No licensed version-4 or version-5 revision or
replacement study is established by those synthetic regressions.[^tests][^v3-preparation]

[^tests]: `libs/covenant_ml/tests/markets/test_crsp_revision.py`, `test_metadata_revision_preserves_originals_and_needs_no_keys` and `test_accounting_amendment_cannot_change_extraction_settings`; real encrypted files, synthetic observations, key-free revision and immutable extraction settings.

The storage owner reuses its encryption implementation for the two replacement
metadata files and copies retained ciphertext into an exclusive destination.
It verifies copied bytes and writes each bundle manifest last. The revision
receipt is published only after both partitions finish. An interrupted write
may leave incomplete ciphertext, but no complete revision receipt.[^storage][^revision]

## Licensed measurement

The frozen `087f67b4c` controller invoked creation and separate verification
on November 2 at 06:48:21 and 06:48:59 UTC. The reports are identical and
service-authored posts preserve their payload digests. This revision starts
from the original extraction, retaining sixteen relation/SQL ciphertexts;
it changes only protocol and manifest metadata.[^v3-preparation]

| Identity | SHA-256 |
|---|---|
| Original extraction receipt | `18bd2028207c56784df784916728b62800b7bc3c6a89ea5b1f5970c7c28f9349` |
| Current protocol | `af5cbced56bc26076db0a3dc769c98c45ba393a507473a18eda839e5ddf61ed9` |
| Revised train/validation bundle | `d8e66902e452c8f0e8790d3d17e31c648b684fbfa2606bea7de363889f94fa6d` |
| Revised final bundle | `4c8382a7e72e0ddbff8ac6db6b5f76212d1c94cdc42c714860f6f86f322ad57e` |
| Revision receipt | `0f4c7aeba291707b6787ce0d71729754ed4a9dbdc2fc0c455aee6f122f64022f` |

Extraction vintage remains `2026-10-31T18:44:04.122752+00:00`. Original
and version-2 bundles and failed-study journals remain historical. The
version-2 revision evidence remains available at its original path.[^v3-preparation][^measured]

## Readiness boundary

Revision itself reads no observations and proves no investment performance.
Subsequent encrypted validation staging is recorded in
[[wrds-licensed-data-evidence]], and the limited licensed qualification and
new study freeze in [[wrds-baseline-v3-freeze]]. Final ciphertext remains
separate and unreleased.[^v3-preparation]

[^revision]: `libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py`, `_inputs`, `revise_crsp_protocol` and `verify_crsp_revision`; current protocol validation, original manifest reconstruction, receipt publication and ciphertext comparison.
[^storage]: `libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py`, `revise_encrypted_bundle`; exclusive destination, shared encryption, copied-byte verification and final manifest publication.
[^measured]: `tools/hpc3/artifacts/wrds-protocol-revision-20261102/evidence.json`, `execution`, `service_reports`, `measurement`, `verification` and `limits`; program reports bind original/new metadata and unchanged ciphertext, without reading observations.

[^v3-preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `rehearsal_audit`, `revision`, `staging`, `qualification`, `freeze`, `queue_submission` and `limits`; linked program reports retain exact inputs and observed boundaries.
