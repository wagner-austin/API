---
title: Encrypted CRSP protocol metadata revision
tags: [wrds, research, encryption, provenance]
hubs: [libs]
related: ["[[wrds-licensed-data-evidence]]", "[[wrds-baseline-readiness]]"]
source_paths:
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py"
  - "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py"
  - "tools/hpc3/artifacts/wrds-protocol-revision-20261102/evidence.json"
source_git_blobs:
  "libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py": "9831dcb7384eafdf97f5a215890a790946f79b7d"
  "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py": "0fa13bb7ae218f95730b91b3968ae2be8d9d9388"
  "tools/hpc3/artifacts/wrds-protocol-revision-20261102/evidence.json": "bd6ac340e20ce528c5a4ad2571714d69cd2fad34"
fact_checked: 2026-11-01
confidence: high
---

# Encrypted CRSP protocol metadata revision

Both licensed partitions have separately verified version-2 successors. All
sixteen CSV/SQL ciphertext files retain their exact original identities;
only protocol and manifest metadata was replaced using public encryption
recipients. No private key was acquired and no observation was opened.[^measured]

## Contract

The existing extraction CLI exposes `revise-protocol` and `verify-revision`.
Both pin the original extraction receipt and protocol, require identical
extraction settings, reconstruct the exact original manifests, and verify
original and revised ciphertext. Only version, entry and cost rules may differ.
The revision is explicit; the evaluator does not accept the old protocol.[^revision][^measured]

The storage owner reuses its encryption implementation for the two replacement
metadata files and copies retained ciphertext into an exclusive destination.
It verifies copied bytes and writes each bundle manifest last. The revision
receipt is published only after both partitions finish. An interrupted write
may leave incomplete ciphertext, but no complete revision receipt.[^storage][^revision]

## Licensed measurement

Published source `b69daf97a0747732787d2036694b2c8d84e48c06` ran on Ziggurat.
The creation report and separate verifier invocation produced identical reports
at 00:20:07 and 00:20:45 UTC on November 2, 2026. Both were posted by
`load-report-9a37c9a5`; full report IDs and hashes are in the evidence artifact.
The source suite passed 5,180 tests at 100% coverage.[^measured]

| Identity | SHA-256 |
|---|---|
| Original extraction receipt | `18bd2028207c56784df784916728b62800b7bc3c6a89ea5b1f5970c7c28f9349` |
| Current protocol | `66547bc7f4d8f942116ae794494257e4dabd01428b5da7a4361b46661a79cf2b` |
| Revised train/validation bundle | `ea669b73b777897345465762e653a91ee33696a5f1bd150b936dc1a27ad22f83` |
| Revised final bundle | `8c753ae372b7e640b797375d79118ffb961a5bf27cec82f1766010e9e10ece85` |
| Revision receipt | `ddb0ad455e91fccb358b2479c2f6a5ab218a0e2dd056b6bfe429f40c56a537eb` |

The extraction vintage remains `2026-10-31T18:44:04.122752+00:00`. The
revision root is `/home/corvis/research-data/wrds/experiments/crsp-protocol-v2-b69daf97a-20261102`.
Original bundles, receipt and failed-study journal remain historical.[^measured]

## Readiness boundary

This is a metadata transformation, not a fresh extraction, installed study
runtime or successful baseline. Invocation used the existing interpreter with
explicit checked-source imports; it did not replace that installed controller.
The revised validation bundle still needs HPC3 staging and the current
controller/image needs installation and queue rehearsal. Final ciphertext
remains separate and unreleased. See [[wrds-baseline-readiness]].[^measured]

[^revision]: `libs/covenant_ml/src/covenant_ml/markets/crsp_revision.py`, `_inputs`, `revise_crsp_protocol` and `verify_crsp_revision`; current protocol validation, original manifest reconstruction, receipt publication and ciphertext comparison.
[^storage]: `libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py`, `revise_encrypted_bundle`; exclusive destination, shared encryption, copied-byte verification and final manifest publication.
[^measured]: `tools/hpc3/artifacts/wrds-protocol-revision-20261102/evidence.json`, `execution`, `service_reports`, `measurement`, `verification` and `limits`; program reports bind original/new metadata and unchanged ciphertext, without reading observations.
