---
title: WRDS licensed CRSP extraction and encrypted staging
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json"
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/staging.json"
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/reconciliation.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
  - "tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json": "f69fefdc21501823201906b30f64122f04c4a931"
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/staging.json": "e65e100b5be0f5575f096653d0e4de61c6f9fcc4"
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/reconciliation.json": "cb08d7053e4363b23ce2740d54bb1823bc4801fb"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
  "tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json": "1100c7de989b5f9fb98273fd26e16477db17637a"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-02
confidence: medium
---

# WRDS licensed CRSP extraction and encrypted staging

The frozen four-relation dataset has been extracted, encrypted and collected.
Its original, version-2 and version-3 validation bundles have separate verified
HPC3 destinations; [[wrds-protocol-revision]] records their metadata lineage.[^v2][^v3-preparation]
Extraction completeness and byte identity do not guarantee that every eligible security has an
executable holding path; [[wrds-baseline-20261101]] records that distinction.[^collection][^staging][^stopped]

## Extraction identity

WRDS native job `40554174` read `crsp_a_stock` in one recorded transaction
on October 31, 2026. The receipt binds the protocol, SQL driver digest, snapshot,
timestamps and separate partition manifests. Relations are membership, daily
returns, exchange calendar and delistings.[^collection]

| Partition | Observation end | Daily return rows | Bundle SHA-256 |
|---|---|---:|---|
| Train/validation | 2021-12-31 | 28,034,527 | `26b46d56c5ab47565813981f0eb4dabad4e0d5260c86dd02cf45becbe6c858f1` |
| Final | 2025-02-28 | 31,465,356 | `377b316ee84d185853412776b63ec8ccb6da7ecd0cbe4d6293bfa703ca0c226e` |

The train/validation calendar extends through February 2022 for window
construction; its security observations stop in 2021. The final bundle remains
separate and unreleased for the stopped baseline.[^collection][^stopped]

## Collection and staging

Collection recovery reused the original native extraction. The evidence
preserves the interrupted collection and its deadline change rather than
claiming a new extraction. Reconciliation checks read the service-authored
record, installed controller and output identities; negative probes change
those identities and must fail.[^collection][^reconciliation]

HPC3 staging verified 11 encrypted bundle files at
`/data/homezvol1/wagnera3/research-data/crsp-licensed-calendar-20261031/train-validation`.
The staging receipt binds each filename, byte count and SHA-256 to the
extraction identities. SQL, manifests and data travel encrypted; keys are
separate from image and bundle.[^staging]

The historical version-2 validation bundle is staged separately at
`/data/homezvol1/wagnera3/research-data/crsp-protocol-v2-1fe61b29/train-validation`.
The existing controller was re-inspected before transfer. A separate complete
readback verified all eleven files plus certificate, no extra entries, and
mode-0700 root and partition directories. The 1,789,968,348 bundle bytes contain
no private keys or final ciphertext; this is staging, not evaluator installation
or a study result.[^v2]

The current version-3 validation bundle is staged at
`/data/homezvol1/wagnera3/research-data/crsp-protocol-v3-f61db3a7/train-validation`.
Complete readback verified all eleven bundle files, the certificate, exact
inventory and mode-0700 directories. The first inspection refused newly created
0755 directories; correcting those study directories and repeating the full
inspection passed. No final ciphertext or keys were staged. Its exact prior
failed training interval then passed native qualification; this does not prove
all eligible intervals are executable.[^v3-preparation]

## Limits and related contracts

The extraction receipt reports plaintext removal, and a later same-node
observer found no matching scratch roots. That is evidence for this WRDS
allocation, not for every subsequent HPC3 job; see [[wrds-cleanup-evidence]].
The table/flag contract remains in [[crsp-monthly-data-contract]], and historical
availability assumptions remain in [[crsp-monthly-study-design]].[^collection]

[^collection]: `tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json`, `extraction`, `collection_recovery`, `collection` and `cleanup_observation`.
[^staging]: `tools/hpc3/artifacts/wrds-licensed-collection-20261031/staging.json`, staging manifest, completion receipt, file identities and `audit_query`.
[^reconciliation]: `tools/hpc3/artifacts/wrds-licensed-collection-20261031/reconciliation.json`, controller measurement, live audit query and negative probes.
[^stopped]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `diagnostic` and `study_action`.
[^v2]: `tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json`, `controller`, `staging`, `readback.inventory`, `stage_bytes` and `service_reports`; three program-written readings distinguish installed-controller verification, transport and complete remote inspection.

[^v3-preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `rehearsal_audit`, `revision`, `staging`, `qualification`, `freeze`, `queue_submission` and `limits`; linked program reports retain exact inputs and observed boundaries.
