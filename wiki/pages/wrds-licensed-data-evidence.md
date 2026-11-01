---
title: WRDS licensed CRSP extraction and encrypted staging
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json"
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/staging.json"
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/reconciliation.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json": "f69fefdc21501823201906b30f64122f04c4a931"
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/staging.json": "e65e100b5be0f5575f096653d0e4de61c6f9fcc4"
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/reconciliation.json": "cb08d7053e4363b23ce2740d54bb1823bc4801fb"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# WRDS licensed CRSP extraction and encrypted staging

The frozen four-relation dataset has been extracted, encrypted and collected.
Its train/validation bundle has been staged to HPC3. Extraction completeness
and byte identity do not guarantee that every eligible security has an
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
