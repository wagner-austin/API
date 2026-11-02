---
title: WRDS installed controller and compute runtime
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json"
  - "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json": "4dc9f9b11c5077f59a7122497a019faa1be8cc0b"
  "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json": "a80e656edb6325abd242bf2f77ddbfca17938434"
  "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json": "3ad95ee9f98aa38ccd89f44a24a1e0c8a893a9d9"
  "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json": "d54e268ee85ba1691c10cead64e501f1814f1f94"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json": "b241815ebdf05dc49c92a4b6a786df80a230c96d"
  "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json": "3850961553a9736d1533616f23ea2b534ac712e1"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json": "3b52554a69ce12c549fc47b3cc60d2c332299fa7"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# WRDS installed controller and compute runtime

The separately installed version-3 runtime candidate uses source `087f67b4c`,
including exact missing-price return spans. Controller and native image
inspections passed. Its queue rehearsal and licensed qualification remain
outstanding; the last completed queue rehearsal used version 2. Current study
status is in [[wrds-baseline-readiness]].[^candidate][^v2]

## Installed identities

Candidate controller and native image inspection recorded these identities.[^candidate]

| Component | Verified identity |
|---|---|
| Controller interpreter | `/home/corvis/research-control/wrds-protocol-v3-087f67b4c/bin/python` |
| Controller source | `087f67b4c4aae9a731c913330840dcc665b51701` |
| HPC3 image | `/data/homezvol1/wagnera3/research-images/protocol-v3-087f67b4c/research-crsp-v3.sif` |
| Image SHA-256 | `369ae415a20a1ad8a7d672d839e12c7e62ee29d1432667d95f479d333c578fde` |

Candidate controller inspection verified 624 installed files against seven
wheels, including both shared libraries. Native image build `58742562` completed
in 95 seconds with exit `0:0`; inspection verified fourteen pinned files and
two self-check passes. Both dependency checks passed. Capture retained the 53
exact dependencies; the generated self-check requires 24 symbols. No licensed
observations or keys were included in the image.[^candidate]

The preserved version-2 controller/image at source `8ef7e0f17` belong to the
stopped second study and its completed rehearsal. Their inspection verified
623 installed members and native build `58739821`; they were not replaced or
relabeled by candidate installation.[^current][^v2]

The historical `b3843a3d3` controller and image remain the runtime of the
stopped licensed attempt. Its inspections verified 585 installed files,
three wheels and native build `58719937`. Those identities are preserved,
not relabeled as the corrected runtime.[^runtime]

## Allocation and execution ownership

The stopped licensed attempt requested four CPUs, 128 GiB and 121 scheduler minutes
on `free`, with a 7,200-second evaluator deadline. These are per-job requests,
not the account's total available compute.[^preparation]

Native accounting records **eight allocated CPUs**, despite the four-CPU
recipe and `ReqTRES`. Partition inspection reports 18 GiB maximum memory per
CPU; rounding 128/18 upward gives eight. This is consistent with memory-driven
allocation adjustment, although the site-specific adjustment was not traced.
Four CPUs was not an enforced allocation ceiling. Preserve that discrepancy
and freeze a feasible CPU/memory pair for the next study.[^allocation]

The shared queue uses the installed `wrds-queue-b53b0ea34` adapter.
Historical synthetic request `eecf34ae` completed as native job `58719001`.
The current rehearsal used queue `f1961f01` and native job `58740021`,
automatically reconciled by `fleet-research-diphtheria`; its immutable
specification selected the new evaluator interpreter. Adapter identity and
evaluator-image identity are separate provenance fields.[^controller][^v2]

## Changing the runtime

The version-3 candidate still needs independent review, its own synthetic queue
rehearsal and licensed return-span qualification before a newly frozen study.
Installation is not a rehearsal or substantive result. No licensed decryption,
study launch or final grant release occurred during installation. Historical
journals, policies and images remain unchanged.[^candidate]

[^candidate]: `tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json`, `controller`, `image`, `construction`, `service_reports`, `source_checks`, `remaining` and `scope`; separate candidate installation, with no version-3 study execution.

[^current]: `tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json`, `controller`, `image`, `service_reports`, `construction`, `historical_state`, `remaining` and `scope`.
[^runtime]: `tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json`, `controller`, `image`, `construction`, `service_reports` and `scope`.
[^preparation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json`, frozen runtime, policy and first-attempt preparation.
[^controller]: `tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json`, `queue.observation` and `scope`.
[^allocation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json`, `allocation`, exact `sacct_row`, partition limit and bounded interpretation.

[^v2]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json`, `scope`, `fixture`, `queue`, `native_accounting`, `native_input_verification`, `result`, `pre_entry_case`, `service_reports` and `remaining`.
