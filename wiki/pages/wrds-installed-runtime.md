---
title: WRDS installed controller and compute runtime
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json"
  - "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json"
source_git_blobs:
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

The current separately installed controller and HPC3 image use source
`8ef7e0f17`, including the audited version-2 CRSP corrections. Installation
passed its service inspections. A complete queue rehearsal with this runtime
and a successful substantive study remain outstanding. Current study status
is in [[wrds-baseline-readiness]].[^current]

## Installed identities

Controller inspection and native image inspection recorded these identities.[^current]

| Component | Verified identity |
|---|---|
| Controller interpreter | `/home/corvis/research-control/wrds-protocol-v2-8ef7e0f17/bin/python` |
| Controller source | `8ef7e0f1769e7127feabb30008d72bc70e908111` |
| HPC3 image | `/data/homezvol1/wagnera3/research-images/protocol-v2-8ef7e0f17/research-crsp-v2.sif` |
| Image SHA-256 | `e1560c685864bf2ccc012c49f475f90141e0804435019286ca08c1ffbb2acc1e` |

Controller inspection verified 623 installed files against seven wheels,
including both shared libraries. Native image build `58739821` completed in
88 seconds with exit `0:0`; inspection verified fourteen pinned files and
two self-check passes. Both dependency checks passed.[^current]

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
A fresh synthetic request `eecf34ae` completed as native job `58719001`
and was reconciled by `fleet-research-diphtheria`. Adapter identity and
evaluator-image identity are separate provenance fields.[^controller]

## Changing the runtime

The corrected evaluator is installed separately. Amended study resources and
a complete synthetic queue rehearsal remain required before a new licensed
study. Installation is not a queue rehearsal or a substantive study result;
no final grant release occurred during installation. The original failed
journal, policy and image remain the record of [[wrds-baseline-20261101]].[^current][^preparation]

[^current]: `tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json`, `controller`, `image`, `service_reports`, `construction`, `historical_state`, `remaining` and `scope`.
[^runtime]: `tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json`, `controller`, `image`, `construction`, `service_reports` and `scope`.
[^preparation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json`, frozen runtime, policy and first-attempt preparation.
[^controller]: `tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json`, `queue.observation` and `scope`.
[^allocation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json`, `allocation`, exact `sacct_row`, partition limit and bounded interpretation.
