---
title: WRDS installed controller and compute runtime
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json"
  - "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json": "d54e268ee85ba1691c10cead64e501f1814f1f94"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json": "b241815ebdf05dc49c92a4b6a786df80a230c96d"
  "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json": "3850961553a9736d1533616f23ea2b534ac712e1"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# WRDS installed controller and compute runtime

The last verified study runtime is source `b3843a3d3`, used by both the
synthetic summary rehearsal and the stopped licensed baseline. Repository
changes do not update these installed bytes. Current study status is in
[[wrds-baseline-readiness]].[^runtime][^preparation]

## Installed identities

Controller inspection and native image inspection recorded these identities.[^runtime]

| Component | Verified identity |
|---|---|
| Controller interpreter | `/home/corvis/research-control/wrds-summary-b3843a3d3/bin/python` |
| Controller source | `b3843a3d3b83cbae77dc4f855be2c4fbaa17b492` |
| HPC3 image | `/data/homezvol1/wagnera3/research-images/summary-b3843a3d3/research-summary.sif` |
| Image SHA-256 | `14b1b5c929b3e89f62edae19cb04714e66c1d122273296e6d532559d717f8f4e` |

Controller inspection verified 585 installed files against three wheels.
Native image build `58719937` completed with exit `0:0`; inspection verified
14 image inputs and two self-check passes. These measurements establish runtime
identity, not valid scientific outcomes.[^runtime]

## Allocation and execution ownership

The licensed attempt requested four CPUs, 128 GiB and 121 scheduler minutes
on `free`, with a 7,200-second evaluator deadline. These are frozen per-job
ceilings, not the account's total available compute. Native allocation can
differ from the CPU request; retain accounting alongside the recipe.[^preparation]

The shared queue uses the installed `wrds-queue-b53b0ea34` adapter.
A fresh synthetic request `eecf34ae` completed as native job `58719001`
and was reconciled by `fleet-research-diphtheria`. Adapter identity and
evaluator-image identity are separate provenance fields.[^controller]

## Changing the runtime

A new evaluator requires a separate frozen controller/image, staging readback
and an operational rehearsal before a new study. The original failed journal,
policy and image remain the record of [[wrds-baseline-20261101]].
Do not describe newly committed code as already installed.[^runtime][^preparation]

[^runtime]: `tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json`, `controller`, `image`, `construction`, `service_reports` and `scope`.
[^preparation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json`, frozen runtime, policy and first-attempt preparation.
[^controller]: `tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json`, `queue.observation` and `scope`.
