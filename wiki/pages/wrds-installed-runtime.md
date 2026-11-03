---
title: WRDS installed controller and compute runtime
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/source-scope-report.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json"
  - "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v5-runtime-20261103/evidence.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json": "fe2f9f8774c053a9e07826ace70b9187ae9b0ef0"
  "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/source-scope-report.json": "bb969cf2a9f3f4048e33df6cc1067f6cf21aa16c"
  "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json": "4dc9f9b11c5077f59a7122497a019faa1be8cc0b"
  "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json": "a80e656edb6325abd242bf2f77ddbfca17938434"
  "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json": "3ad95ee9f98aa38ccd89f44a24a1e0c8a893a9d9"
  "tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json": "d54e268ee85ba1691c10cead64e501f1814f1f94"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json": "b241815ebdf05dc49c92a4b6a786df80a230c96d"
  "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json": "3850961553a9736d1533616f23ea2b534ac712e1"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json": "3b52554a69ce12c549fc47b3cc60d2c332299fa7"
  "tools/hpc3/artifacts/wrds-protocol-v5-runtime-20261103/evidence.json": "6c1a4474a495bdbd5dd66b0b50c78e180281cdcf"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-03
confidence: medium
---

# WRDS installed controller and compute runtime

The separately installed version-3 runtime candidate uses source `087f67b4c`,
including exact missing-price return spans. Controller and native image
inspections and independent review passed. Its synthetic queue rehearsal
completed as job `58747974` and passed independent review. Licensed training-span
qualification also passed as job `58748788`; this checks the previously failed
interval, not the whole dataset. Current study status is in
[[wrds-baseline-readiness]].[^candidate][^v3][^v3-preparation]

## Version-5 runtime

A separate version-5 runtime was installed on November 3 from API
`cfd829d03` for the native full-panel qualification; it replaces nothing above.
Its controller and native image were inspected, and it has not been
independently audited or used for a study.[^v5]

| Component | Verified identity |
|---|---|
| Controller interpreter | `/home/corvis/research-control/wrds-protocol-v5-cfd829d03/bin/python` |
| Controller source | `cfd829d03590ea57c247057e27a48510ab2b2078` |
| HPC3 image | `/data/homezvol1/wagnera3/research-images/protocol-v5-cfd829d03/research-crsp-v5.sif` |
| Image SHA-256 | `a266a8b3bf6bae74b3a390af16b0ba439782f35bd2685b4a5d5287fec03cfb92` |

The controller inspection verified 650 installed files against seven wheels;
six pure-Python wheels were built on Loki from the exported commit, and the
compiled Rust wheel and the 53 dependency pins were reused because neither
changed since `087f67b4c`. Native image build `58820002` completed in 94
seconds with exit `0:0`; inspection verified fourteen files and two self-check
passes. The image spec's self-check names the version-5 resolvers and the
qualification entry points in place of a symbol version 5 removed.[^v5]

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

The follow-up source measurement compared all **622 application members** in
the six Python wheels with their paths in the exact published source archive.
All matched. The retained Rust wheel matched the previous binary, and all
224 files in its source tree matched the previous frozen commit. This verifies
source equality and binary identity; it is not a reproducible Rust rebuild or
a retroactive build-time attestation.[^source-scope]

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
The version-2 rehearsal used queue `f1961f01` and native job `58740021`,
automatically reconciled by `fleet-research-diphtheria`; its immutable
specification selected the new evaluator interpreter. Adapter identity and
evaluator-image identity are separate provenance fields.[^controller][^v2]

## Changing the runtime

Independent runtime audit `29717b73` passed on November 2 at 06:23 UTC.
Queue `b45ef01a` then selected this exact evaluator and completed native job
`58747974`; [[wrds-operational-rehearsals]] records the result and its limits.
The subsequent licensed qualification and new freeze are recorded in
[[wrds-baseline-v3-freeze]].[^v3-preparation]
No licensed decryption, study launch or final grant release was requested
during installation.[^candidate][^v3]

The follow-up scope measurement read 63 native ledger entries and validated
277 events across all thirteen journals found under the WRDS experiment root.
During the installation window, 03:36:25–03:45:36 UTC on November 2, the only
submission by this session was image build `58742562`, and no journal event
occurred. Both retained release metadata files predate that window. The
candidate runtime directory contained no journal or grant metadata. Journal
and ledger hashes were unchanged across the reading. These are measurements
of retained records, not a historical filesystem-access audit.[^source-scope]

[^v5]: `tools/hpc3/artifacts/wrds-protocol-v5-runtime-20261103/evidence.json`, `source_export`, `controller`, `image`, `service_reports` and `remaining`.
[^candidate]: `tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json`, `controller`, `image`, `construction`, `service_reports`, `source_checks`, `remaining` and `scope`; separate candidate installation, with no version-3 study execution.
[^v3]: `tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json`, `prerequisite_audits`, `controller_python`, `image_sha256`, `queue`, `native_accounting` and `remaining`; subsequent synthetic execution is separate from installation and licensed qualification.

[^source-scope]: `tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/source-scope-report.json`, `python_wheels`, `source_members`, `retained_rust`, `submissions_in_window`, `journals`, `release_metadata` and `scope`; service post `0873ab5e-a859-4ae5-bf15-18a9780add22`, measured at 06:11:25 UTC. The adjacent measurement program and member manifests are bound by SHA-256 in this report; the existing archive exporter, native ledger reader and shared event-journal reader performed the reads.

[^current]: `tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json`, `controller`, `image`, `service_reports`, `construction`, `historical_state`, `remaining` and `scope`.
[^runtime]: `tools/hpc3/artifacts/wrds-summary-runtime-20261101/evidence.json`, `controller`, `image`, `construction`, `service_reports` and `scope`.
[^preparation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json`, frozen runtime, policy and first-attempt preparation.
[^controller]: `tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json`, `queue.observation` and `scope`.
[^allocation]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/audit-evidence.json`, `allocation`, exact `sacct_row`, partition limit and bounded interpretation.

[^v2]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json`, `scope`, `fixture`, `queue`, `native_accounting`, `native_input_verification`, `result`, `pre_entry_case`, `service_reports` and `remaining`.

[^v3-preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `rehearsal_audit`, `revision`, `staging`, `qualification`, `freeze`, `queue_submission` and `limits`; linked program reports retain exact inputs and observed boundaries.
