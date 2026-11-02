---
title: WRDS operational rehearsal evidence
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/audit-measurement.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-seed-summary-rehearsal-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json": "fe2f9f8774c053a9e07826ace70b9187ae9b0ef0"
  "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/audit-measurement.json": "c6de49289a168f4bd12cdcd78419acaa972f3107"
  "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json": "a80e656edb6325abd242bf2f77ddbfca17938434"
  "tools/hpc3/artifacts/wrds-seed-summary-rehearsal-20261101/evidence.json": "fc1b6b3b20e99d4ea3d594c7cc11b222f9dfcf77"
  "tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json": "0fe368911bbfdd7de3d68047e27f13463e86593e"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json": "2ef67b9d8ea85e09cf92c99d7ec56b96b07cb609"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json": "5d9afac1cd73df5108a17c49127f811dd8ae8a05"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-02
confidence: medium
---

# WRDS operational rehearsal evidence

These are synthetic workflow measurements. They show which execution and
control paths ran; they do not estimate licensed CRSP performance. Current
deployment and study status are separate in [[wrds-installed-runtime]] and
[[wrds-baseline-readiness]].[^summary][^selection][^final][^wrds]

## Current protocol-v3 queue rehearsal

The independently reviewed `087f67b4c` controller/image executed one encrypted
synthetic request containing both the entry-date delisting and an explicit
missing-price row closed by a linked two-session return. Native job `58747974`
completed in ten seconds with four CPUs and 16 GiB. The existing controller
automatically reconciled queue `b45ef01a` to passed; replay before and after
completion returned that same identity, with one completed journal attempt.[^v3]

Service inspection verified sixteen pinned inputs and sixteen compute checksum
receipts. Readback checked all 240 monthly accounting observations across
fifteen strategy/cost cases and sixteen months, plus four paired comparisons.
The stock-7 ending weight divided by the rejected-order cash slot recovered
the original daily-product growth across the missing-price span within
`1e-12`; entry rejection preserved the old terminal receivable until its
next rebalance. No candidate was selected and no final holdout was released.
Independent audit `915920df` returned VERIFIED at 06:54:36 UTC on
November 2. Its closure checks include actual replay calls, native runtime
binding and the wiki's limits statement.[^v3][^v3-preparation]

Initial preflight reached its 600-second timeout while Apptainer waited on
`/pub` BeeGFS metadata. The unchanged canonical preflight passed after a bounded
filesystem probe recovered; no frozen request was edited. The unnecessary
mount dependency is tracked separately as `0a0e5913`; recovery is not a
permanent storage fix. The licensed study follows under `f61db3a7`.[^v3]

## Historical protocol-v2 queue rehearsal

One fresh encrypted synthetic attempt used the installed `8ef7e0f17`
controller and image. The maintained pre-entry fixture delists security 8 on
2019-02-28 before entry on 2019-03-01. The existing shared queue submitted
native job `58740021`, which completed in seven seconds with four CPUs and
16 GiB; automatic reconciliation marked queue `f1961f01` passed. Replay
returned the same queue identity and the journal retained one complete job.[^v2]

A subsequent service measurement regenerated all ten plaintext files from
the frozen source archive and matched their bundle digests, including the
pre-entry delisting row. It also verified the immutable queue specification's
selection of the version-2 evaluator and its four pinned input documents.
The queue adapter and selected evaluator are separate installations.[^measurement]

Service inspection verified sixteen pinned inputs and sixteen compute checksum
receipts, then bound the result and shared RunRecord to that allocation.
The result has fifteen strategy/cost cases over sixteen validation months and
four paired comparisons. Selected new purchases of security 8 are unfilled;
the old holding's terminal receivable is separate. Readback checks confirm
ending weights sum to one and costs equal actual turnover times the cost rate.
This synthetic rehearsal does not establish licensed baseline performance or
cleanup of the earlier failed allocation. No final grant was released.[^v2]

## Queue through seed summary

Five fixed seeds of one candidate ran through the shared queue on encrypted
synthetic data using source `b3843a3d3`. Native jobs `58721392`,
`58721481`, `58721691`, `58721976` and `58722099` passed. The maintained
summary command exited zero and left the journal digest unchanged. The
service-authored summary reading recomputed the payload and shared RunRecord;
the evidence binds invocation, code hashes and result digests.[^summary]

## Selection controls

A separate rehearsal completed two candidates with five seeds each.
The maintained selector refused the wrong candidate, wrong seed and repeated
selection, and selected attempt 1 at seed 11. No licensed data or final grant
was used. The implementation contract belongs in [[wrds-final-scoring]].[^selection]

## Controlled final evaluation

An earlier synthetic rehearsal completed validation job `58635333` and final
job `58635492` through the shared queue. Both verified 27 inputs. The grant
was consumed locally and remotely; delivered keys were absent afterward, and
delivery/submission replays exited nonzero. This proves the recorded synthetic
one-shot path, not a licensed final result or a current-runtime replay.[^final]

## WRDS native storage

WRDS jobs `40550023` and `40550033` opened encrypted synthetic validation
bundles in private directories with mode 0700 and reported plaintext removal
on normal exit. Compared observation fields matched across both runs.
WRDS used a portable Python runtime; this was not an Apptainer execution or
a monthly scientific study. Forced-termination evidence is kept separately
in [[wrds-cleanup-evidence]].[^wrds]

The records retain earlier failures and their limits. A later successful
rehearsal does not erase a failed run or authorize a silent study retry.
Final-data permissions remain governed by [[wrds-proposer-isolation]] and
[[wrds-final-scoring]].[^final][^wrds]

[^summary]: `tools/hpc3/artifacts/wrds-seed-summary-rehearsal-20261101/evidence.json`, `trials`, `summarize`, `summary_report` and `measurement_boundary`.
[^selection]: `tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json`, `scope`, `required_seeds`, `queue` and `reports`.
[^final]: `tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json`, `validation`, `final`, `grant`, `recorded_failures` and `limitations`.
[^wrds]: `tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json`, `measurements`, `repeatability` and `limits`.

[^v2]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json`, `scope`, `fixture`, `queue`, `native_accounting`, `native_input_verification`, `result`, `pre_entry_case`, `service_reports` and `remaining`.

[^measurement]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/audit-measurement.json`, `measurement`, `program`, `execution` and `interpretation`; service post `93e273e2-6e02-46a1-973f-f169925a3be3` binds regenerated fixture identities and the immutable queue selection, not a new historical process attestation.

[^v3]: `tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json`, `prerequisite_audits`, `queue`, `queue_replay`, `native_accounting`, `native_input_verification`, `result`, `service_reports`, `preflight_incident` and `remaining`; sibling files retain program-authored fixture and result measurements, measurement programs and native evidence.

[^v3-preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `rehearsal_audit`, `revision`, `staging`, `qualification`, `freeze`, `queue_submission` and `limits`; linked program reports retain exact inputs and observed boundaries.
