---
title: WRDS operational rehearsal evidence
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-seed-summary-rehearsal-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-seed-summary-rehearsal-20261101/evidence.json": "fc1b6b3b20e99d4ea3d594c7cc11b222f9dfcf77"
  "tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json": "0fe368911bbfdd7de3d68047e27f13463e86593e"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json": "2ef67b9d8ea85e09cf92c99d7ec56b96b07cb609"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json": "5d9afac1cd73df5108a17c49127f811dd8ae8a05"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# WRDS operational rehearsal evidence

These are synthetic workflow measurements. They show which execution and
control paths ran; they do not estimate licensed CRSP performance. Current
deployment and study status are separate in [[wrds-installed-runtime]] and
[[wrds-baseline-readiness]].[^summary][^selection][^final][^wrds]

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
