---
title: WRDS and HPC3 plaintext cleanup evidence
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/cleanup.json"
  - "tools/hpc3/artifacts/wrds-forced-kill-20261031/provenance.json"
  - "tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json": "7fc3888de5afbfcd89731034f8c401f855de4d28"
  "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/cleanup.json": "976cfc40fbaca4955084948061d6a2fcd90d54ab"
  "tools/hpc3/artifacts/wrds-forced-kill-20261031/provenance.json": "8325a595bcd3b5048f9050383347ff731d923a6f"
  "tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json": "f69fefdc21501823201906b30f64122f04c4a931"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json": "8e2fb19391d79f19118806c6420625050051c298"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# WRDS and HPC3 plaintext cleanup evidence

Cleanup claims belong to the allocation and observation that measured them.
Normal process cleanup, scheduler reclamation after forced termination and
physical secure erasure are different claims. The recorded probes establish
the first two in bounded cases; none establishes physical secure erasure or
cleanup after node failure.[^hpc3][^wrds]

## Recorded observations

The allocation-specific receipts support these observations.[^hpc3][^wrds][^collection][^diagnostics][^failed]

| Subject | Observer or receipt | What was measured |
|---|---|---|
| HPC3 synthetic forced-kill job `58629286` | Later job `58630827`, same node `hpc3-22-05` | Scratch absent several hours after termination. |
| WRDS synthetic forced-kill job `40551337` | Same-node job `40551338`, `wrds-sas30-w` | Scratch and private plaintext directory absent. |
| WRDS licensed extraction `40554174` | Job `40554207`, same node `wrds-sas31-w` | No matching scratch roots after extraction. |
| HPC3 diagnostic jobs `58737368`, `58737429`, `58737536` | Each diagnostic's own post-context assertion | Its private decrypted directory absent after normal exit. |
| Failed licensed HPC3 job `58733020` | Job `58740123`, same node `hpc3-14-24` | Both configured job directories absent 14,819 seconds after failure; original expanded scratch path was not logged. |

The first two observations are synthetic forced-kill probes; the extraction
and diagnostics have different subjects and termination modes. They cannot
substitute for one another.[^hpc3][^wrds][^collection][^diagnostics]

## Delayed accounting

The first WRDS reporting attempt encountered unavailable `qacct` data.
The corrected controller records submission before accounting and reconciles
that same observer without resubmitting it. The evidence retains the initial
error and the first observation; eventual scheduler accounting is not grounds
to create a duplicate observer.[^wrds]

## Failed licensed allocation

Observer `58740123` completed in five seconds on the original node. At
01:21:44 UTC on November 2 it measured absence of `/tmp/wagnera3/58733020`
and `/dev/shm/wagnera3/58733020`. Explicit host mounts, parent ownership and
the observer's own directories checked that it inspected host scratch.
The existing inspection service verified native accounting, six pinned inputs,
the result and its shared run record. No licensed files or keys were opened,
and nothing was deleted.[^failed]

This closes the missing **current configured-path observation**. The original
expanded `TMPDIR` was not logged; current prolog bytes cannot establish its
historical configuration. Absence four hours later establishes neither an
immediate cleanup deadline nor physical secure erasure.[^failed]

An evaluator-account administrator remains privileged. Process/key isolation
and final-grant ownership are described in [[wrds-proposer-isolation]] and
[[wrds-final-scoring]]; these cleanup observations do not establish those
controls by themselves.[^hpc3][^wrds]

[^hpc3]: `tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/cleanup.json`, `kill_job`, `observer_job` and `limits`.
[^wrds]: `tools/hpc3/artifacts/wrds-forced-kill-20261031/provenance.json`, `first_observation`, `measurement`, `preserved_failure` and `limits`.
[^collection]: `tools/hpc3/artifacts/wrds-licensed-collection-20261031/provenance.json`, `cleanup_observation`.
[^diagnostics]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, native reports, `cleanup_limit` and `study_action`.
[^failed]: `tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json`, `subject`, `observer`, `path_derivation`, `results`, `verified_measurement`, `record`, `service_reports`, `scope` and `limits`; retained inputs and reports are pinned by `files`.
