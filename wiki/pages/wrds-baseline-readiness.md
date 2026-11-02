---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/stopped.json"
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/launch.json"
  - "tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json"
  - "libs/covenant_ml/docs/crsp-monthly.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json"
  - "tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/stopped.json": "83fe5d2f5fa2468079f00334c0c2ddd2548ffddf"
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/launch.json": "ada422b51719ecf7356cfc273f49015fe77aa783"
  "tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json": "7fc3888de5afbfcd89731034f8c401f855de4d28"
  "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json": "a80e656edb6325abd242bf2f77ddbfca17938434"
  "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json": "3ad95ee9f98aa38ccd89f44a24a1e0c8a893a9d9"
  "libs/covenant_ml/docs/crsp-monthly.json": "5f1ba7b27191aff1ace131976c28a8c3fd596e14"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json": "8e2fb19391d79f19118806c6420625050051c298"
  "tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json": "1100c7de989b5f9fb98273fd26e16477db17637a"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# WRDS substantive baseline readiness

**The corrected study stopped before fitting; no licensed result is complete.**
Version-2 job `58740674` failed after 2,245 seconds while constructing training
labels: a held daily observation was absent or failed the ordinary-return
predicate (`CRSP-009`). The shared queue automatically recorded the failure.
A targeted diagnostic is pending; no rule change is justified yet. Fourteen
admitted requests remain unsubmitted, no candidate is selected, and the final
holdout remains unreleased.[^v2stop]

Preparation audit and launch checks passed: the job had 8 CPUs, 128 GiB, no
GPU and sixteen successful input checksums. Those establish execution
identity, not complete real-data coverage.[^launch]

The earlier version-1 attempt failed before fitting. Its journal remains
historical; see [[wrds-baseline-20261101]] for that failure and correction
boundary, and [[wrds-baseline-v2-freeze]] for the new fixed study.[^stop]

## Readiness by responsibility

| Area | Evidence and current boundary |
|---|---|
| Runtime | [[wrds-installed-runtime]] — version-2 controller/image installed; current-runtime queue rehearsal passed.[^runtime][^v2] |
| Data | [[wrds-licensed-data-evidence]] — version-2 validation staged and independently reread on HPC3; [[wrds-protocol-revision]] records lineage. |
| Operations | [[wrds-operational-rehearsals]] — current version-2 queue rehearsal plus historical seed summary, selection and controlled final execution.[^v2] |
| Isolation | [[wrds-proposer-isolation]] and [[wrds-final-scoring]] — proposal boundary and evaluator-owned final release. |
| Cleanup | [[wrds-cleanup-evidence]] — original failed job's configured paths observed absent on its node; historical-path and timing limits remain.[^cleanup] |
| Scientific rules | [[crsp-monthly-study-design]] and [[crsp-monthly-holdings-validation]] — frozen target, chronology, accounting and uncertainty. |

These links separate code contracts from installed-runtime measurements and
study outcomes. Synthetic rehearsals do not establish real-data completeness
or investment performance.[^stop][^diagnostic]

## Baseline preparation sequence

The current protocol declares rejected-order cash and actual traded-notional
costs. These version-2 rules differ from the stopped study's installed inputs.[^current]

- [x] Stage the separately verified version-2 validation bundle described in
   [[wrds-protocol-revision]]. Transfer and complete remote readback passed;
   the staged version-1 bundle remains historical.[^stage]
- [x] Install the separately frozen version-2 controller and HPC3 image.[^runtime]
- [x] Rehearse proposal → admission → queue → native job → verified result
   with the installed version-2 runtime and pre-entry delisting fixture.[^v2]
- [x] Freeze amended resource and candidate rules in [[wrds-baseline-v2-freeze]];
   fifteen requests admitted, native preflight and independent task audit passed.[^launch]
- [x] Observe the failed allocation's configured paths on its original node;
   service verification and independent task audit passed.[^cleanup]
- [ ] Run the new fixed candidate/seed study. Select only after all required
   validation attempts complete; queue `3d3aad6e` recorded the first attempt's
   failure. Investigate it before further submission.[^v2stop]

The original failed journal remains immutable. Current diagnosis and its
limits are recorded in [[wrds-baseline-20261101]]; machine registration is not
evidence that an idle AI session will be resumed by a queue event.[^stop][^diagnostic]

[^current]: `libs/covenant_ml/docs/crsp-monthly.json`, `version`, `entry_rule` and `cost_rule`; this is the current rule document, not an installed runtime or a completed study.
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action`, failed queue/native records and `wake`.
[^diagnostic]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, `interpretation`, `cleanup_limit` and `study_action`; subsequent code correction is described in the linked incident page.
[^stage]: `tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json`, `staging`, `readback` and `scope`; encrypted validation staging only, with no evaluator installation or final release.

[^runtime]: `tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json`, `controller`, `image`, `remaining` and `scope`; installation observation only; subsequent rehearsal is recorded separately and the substantive study remains outstanding.

[^v2]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json`, `scope`, `fixture`, `queue`, `native_accounting`, `native_input_verification`, `result`, `pre_entry_case`, `service_reports` and `remaining`.

[^cleanup]: `tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json`, `subject`, `observer`, `results`, `verified_measurement`, `service_reports` and `limits`; original expanded scratch path was not logged. Independent board audit `b239dc19-84e4-49c1-b429-7002656cf0bd` returned VERIFIED at 2026-11-02T01:31:49Z.

[^launch]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/launch.json`, `freeze_audit`, `queue`, `native`, `native_service_post`, `attempts`, `execution_cluster_change` and `submission_rule`; pinned sibling documents retain queue transitions, native service inspection, allocation readout and checksum receipts. This is launch evidence, not a completed evaluation.

[^v2stop]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/stopped.json`, `native`, `queue`, `failure`, `attempts`, `actions` and `cleanup_limit`; raw identifiers and return observations remain private. Prior same-node cleanup evidence concerns the earlier allocation, not this new failure.
