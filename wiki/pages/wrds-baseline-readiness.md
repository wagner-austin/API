---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/preserved-study-report.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json"
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
  "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json": "fe2f9f8774c053a9e07826ace70b9187ae9b0ef0"
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/preserved-study-report.json": "a3cb9cfa855ff7e11dd6f66884957fa3630af1a2"
  "tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json": "4dc9f9b11c5077f59a7122497a019faa1be8cc0b"
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json": "71ec4235748ce5b67ee40c7bec09b27926ff9d97"
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/stopped.json": "83fe5d2f5fa2468079f00334c0c2ddd2548ffddf"
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/launch.json": "ada422b51719ecf7356cfc273f49015fe77aa783"
  "tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json": "7fc3888de5afbfcd89731034f8c401f855de4d28"
  "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json": "a80e656edb6325abd242bf2f77ddbfca17938434"
  "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json": "3ad95ee9f98aa38ccd89f44a24a1e0c8a893a9d9"
  "libs/covenant_ml/docs/crsp-monthly.json": "a095650070b230379e759563df3c20b3c2476ded"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json": "8e2fb19391d79f19118806c6420625050051c298"
  "tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json": "1100c7de989b5f9fb98273fd26e16477db17637a"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-02
confidence: medium
---

# WRDS substantive baseline readiness

**The corrected study stopped before fitting; no licensed result is complete.**
Version-2 job `58740674` failed after 2,245 seconds while constructing training
labels (`CRSP-009`). The shared queue automatically recorded the failure.
Fourteen admitted requests remain unsubmitted, no candidate is selected, and
the final holdout remains unreleased. The maintained preparation reporter
reread the stopped study at 05:58 UTC on November 2: its journal matched the
independently observed post-failure digest, with one failed and fourteen
admitted attempts, no selection and no final manifest release.[^v2stop][^preserved]

Verified diagnostics `58741562` and `58741598` found an existing daily row with
missing price/return (`MP`/`MR`), followed by an available two-session return
(`P1`) linked to the earlier price date. The frozen one-session rule correctly
refuses this span. This differs from the version-1 pre-entry delisting. A
version-3 accounting amendment and its separately frozen runtime have passed
independent review. Its encrypted synthetic queue rehearsal completed as job
`58747974`, automatically reconciled and verified; the rehearsal's independent
review and licensed qualification remain before another frozen study.
The missing close has no observed value
and must not be silently filled or used to remove the security. See
[[crsp-monthly-holdings-validation]] for the new source contract.[^v2diagnosis][^current][^v3]

Preparation audit and launch checks passed: the job had 8 CPUs, 128 GiB, no
GPU and sixteen successful input checksums. Those establish execution
identity, not complete real-data coverage.[^launch]

The earlier version-1 attempt failed before fitting. Its journal remains
historical; see [[wrds-baseline-20261101]] for that failure and correction
boundary, and [[wrds-baseline-v2-freeze]] for the new fixed study.[^stop]

## Readiness by responsibility

| Area | Evidence and current boundary |
|---|---|
| Runtime | [[wrds-installed-runtime]] — version-3 installation independently verified and exercised by its synthetic queue rehearsal.[^v3] |
| Data | [[wrds-licensed-data-evidence]] — version-2 validation staged and independently reread on HPC3; [[wrds-protocol-revision]] records lineage. |
| Operations | [[wrds-operational-rehearsals]] — version-3 queue rehearsal completed; its independent review remains, and earlier selection/final rehearsals remain historical.[^v3] |
| Isolation | [[wrds-proposer-isolation]] and [[wrds-final-scoring]] — proposal boundary and evaluator-owned final release. |
| Cleanup | [[wrds-cleanup-evidence]] — original failed job's configured paths observed absent on its node; historical-path and timing limits remain.[^cleanup] |
| Scientific rules | [[crsp-monthly-study-design]] and [[crsp-monthly-holdings-validation]] — frozen target, chronology, accounting and uncertainty. |

These links separate code contracts from installed-runtime measurements and
study outcomes. Synthetic rehearsals do not establish real-data completeness
or investment performance.[^stop][^diagnostic]

## Baseline preparation sequence

The source protocol retains rejected-order cash and actual traded-notional
costs and adds the version-3 held-return rule. The completed preparation below
concerns the stopped version-2 study. The separate version-3 installation is
recorded above; its synthetic rehearsal completed without a licensed study or
final release. Substantive work is tracked by board task `f61db3a7`; the separate
Bermudan Asian options study is `5bec93df` and needs its own extraction.[^current][^v3]

- [x] Stage the separately verified version-2 validation bundle described in
   [[wrds-protocol-revision]]. Transfer and complete remote readback passed;
   the staged version-1 bundle remains historical.[^stage]
- [x] Install the separately frozen version-2 controller and HPC3 image.[^runtime]
- [x] Rehearse proposal → admission → queue → native job → verified result
   with the installed version-2 runtime and pre-entry delisting fixture.[^v2]
- [x] Freeze amended resource and candidate rules in [[wrds-baseline-v2-freeze]];
   fifteen requests admitted, native preflight and independent task audit passed.[^launch]
- [x] Observe the original version-1 allocation's configured paths on its node;
   service verification and independent task audit passed.[^cleanup]
- [ ] Run the new fixed candidate/seed study. Select only after all required
   validation attempts complete; queue `3d3aad6e` recorded the first attempt's
   failure. Version 3 requires licensed qualification and a new freeze before
   submission; neither failed study is silently retried.[^v2stop][^v3]

The original failed journal remains immutable. Its diagnosis and its
limits are recorded in [[wrds-baseline-20261101]]; machine registration is not
evidence that an idle AI session will be resumed by a queue event.[^stop][^diagnostic]

[^current]: `libs/covenant_ml/docs/crsp-monthly.json`, `version`, `entry_rule`, `cost_rule` and `held_return_rule`; this is the current rule document, not an installed runtime or a completed study.
[^v3]: `tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json`, `prerequisite_audits`, `queue`, `native_accounting`, `result`, `preflight_incident` and `remaining`; source/runtime reviews passed, while synthetic rehearsal review and licensed execution are separate milestones.
[^preserved]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/preserved-study-report.json`, `observed_at`, `policy_digest`, `journal_sha256` and `attempts`; service post `6029b64c-80a4-444a-9f73-96a604aad1b5` from the existing preparation reporter. Digest `82450a7994d16019b9f699104b5e8891d943ad426becd93c5fb56402b0854a40` matches the independent audit reading, and was unchanged before and after this report.
[^candidate]: `tools/hpc3/artifacts/wrds-protocol-v3-runtime-20261102/evidence.json`, `source_checks`, `controller`, `image`, `remaining` and `scope`; passed source gate and separately inspected candidate runtime, without a new study launch.
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action`, failed queue/native records and `wake`.
[^diagnostic]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, `interpretation`, `cleanup_limit` and `study_action`; subsequent code correction is described in the linked incident page.
[^stage]: `tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json`, `staging`, `readback` and `scope`; encrypted validation staging only, with no evaluator installation or final release.

[^runtime]: `tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json`, `controller`, `image`, `remaining` and `scope`; installation observation only; subsequent rehearsal is recorded separately and the substantive study remains outstanding.

[^v2]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json`, `scope`, `fixture`, `queue`, `native_accounting`, `native_input_verification`, `result`, `pre_entry_case`, `service_reports` and `remaining`.

[^cleanup]: `tools/hpc3/artifacts/wrds-failed-scratch-20261102/evidence.json`, `subject`, `observer`, `results`, `verified_measurement`, `service_reports` and `limits`; original expanded scratch path was not logged. Independent board audit `b239dc19-84e4-49c1-b429-7002656cf0bd` returned VERIFIED at 2026-11-02T01:31:49Z.

[^launch]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/launch.json`, `freeze_audit`, `queue`, `native`, `native_service_post`, `attempts`, `execution_cluster_change` and `submission_rule`; pinned sibling documents retain queue transitions, native service inspection, allocation readout and checksum receipts. This is launch evidence, not a completed evaluation.

[^v2stop]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/stopped.json`, `native`, `queue`, `failure`, `attempts`, `actions` and `cleanup_limit`; raw identifiers and return observations remain private. Prior same-node cleanup evidence concerns the earlier allocation, not this new failure.

[^v2diagnosis]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json`, `observed_boundary`, `interpretation`, `vendor_source`, `service_posts`, `study_action` and `limits`; sibling diagnostic evidence retains native accounting, all sixteen input identities, shared records and artifact digests. Diagnostic plaintext removal does not establish the failed study allocation's post-exit cleanup.
