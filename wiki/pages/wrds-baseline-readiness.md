---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json"
  - "libs/covenant_ml/docs/crsp-monthly.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json"
  - "tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json"
source_git_blobs:
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

**The substantive study is stopped, not complete.** Its first licensed attempt
failed before fitting; the remaining fourteen attempts were not submitted,
no candidate was selected and the final holdout remains unreleased.
See [[wrds-baseline-20261101]] for the cause and correction boundary.[^stop]

## Readiness by responsibility

| Area | Evidence and current boundary |
|---|---|
| Runtime | [[wrds-installed-runtime]] — version-2 controller/image installed; current-runtime queue rehearsal passed.[^runtime][^v2] |
| Data | [[wrds-licensed-data-evidence]] — version-2 validation staged and independently reread on HPC3; [[wrds-protocol-revision]] records lineage. |
| Operations | [[wrds-operational-rehearsals]] — current version-2 queue rehearsal plus historical seed summary, selection and controlled final execution.[^v2] |
| Isolation | [[wrds-proposer-isolation]] and [[wrds-final-scoring]] — proposal boundary and evaluator-owned final release. |
| Cleanup | [[wrds-cleanup-evidence]] — bounded WRDS/HPC3 probes; failed licensed allocation still needs its own observation. |
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
- [ ] Freeze amended licensed-study resource and candidate rules.
- [ ] Close the failed allocation's cleanup-evidence gap.
- [ ] Run the new fixed candidate/seed study. Select only after all required
   validation attempts complete; final access stays a separate controlled step.

The original failed journal remains immutable. Current diagnosis and its
limits are recorded in [[wrds-baseline-20261101]]; machine registration is not
evidence that an idle AI session will be resumed by a queue event.[^stop][^diagnostic]

[^current]: `libs/covenant_ml/docs/crsp-monthly.json`, `version`, `entry_rule` and `cost_rule`; this is the current rule document, not an installed runtime or a completed study.
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action`, failed queue/native records and `wake`.
[^diagnostic]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, `interpretation`, `cleanup_limit` and `study_action`; subsequent code correction is described in the linked incident page.
[^stage]: `tools/hpc3/artifacts/wrds-v2-staging-20261102/evidence.json`, `staging`, `readback` and `scope`; encrypted validation staging only, with no evaluator installation or final release.

[^runtime]: `tools/hpc3/artifacts/wrds-protocol-v2-runtime-20261102/evidence.json`, `controller`, `image`, `remaining` and `scope`; installation observation only; subsequent rehearsal is recorded separately and the substantive study remains outstanding.

[^v2]: `tools/hpc3/artifacts/wrds-protocol-v2-rehearsal-20261102/evidence.json`, `scope`, `fixture`, `queue`, `native_accounting`, `native_input_verification`, `result`, `pre_entry_case`, `service_reports` and `remaining`.
