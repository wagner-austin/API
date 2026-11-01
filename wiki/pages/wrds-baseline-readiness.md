---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "libs/covenant_ml/docs/crsp-monthly.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json"
source_git_blobs:
  "libs/covenant_ml/docs/crsp-monthly.json": "5f1ba7b27191aff1ace131976c28a8c3fd596e14"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json": "8e2fb19391d79f19118806c6420625050051c298"
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
| Runtime | [[wrds-installed-runtime]] — verified controller/image identities; repository changes require a new installation. |
| Data | [[wrds-licensed-data-evidence]] — four CRSP relations extracted and encrypted; train/validation staged to HPC3. |
| Operations | [[wrds-operational-rehearsals]] — synthetic queue, seed summary, selection and controlled final execution. |
| Isolation | [[wrds-proposer-isolation]] and [[wrds-final-scoring]] — proposal boundary and evaluator-owned final release. |
| Cleanup | [[wrds-cleanup-evidence]] — bounded WRDS/HPC3 probes; failed licensed allocation still needs its own observation. |
| Scientific rules | [[crsp-monthly-study-design]] and [[crsp-monthly-holdings-validation]] — frozen target, chronology, accounting and uncertainty. |

These links separate code contracts from installed-runtime measurements and
study outcomes. Synthetic rehearsals do not establish real-data completeness
or investment performance.[^stop][^diagnostic]

## Work required before a new baseline

The current protocol declares rejected-order cash and actual traded-notional
costs. These version-2 rules differ from the stopped study's installed inputs.[^current]

1. Prepare a separately verified version-2 bundle: retain the original CSV/SQL
   identities and extraction vintage, and record changed protocol/manifest
   identities. The current staged bundle embeds version 1 and remains historical.
2. Freeze and install the version-2 evaluator and amended study rules,
   then rehearse proposal → admission → queue → native job → verified result.
3. Close the failed allocation's cleanup-evidence gap.
4. Run the new fixed candidate/seed study. Select only after all required
   validation attempts complete; final access stays a separate controlled step.

The original failed journal remains immutable. Current diagnosis and its
limits are recorded in [[wrds-baseline-20261101]]; machine registration is not
evidence that an idle AI session will be resumed by a queue event.[^stop][^diagnostic]

[^current]: `libs/covenant_ml/docs/crsp-monthly.json`, `version`, `entry_rule` and `cost_rule`; this is the current rule document, not an installed runtime or a completed study.
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action`, failed queue/native records and `wake`.
[^diagnostic]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, `interpretation`, `cleanup_limit` and `study_action`; subsequent code correction is described in the linked incident page.
