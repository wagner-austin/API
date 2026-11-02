---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-execution-audit-20261102/incomplete.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/submit.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-execution-audit-20261102/incomplete.json": "8ee14efd7aefaa88ffdaa7ed66c6ca0effe2fd51"
  "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/submit.json": "8977f84838fbcd6346de5cffdcf86921e72883b0"
  "tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json": "46ea508c6280a6626b7803d9b8cd87f25973d8eb"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json": "fe2f9f8774c053a9e07826ace70b9187ae9b0ef0"
  "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json": "71ec4235748ce5b67ee40c7bec09b27926ff9d97"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-02
confidence: medium
---

# WRDS substantive baseline readiness

**The first licensed version-3 run stopped before fitting. Native job
`58749373` failed after 1,099 seconds with `CRSP-004` at an ordinary exit
requiring a finite positive closing trade. The queue recorded the failure
automatically. One attempt failed; fourteen remain admitted and unsubmitted.
No candidate is selected and the final holdout remains unreleased.**[^v3stop]

## Current evidence

| Area | Evidence and current boundary |
|---|---|
| Runtime | [[wrds-installed-runtime]] — source `087f67b4c`, separately inspected controller and image. |
| Data | [[wrds-licensed-data-evidence]] — eleven encrypted version-3 validation files staged and independently reread on HPC3. |
| Operations | [[wrds-operational-rehearsals]] — version-3 queue rehearsal independently verified; earlier selection/final rehearsals remain historical. |
| Qualification | Native job `58748788` passed the exact two-session training interval that stopped version 2, with sixteen verified inputs. |
| Study | [[wrds-baseline-v3-freeze]] — three candidates, five seeds, fifteen admissions; one failed attempt, fourteen unsubmitted; study stopped.[^v3stop] |
| Isolation | [[wrds-proposer-isolation]] and [[wrds-final-scoring]] — proposer boundary and evaluator-owned final release. |
| Scientific rules | [[crsp-monthly-study-design]] and [[crsp-monthly-holdings-validation]] — target, chronology, accounting and uncertainty. |

The earlier exact missing-return-span qualification passed, but it did not
establish whole-dataset completeness. All sixteen frozen inputs of the failed
allocation passed native inspection; the error arose in label construction,
before model fitting. The exact exit-date source flag and price remain to be
diagnosed. This is distinct from the earlier held-return-span failure.[^v3stop]

The first full-panel diagnostic (`58759614`) constructed 262 decisions but
stopped on an entry's missing price: its collector caught `ValueError`, while
the validator raised `JSONTypeError`. Its last progress message covered 84
decisions and 84,000 paths; no complete counts were produced. Replacement
diagnostic `58760716` is submitted; its completion is not yet verified.[^pathaudit]

The `/pub` BeeGFS stall also recurred during submission and cleared without
changing the request. Its unnecessary mount dependency remains tracked as
`0a0e5913`; it was not the recorded cause of this evaluator failure.[^preparation][^v3stop]

## Remaining work

1. Complete and verify the full-panel diagnostic, including typed missing-value
   refusals; the traceback alone does not justify a new fill rule.
2. Determine whether data correction or an explicit reviewed protocol amendment
   is warranted. Preserve failed requests; do not silently retry or exclude
   securities on future outcome availability.
3. Only after a qualified new freeze, execute validation, select under fixed
   rules, and perform the controlled final evaluation.[^v3stop]

Stock-selection study task `f61db3a7` owns this sequence. Bermudan Asian
options task `5bec93df` is a separate study; its experiment and execution
contract must be frozen separately.[^studies]

## Preserved failures

Version-1 job `58733020` stopped before fitting on a pre-entry delisting;
[[wrds-baseline-20261101]] records its correction. Version-2 job `58740674`
failed after 2,245 seconds constructing training labels (`CRSP-009`). Verified
diagnostics found an `MP`/`MR` row followed by a linked `P1` return. Version 3
accounts for that exact missing-price span once, without fabricating a close.
The failed journals remain historical; [[wrds-baseline-v2-freeze]] retains
the earlier freeze.[^stop][^diagnosis][^preparation]

The normal qualification's plaintext cleanup does not prove forced-kill
cleanup for another allocation. [[wrds-cleanup-evidence]] retains the
allocation-specific observations and their limits.[^preparation]

[^preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `rehearsal_audit`, `staging`, `qualification`, `freeze`, `queue_submission` and `limits`; sibling program reports retain exact identities and recorded observations.
[^rehearsal]: `tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json`, `preflight_incident`; recovery of the unchanged preflight did not establish a permanent filesystem fix.
[^studies]: `tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json`, `remaining`; separate stock-selection and options task identities, without making its historical extraction plan a current requirement.
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action` and native failure; the original failed study remains historical.
[^diagnosis]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json`, `observed_boundary`, `interpretation`, `service_posts` and `limits`; no raw return observations or security identifiers are published.

[^v3stop]: `tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json`, `native`, `queue`, `failure`, `attempts`, `study_action` and `limits`; sibling native-input report, stopped preparation report and retained traceback distinguish observed failure from unresolved data diagnosis.

[^pathaudit]: `tools/hpc3/artifacts/wrds-baseline-v3-execution-audit-20261102/incomplete.json`, `native`, `panel_decisions`, `last_progress_paths`, `failure` and `limits`; `tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/submit.json`, `job_id`, `experiment` and `image_digest`. Submission is not completion or a study result.
