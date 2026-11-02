---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/diagnosis-audit.json"
  - "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/parent-audit.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/result.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/native-evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/completion-preparation-report.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/diagnosis-audit.json": "97556c4ac055d150ce3de0964fc0206a3cd862bb"
  "tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/parent-audit.json": "78746db728bf7fd34597498d2802519915b81291"
  "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/result.json": "6a91f67fc36dc4d3aa6b9bbcc7279ca019a78d62"
  "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/native-evidence.json": "98ee300170fb1608bb4cf3f4b7298c0323d762d1"
  "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/completion-preparation-report.json": "97ff089365009ac9ba4d9284d4c420b1df60a391"
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
No candidate is selected and the final holdout remains unreleased. The completed
diagnostic found 27 refused paths out of 262,000; it is not a model result.**[^v3stop][^pathaudit]

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
before model fitting. This is distinct from the earlier held-return-span
failure.[^v3stop]

Full-panel diagnostic `58760716` completed in 2,126 seconds with all sixteen
inputs, its result and RunRecord independently verified. Across 262 decisions,
261,973 paths were accepted and 27 refused: 3 ordinary exits, 4 entries,
15 held-return paths and 5 terminal valuations; 25 were in training and 2 in
validation. The first refusal has a finite positive `BA` price at an ordinary
exit with a delisting on that exit date. A bid/ask average does not satisfy the
frozen closing-trade rule. Counts measure the first refusal per path; they do
not establish every underlying cause. [[wrds-baseline-v3-freeze]] retains the
failed first diagnostic and the completed scan's limits.[^pathaudit]

Follow-up native job `58766611` reproduced all counts and collected source
fields for every refusal, with seventeen inputs and its result/RunRecord
verified. [[wrds-v3-refused-path-diagnosis]] distinguishes vendor missingness,
quote-only execution and valuation timing. Five earlier `held_path` refusals
are actually terminal valuations dated after exit. No extraction repair or
complete replacement outcome policy has been established.[^sources]

The `/pub` BeeGFS stall also recurred during submission and cleared without
changing the request; it was not the recorded cause of this evaluator failure.
[HPC3 mount ownership](../../tools/hpc3/wiki/pages/declared-image-mounts.md)
records the published launcher correction and successful home/shared native
probes. Those probes used a pinned source overlay, not a replacement of the
frozen study controller or its requests.[^preparation][^v3stop][^mounts]

## Remaining work

1. Resolve the execution, late-valuation and incomplete-outcome rules through
   evidenced data corrections or an explicit reviewed protocol amendment.
   Preserve failed requests; do not silently retry, treat quotes as trades,
   accept flagged partial returns or exclude securities on future availability.
2. Only after a qualified new freeze, execute validation, select under fixed
   rules, and perform the controlled final evaluation.[^v3stop]

Follow-up task `f963ee0b` tracks that protocol/source qualification. The reviewed
source diagnosis remains the starting evidence; no new outcome policy is approved.[^sources]

Stock-selection task `f61db3a7` closed after independent audit `fe0cc4fc`
verified the explicit protocol stop. Source-diagnosis task `a9ad53ae` then passed
independent audit `e9d32bd4` on November 2 at 18:30 UTC. That review verifies the
diagnosis and its evidence; it does not qualify a new protocol or establish a
successful model result.[^diagnosis-review] Bermudan Asian options task `5bec93df` is a separate study
with a separate experiment and execution contract.[^parent][^studies]

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
[^parent]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/parent-audit.json`, captured live `task_get` response: parent status, audit-verifier transition at 2026-11-02T15:21:09.847Z, closure outcome and follow-up ownership.
[^diagnosis-review]: `tools/hpc3/artifacts/wrds-v3-source-path-audit-20261102/diagnosis-audit.json`, captured live `task_get` response, task status `done` and audit-verifier transition at 2026-11-02T18:30:14.804Z.
[^mounts]: [HPC3 mount ownership](../../tools/hpc3/wiki/pages/declared-image-mounts.md), § "Native verification and limits", jobs `58776025` and `58776026`, with original native reports and controller identity.
[^sources]: [[wrds-v3-refused-path-diagnosis]], § "Every original refusal", § "Data limitations versus implementation" and § "Next action", with native source-field evidence and the primary CIZ definitions.
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action` and native failure; the original failed study remains historical.
[^diagnosis]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json`, `observed_boundary`, `interpretation`, `service_posts` and `limits`; no raw return observations or security identifiers are published.

[^v3stop]: `tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json`, `native`, `queue`, `failure`, `attempts`, `study_action` and `limits`; sibling native-input report, stopped preparation report and retained traceback distinguish observed failure from unresolved data diagnosis.

[^pathaudit]: `tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/result.json`, `results[0]` counts, `refusal_counts`, `first_failure` and `scope`; sibling `native-evidence.json`, `accounting`, `measurement` and `record`; sibling `completion-preparation-report.json`, fifteen `attempts` with one failed, fourteen admitted, no selection or final-manifest identity and the unchanged journal digest. Diagnostic completion is not a model fit or a final evaluation.
