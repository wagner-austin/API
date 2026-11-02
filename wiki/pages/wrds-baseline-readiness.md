---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
source_git_blobs:
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

**The version-3 runtime and synthetic rehearsal passed independent review;
the exact previously failed licensed training span now passes. The new
substantive study has been frozen and its first shared-queue request reached
native HPC3 job `58749373`. No licensed evaluation result or selected candidate exists
yet; the final holdout remains unreleased.**[^preparation]

## Current evidence

| Area | Evidence and current boundary |
|---|---|
| Runtime | [[wrds-installed-runtime]] — source `087f67b4c`, separately inspected controller and image. |
| Data | [[wrds-licensed-data-evidence]] — eleven encrypted version-3 validation files staged and independently reread on HPC3. |
| Operations | [[wrds-operational-rehearsals]] — version-3 queue rehearsal independently verified; earlier selection/final rehearsals remain historical. |
| Qualification | Native job `58748788` passed the exact two-session training interval that stopped version 2, with sixteen verified inputs. |
| Study | [[wrds-baseline-v3-freeze]] — three candidates, five seeds, fifteen admissions; first native job submitted, fourteen withheld until its verified result. |
| Isolation | [[wrds-proposer-isolation]] and [[wrds-final-scoring]] — proposer boundary and evaluator-owned final release. |
| Scientific rules | [[crsp-monthly-study-design]] and [[crsp-monthly-holdings-validation]] — target, chronology, accounting and uncertainty. |

The current milestone is qualification and submission, not investment
performance or whole-dataset completeness. Queue `766bc4c8` recorded the native submission. The `/pub` BeeGFS
metadata stall recurred during submission, then cleared without changing the
request. Its unnecessary mount dependency remains tracked as task `0a0e5913`;
a passed preflight does not establish permanent recovery.[^preparation][^rehearsal]

## Remaining work

1. Complete and verify the first licensed run through the existing queue.
2. Complete the other fourteen fixed requests, or preserve a protocol stop
   if the evaluator finds unsupported data; never silently skip securities.
3. Aggregate the required seeds, select under the frozen tie rules, then
   perform the single controlled final evaluation.
4. Record validation and final results separately before allowing broader
   agent proposals through the same admission controls.[^preparation]

Stock-selection study task `f61db3a7` owns this sequence. Bermudan Asian
options task `5bec93df` is a separate study; its experiment and execution
contract must be frozen separately.

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
[^stop]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `study_action` and native failure; the original failed study remains historical.
[^diagnosis]: `tools/hpc3/artifacts/wrds-baseline-v2-validation-20261102/diagnosis.json`, `observed_boundary`, `interpretation`, `service_posts` and `limits`; no raw return observations or security identifiers are published.
