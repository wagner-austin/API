---
title: CRSP version-3 licensed study freeze
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]", "[[wrds-baseline-v2-freeze]]", "[[wrds-protocol-revision]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/result.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/native-evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-execution-audit-20261102/incomplete.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/freeze-report.json"
  - "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evaluation.json"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/result.json": "6a91f67fc36dc4d3aa6b9bbcc7279ca019a78d62"
  "tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/native-evidence.json": "98ee300170fb1608bb4cf3f4b7298c0323d762d1"
  "tools/hpc3/artifacts/wrds-baseline-v3-execution-audit-20261102/incomplete.json": "8ee14efd7aefaa88ffdaa7ed66c6ca0effe2fd51"
  "tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json": "46ea508c6280a6626b7803d9b8cd87f25973d8eb"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json": "6e90bce84aae5ac72a0bca3a10d84da0e7c05c33"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/freeze-report.json": "66ac30dcc341698ef31b3c144a633303fe10025d"
  "tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evaluation.json": "cc3375f54d1b5ced0efc206cc9e42f13d019fd49"
fact_checked: 2026-11-02
confidence: medium
---

# CRSP version-3 licensed study freeze

This is an explicit accounting/runtime amendment of the stopped version-2
study. The candidate matrix, estimator settings, seed aggregation, tie rules
and resource ceilings are unchanged. Fifteen requests were admitted before
the first queue submission; this freeze is not an evaluation result.[^freeze]

## Frozen design

Training covers 2000–2014, validation 2015–2021 and the sealed final period
2022–2024. The current amendment retains rejected-entry cash and actual
traded-notional costs, and adds `exact-missing-price-spans`. Universe filters,
calendar and terminal rules remain in [[crsp-monthly-study-design]].[^design]

| Control | Frozen value |
|---|---|
| Candidate features | momentum + liquidity; momentum only; liquidity only |
| Seeds per candidate | 11, 23, 37, 53, 71 |
| Estimator | 200 trees, depth 6, 31 leaves |
| Selection | LightGBM minus momentum at 10 bps; average seeds within month, then 82 required months |
| Tie | Fewer leaves, then lexicographic candidate digest |
| Final seed | 11 |
| Resources per request | 8 CPUs, 128 GiB, 7,200-second payload timeout; 121-minute allocation, no GPU |
| Uncertainty | 95% confidence; 12-month moving blocks, 10,000 draws, bootstrap seed 7 |

Selection and estimator settings are inherited unchanged from the prior
freeze. The exact request digests and candidate identities are in the
program-authored matrix. Policy digest is
`2a3ae28753525fc7bd8818add1d369d2ff224f71557cb30cc49dd26b91025c06`.[^freeze][^design]

## Qualification and execution boundary

Native job `58748788` used source `087f67b4c` and the frozen image to evaluate
the exact prior failed training span. It completed in 76 seconds; native
inspection bound all sixteen input digests, its result and shared RunRecord.
The check establishes the linked two-session return is applied once, within
`1e-12`, and normal-exit plaintext removal. It does not qualify every interval
or prove cleanup after a different allocation is killed.[^preparation]

The first request reached native job `58749373` through shared queue
`766bc4c8`, then failed after 1,099 seconds while constructing labels at an
ordinary exit (`CRSP-004`). No model was fitted. The fourteen other requests
remain unsubmitted; no candidate is selected and no final data is released.
[[wrds-baseline-readiness]] records the diagnosis boundary. This freeze remains
historical and must not be silently changed for a retry.[^v3stop]

Diagnostic job `58759614` built the 262-decision panel but failed on a typed
missing entry price after its last progress report at 84,000 paths. All sixteen
inputs were verified; it produced no complete diagnostic result or study
score. The collector's correction and replacement submission do not amend
this freeze.[^pathaudit]

## Completed path diagnostic

Replacement `58760716` completed with exit `0:0` in 2,126 seconds. Native
inspection verified all sixteen inputs, the result and shared RunRecord.
Its unchanged frozen evaluator visited all 262,000 paths from 262 decisions:
261,973 accepted, 27 refused, and zero empty decisions. Accepted paths include
153 confirmed rejected-entry cash outcomes and 1,093 terminal paths.[^complete]

| First refusal boundary | Training paths | Validation paths | Total |
|---|---:|---:|---:|
| Ordinary exit | 3 | 0 | 3 |
| Entry | 4 | 0 | 4 |
| Held returns | 14 | 1 | 15 |
| Terminal valuation | 4 | 1 | 5 |
| All refusals | 25 | 2 | 27 |

The first refused path, ordinal 9,165, has a positive `BA` ordinary exit and
a delisting on that exit date. The frozen interval excludes that event from
terminal valuation and requires a closing trade for sale. The scan records
first refusals, not every fault within a path; its boundary examples do not
identify every problematic held-return row or terminal field. It assigns no
replacement prices, returns or labels, fits no model and accesses no final
data. Normal-exit plaintext removal is asserted by this program, not a separate
same-node forced-kill probe. Follow-up [[wrds-v3-refused-path-diagnosis]] adds
source fields for every refusal and distinguishes actual held-return failures
from late terminal valuations. The version-3 study remains a protocol stop;
[[wrds-baseline-readiness]] records the remaining work.[^complete][^sources]

[^freeze]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/freeze-report.json`, `old_policy`, `new_policy`, `policy_fields_changed`, `design_fields_changed`, `selection_unchanged`, `estimator_unchanged` and `matrix`; unchanged selection contract from [[wrds-baseline-v2-freeze]].
[^design]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evaluation.json`, `design`, `stopping_months`, `fold_months` and `confidence`.
[^preparation]: `tools/hpc3/artifacts/wrds-baseline-v3-preparation-20261102/evidence.json`, `qualification`, `freeze`, `queue_submission` and `limits`; qualification is separate from a complete licensed study.

[^v3stop]: `tools/hpc3/artifacts/wrds-baseline-v3-validation-20261102/stopped.json`, `native`, `queue`, `failure`, `attempts`, `study_action` and `limits`; sibling native-input report, stopped preparation report and retained traceback distinguish observed failure from unresolved data diagnosis.

[^pathaudit]: `tools/hpc3/artifacts/wrds-baseline-v3-execution-audit-20261102/incomplete.json`, `native`, `input_count`, `panel_decisions`, `last_progress_paths`, `failure` and `limits`; sibling native-input report and logs retain the program observations.

[^complete]: `tools/hpc3/artifacts/wrds-baseline-v3-typed-execution-audit-20261102/result.json`, `results[0]` counts, `refusal_counts`, `first_failure`, `plaintext_removed` and `scope`; sibling `native-evidence.json`, `accounting`, `measurement.verified_files`, result/record digests and `record`. Public observations are flags, counts and row digests, without security identifiers, dates, prices or returns.
[^sources]: [[wrds-v3-refused-path-diagnosis]], "Every original refusal", using the completed source-level native job `58766611`; its interpretation does not amend this frozen evaluator.
