---
title: November 1 licensed CRSP baseline stop and diagnosis
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-baseline-readiness]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json"
  - "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json": "3e3df28fd555e8d2b8165cc1fe8c56036d3cf95f"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json": "8e2fb19391d79f19118806c6420625050051c298"
  "tools/hpc3/artifacts/wrds-licensed-baseline-20261101/preparation.json": "b241815ebdf05dc49c92a4b6a786df80a230c96d"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "35c5af1aa9768da3ddd0d9c49818d699751cbe6c"
  "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py": "97ee0fc04a51497e2130087475c5780de6757d5d"
provenance:
  - "Reorganized from wrds-baseline-readiness at API fbff31147581c22e1a91a13108e89217c41a4380 on 2026-11-01; original run records remain in artifacts and git history. Board task c6bc5d05-22f5-4462-a1f7-eb40de1f28f1."
fact_checked: 2026-11-01
confidence: medium
---

# November 1 licensed CRSP baseline stop and diagnosis

The first required licensed attempt failed before model fitting.
The preregistration stopped without retry, replacement or candidate selection;
attempts 2–15 were not submitted and the final holdout remained unreleased.
This is a stopped study, not a completed validation result.[^stopped]

## Failure and cause

Shared queue `b799f91b-4c5b-4f96-9332-b781b9ae1330` reconciled native job
`58733020` as failed with exit `1:0` after 2,131 seconds. Peak memory was
about 36.2 GiB against the 128-GiB request. Label construction requested a
missing ordinary daily return; this was not an out-of-memory failure.[^stopped]

Three frozen-runtime diagnostic jobs completed with exit `0:0`:
`58737368`, `58737429` and `58737536`. They confirmed that the security
delisted on the decision date, before the planned entry. The entry-day row was
a terminal record without the accepted trade-price flag, and no later daily
rows existed. The decision panel retained the security; the outcome path
searched for ordinary returns because its terminal event preceded entry.
The probes emitted metadata, not licensed prices or returns.[^diagnostic]

## Boundary correction

The holding-window terminal interval is `[entry, exit)`, so an event
**on entry** is already included. That differs from the observed event
**before entry**. A real closing trade on entry can accrue a subsequent
terminal value; a later conventional terminal record is not a purchase.[^path]

The shared outcome path now calls the existing fill validator before
constructing labels and checks an ordinary exit's fill. The validator rejects
terminal records even if a contradictory trade-price flag is present.
Regression cases cover events before entry, exactly on entry and exactly on
exit, endpoint quote prices, missing entry rows, and a verified-file reproduction
that preserves decision-time selection.[^path][^tests]

This is a fail-closed validation correction, not permission to drop unfillable
securities or impute their returns. It does not turn the stopped study into a
successful one. Installed-runtime identity is tracked separately in
[[wrds-installed-runtime]].[^path][^stopped]

## Next study requirements

Define how unfillable orders affect targets and portfolio cash before running
a new preregistration. Do not select the historical universe by future outcomes.
Freeze changed rules and runtime explicitly, rehearse them through the existing
queue and preserve this failed journal. Cleanup of the original failed
allocation remains an evidence gap in [[wrds-cleanup-evidence]].[^diagnostic]

[^stopped]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `native`, `queue`, `diagnostic` and `study_action`.
[^diagnostic]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, `terminal_report`, `interpretation`, `repair_status` and `cleanup_limit`; its repair-status field is the historical pre-correction observation.
[^path]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_row`, `crsp_window_path` and `_require_fill`.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py`, boundary arithmetic and verified-extract label-refusal cases.
