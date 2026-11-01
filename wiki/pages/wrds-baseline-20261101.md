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
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "13031c17a0e67e2f4fc75d2486d2b6689092536a"
  "libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py": "360c80d81411228346f695f26427b80ce15f6276"
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

The first correction at API `d3fac047a` checked label execution endpoints and
refused the conventional terminal record as a fill. The subsequent version-2
rule distinguishes confirmed rejected new purchases from missing evidence:
their allocations remain cash in the same outcome path used by labels and
holdings. Contradictory evidence still fails, and held terminal losses remain
owed values. Current rules are in [[crsp-monthly-holdings-validation]].[^path][^tests]

Neither correction rewrites the stopped study or supplies a successful result.
Installed-runtime identity is tracked separately in
[[wrds-installed-runtime]].[^path][^stopped]

## Next study requirements

Version 2 needs new verified protocol/bundle and runtime identities before a
new preregistration. The staged licensed bundle embeds version 1 and cannot
be silently reinterpreted. Rehearse changed inputs through the existing queue
and preserve this failed journal. Cleanup of the original failed
allocation remains an evidence gap in [[wrds-cleanup-evidence]].[^diagnostic]

[^stopped]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/stopped.json`, `native`, `queue`, `diagnostic` and `study_action`.
[^diagnostic]: `tools/hpc3/artifacts/wrds-licensed-baseline-20261101/return-diagnostic.json`, `terminal_report`, `interpretation`, `repair_status` and `cleanup_limit`; its repair-status field is the historical pre-correction observation.
[^path]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `_terminal_row`, `crsp_window_path` and `_require_fill`.
[^tests]: `libs/covenant_ml/tests/markets/test_crsp_execution_boundaries.py`, boundary arithmetic and verified-extract cash labels with unchanged selection.
