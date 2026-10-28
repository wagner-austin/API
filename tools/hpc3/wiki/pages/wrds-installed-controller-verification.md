---
title: Installed WRDS research controllers submit and reconcile bounded validation
tags: [wrds, validation, provenance, operations]
hubs: [operations]
related: ["[[wrds-repeatable-attempts]]", "[[one-ledger-per-repository]]"]
source_paths:
  - "src/hpc3/cli/execution_inspect.py"
  - "src/hpc3/core/execution_evidence.py"
  - "src/hpc3/core/status.py"
  - "artifacts/wrds-controller-verification-20261028/provenance.json"
source_git_blobs:
  "src/hpc3/cli/execution_inspect.py": "fd0d98dc12dba896ebc552149a7da69f76d4316e"
  "src/hpc3/core/execution_evidence.py": "f0347915a8720f9a95d62be1221340a876f31e49"
  "src/hpc3/core/status.py": "2076960193a9b3b1322908b13f8abaeeaad58fd7"
  "artifacts/wrds-controller-verification-20261028/provenance.json": "b678f473462c24db2cbd1e510ffe6c034e36a9bb"
provenance:
  - "Board task 8a113204-42d0-41a0-8094-73776bae8846; native reports by hpc-report-8a113204, prefix HPC3 ATTEMPT LOAD-REPORT, jobs 58578791 and 58578878"
  - "API libs/covenant_ml/src/covenant_ml/markets/research_execution.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421; outside this wiki workspaceRoot"
  - "API libs/covenant_ml/tests/markets/test_research_execution.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421; fleet b0e902f6-b54c-41eb-a1d9-03c5d0c84131 on Loki: 4511 passed, 100% coverage"
  - "API libs/covenant_ml/src/covenant_ml/markets/research_cluster.py and research_trial.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421: fixed resources, no requeue, validation partition; outside this wiki workspaceRoot"
  - "MCPs fleet-mcp/src/tools/dispatch-submit.ts at 14b0c4c7d5467ad1275570d726db66180f168bb6, blob 609ece4ff9bfeee7384f1fe8df54c9098727350c; compared with served dispatch_submit schema on 2026-10-28; outside this wiki workspaceRoot"
  - "API libs/covenant_ml/scripts/research_proposals.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421: explicit status/submit/reconcile commands; outside this wiki workspaceRoot"
  - "Board task f12aa47e-328f-4473-bb89-cc54c49fed30; hpc-report-f12aa47e independently remeasured retained job 58578878 on 2026-10-28; no new trial submitted"
fact_checked: 2026-10-28
confidence: high
---

# Installed WRDS research controllers submit and reconcile bounded validation

The installed proposal controller uses the existing admission journal and
HPC3 stage, submit and inspect commands. Its evidence reader resolves exactly
one ledger entry for a policy/attempt identity. Completion additionally
requires successful native accounting, verified result and RunRecord bytes,
matching fingerprints, output timestamps inside the allocation and checksum
success lines from the compute job.[^inspect][^measure]

## Measured installed runs

The installed tools also re-read the original extraction on WRDS: job 40534563
still reports failed=0, exit_status=0 and matching output hashes. This was a
remote verification of retained evidence, not a new extraction.[^wrds]

Both jobs used the same 505-row CRSP snapshot, which produces 504 examples,
and the same SIF hash `7d91f3436403abca44cd2265340f3960da9a27972396cb8e0877f5499498e550`.
Each used one CPU, no GPU, eight verified files and thirteen shared observations.
They are distinct diagnostic configurations, not identical-seed replays.[^live]

| Attempt | Job | Controller | Diagnostic | Native result |
|---|---|---|---|---|
| 3 | 58578791 | 2e445e25e | Existing candidate, seed 23 | COMPLETED, 0:0, 11 seconds, hpc3-20-19 |
| 4 | 58578878 | bda8ca2af | Reordered JSON, 11-tree candidate, seed 11 | COMPLETED, 0:0, 14 seconds, hpc3-20-19 |

The second controller fixes a measured admission/submission mismatch: raw
JSON key order was previously compared with codec output order. It now
compares decoded proposal values, preserving the original request in the
journal and refusing altered values. Nested and top-level reordering pass
the regression and the live attempt above.[^repair][^live]

The installed controller is `/home/corvis/research-control/wrds-validation-bda8ca2af/bin/python`,
with its entrypoint in the adjacent `entrypoint/` directory. The private policy,
journal, configurations and reports remain under
`/home/corvis/research-data/wrds/experiments/admitted-validation-20261028/`.
The metadata record pins the controller wheel, source archive and installed
requirements. The compute SIF remains the earlier immutable evaluator.[^live]

## Boundaries of this evidence

### Agents and the queues

Shell-capable agents on the controller can submit to the native queues using
the installed commands: HPC3 uses Slurm through the maintained admission and
HPC tools; WRDS uses Grid Engine through the WRDS CLI. The fleet's served
`dispatch_submit` accepts only `check`, `lint`, `test` and `build-bases`.
It does not accept a research job payload. A board task is coordination,
not a scheduler submission.[^routes]

The proposal controller has no unattended consumer or reconciliation daemon.
Agents explicitly call `reconcile` for submitted attempts, preserving the
existing journal and consumed candidate/seed budgets. Disconnecting the agent
does not cancel an accepted native job. Recovery inspects the recorded identity
and refuses to guess when the submission ledger is missing or ambiguous.
The package README's **Agent submission routes** section gives the installed
status command and routes to the existing lifecycle references.[^routes][^inspect]

The fresh inspection for task f12aa47e revalidated job 58578878's zero-exit
allocation and the same result/RunRecord hashes recorded above. The installed
journal still contained three completed attempts and one refusal. This read
consumed no additional candidate or seed.[^recheck]

### Evaluation and evidence limits

The worker scores chronological validation and uses training rows for early
stopping. Final-fold exclusion is logical; the shared account can still read
the source snapshot. Monthly portfolio evaluation and a restricted final-holdout
identity are not implemented by this diagnostic.[^repair]

Recipes disable requeue. The evidence reader uses allocation-only `sacct`
and does not request historical duplicate records. Its one-row check is a
query-result invariant, not proof that a job ID has only one historical
execution. Results and local journals remain owner-writable evidence.[^query][^measure]

[^inspect]: `src/hpc3/cli/execution_inspect.py:23` `main`.
[^measure]: `src/hpc3/core/execution_evidence.py:77` `allocation_document` and `:154` `measure`.
[^query]: `src/hpc3/core/status.py:33` `sacct_commands`; the recipe constructor named in provenance sets `requeue=False`. Upstream semantics are archived in tech-wiki pages `slurm-accounting-allocation-and-requeue-records` and `slurm-batch-success-and-artifact-evidence`.
[^live]: `artifacts/wrds-controller-verification-20261028/provenance.json`, fields `attempts`, `controller_build_hashes` and `interpretation`; program-authored board reports named in provenance.
[^repair]: API commit bda8ca2af532d89b301b56cc0b55a4346bb6e421, `research_execution.py` function `_require_trial` and regression `test_reordered_json_preserves_raw_admission_and_executes`; source and fleet evidence are named in provenance. API wiki `wrds-equity-research-protocol` records the separate monthly study and holdout requirements.
[^wrds]: Board task 8a113204-42d0-41a0-8094-73776bae8846, program-authored post by load-report-8a113204 at 2026-10-28T19:57:42.310Z, native accounting and output digest lines.
[^routes]: MCPs `dispatch-submit.ts` function `registerDispatchSubmit` and the served schema named in provenance; API `research_proposals.py` function `main`, plus `research_execution.py` functions `submit_attempt` and `reconcile_attempt` at the source commit named in provenance. The WRDS lifecycle is documented in [[wrds-repeatable-attempts]].
[^recheck]: Board task f12aa47e-328f-4473-bb89-cc54c49fed30, program-authored HPC3 ATTEMPT LOAD-REPORT by hpc-report-f12aa47e; controller status saved privately as `agent-route-journal-status.json` beside the installed study journal on 2026-10-28. Status is a local file reading; the report measures live cluster evidence.
