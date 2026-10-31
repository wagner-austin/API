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
  - "artifacts/research-queue-verification-20261028/provenance.json"
  - "artifacts/research-queue-fresh-20261029/provenance.json"
  - "artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json"
source_git_blobs:
  "src/hpc3/cli/execution_inspect.py": "fd0d98dc12dba896ebc552149a7da69f76d4316e"
  "src/hpc3/core/execution_evidence.py": "bf31a1091302ae2786425afa629677fccf07f681"
  "src/hpc3/core/status.py": "2076960193a9b3b1322908b13f8abaeeaad58fd7"
  "artifacts/wrds-controller-verification-20261028/provenance.json": "b678f473462c24db2cbd1e510ffe6c034e36a9bb"
  "artifacts/research-queue-verification-20261028/provenance.json": "3a968954ceb6b17ac3798dbf0861b744fa098ea4"
  "artifacts/research-queue-fresh-20261029/provenance.json": "da6266d1583b6ad0bab4a66f70de69aafdd8db83"
  "artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json": "3aa4b35290d76b1a8f3c404521c96b0bddb96291"
provenance:
  - "Board task 8a113204-42d0-41a0-8094-73776bae8846; native reports by hpc-report-8a113204, prefix HPC3 ATTEMPT LOAD-REPORT, jobs 58578791 and 58578878"
  - "API libs/covenant_ml/src/covenant_ml/markets/research_execution.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421; outside this wiki workspaceRoot"
  - "API libs/covenant_ml/tests/markets/test_research_execution.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421; fleet b0e902f6-b54c-41eb-a1d9-03c5d0c84131 on Loki: 4511 passed, 100% coverage"
  - "API libs/covenant_ml/src/covenant_ml/markets/research_cluster.py and research_trial.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421: fixed resources, no requeue, validation partition; outside this wiki workspaceRoot"
  - "MCPs fleet-mcp/src/tools/dispatch-submit.ts at 14b0c4c7d5467ad1275570d726db66180f168bb6, blob 609ece4ff9bfeee7384f1fe8df54c9098727350c; compared with served dispatch_submit schema on 2026-10-28; outside this wiki workspaceRoot"
  - "API libs/covenant_ml/scripts/research_proposals.py at bda8ca2af532d89b301b56cc0b55a4346bb6e421: explicit status/submit/reconcile commands; outside this wiki workspaceRoot"
  - "Board task f12aa47e-328f-4473-bb89-cc54c49fed30; hpc-report-f12aa47e independently remeasured retained job 58578878 on 2026-10-28; no new trial submitted"
  - "Board task 3dff6fbf-e09a-430f-b13e-5dccb505ba80; API rolled consumer f4064ef451f9dc66ecbd09632f0bfca22fd3718e; MCPs deployment a09431951c0cfa1eb5b67e2ce11c2bc5e9938e6b; automatic scheduler and live queue receipts captured in the research-queue provenance artifact"
  - "Board task 8b19319c-1aa0-4e6a-bc83-0c36bd3b50bb; fresh queue fb62c111-f5c8-48f6-a790-899e68f76920 submitted and reconciled native allocation 58586013 through fleet-research-diphtheria on 2026-10-29 UTC; independent program-owned reports by hpc-report-8b19319c and load-report-8b19319c"
  - "MCPs execution/tests/served/research-queue-replay.test.ts at 06512f6450c960e52409448f306c1254ed633fb6; fleet c469a100-f103-4e26-a246-396d81b5ee04 on Demure passed this case in 2.497 seconds on 2026-10-29 UTC; outside this wiki workspaceRoot"
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
HPC tools; WRDS uses Grid Engine through the WRDS CLI. Prepared requests now
also enter through Corvis `dispatch_submit`, using `command: "research"`,
project `research/hpc3` or `research/wrds`, controller node, owning task and
`researchRequest: {backend, specSha256, attempt}`. The reference names exact
private specification bytes; it carries no shell command or data. A board task
alone remains coordination, not a scheduler submission.[^routes][^queue]

The existing fleet scheduler invokes the frozen native adapter through the
rolled consumer every 180 seconds. Its shared lease bounds a controller
operation; waiting for Slurm or Grid Engine consumes no local worker slot.
The adapter reuses the proposal journal, consumed candidate/seed budgets and
maintained WRDS lifecycle. Direct CLI submissions still require explicit
reconciliation unless adopted by a prepared queue request. Disconnecting the
agent does not cancel an accepted native job; missing or ambiguous receipts
cannot authorize another submission.[^queue][^inspect]

Automatic ticks by `fleet-research-diphtheria` completed both retained pilots:[^queue]

| Backend | Shared queue receipt | Native job | Completed UTC |
|---|---|---|---|
| HPC3 | d925f987-1e9b-4ff2-aca2-0a85c1a3b07d | 58578878 | 22:28:06 |
| WRDS | 84ae382d-ff5f-4292-8d8a-3271b8fa30c7 | 40534563 | 22:31:33 |

Both ticks exited zero without timing out and posted their outcomes to the
owning board task. Repeated submissions returned the same completed receipts;
separate repeated native inspections returned the same allocation IDs and
RunRecord hashes. The metadata artifact contains the live queue trails,
scheduler rows, frozen controller's 23 wheel hashes and offline hashed
requirements. No new allocation or extraction was requested by this check.
This supersedes the manual-only route measured earlier under f12aa47e.[^queue]

The fresh inspection for task f12aa47e revalidated job 58578878's zero-exit
allocation and the same result/RunRecord hashes recorded above. The installed
journal still contained three completed attempts and one refusal. This read
consumed no additional candidate or seed.[^recheck]

### Fresh submission through the shared queue

On 2026-10-29 UTC, request `admitted-5` used the last unused candidate/seed pair
under the existing policy: the 11-tree candidate and seed 23. Admission appended
attempt 5 to the original journal before queue submission. Queue receipt
`fb62c111-f5c8-48f6-a790-899e68f76920` was submitted at 01:21:50, claimed by the
scheduled runner at 01:24:09 and closed passed at 01:27:37. The runner submitted
new Slurm allocation **58586013**, which completed on `hpc3-22-05` in 11 seconds,
exit `0:0`, one CPU, no GPU, on the free partition.[^fresh]

Both automatic scheduler ticks exited zero without timing out. Repeating the
request before claim and after completion returned the same queue receipt.
The verified evidence includes eight input checksums, native accounting and
result/RunRecord hashes. Canonical RunRecord hash
`49b7a44e64b76b7d153a6a3fb1a3f1805ea328720fbc855ca6e0bc892d75af17`
is distinct from the file-byte hash because the queue re-encodes the decoded
record. Both encodings are recorded in the metadata artifact.[^fresh]

All four earlier admission rows compare equal before and after this run,
including the refused duplicate. The policy now has four completed candidate/seed
pairs and that refusal; no budget or history was reset. A separate retained-WRDS
inspection returned allocation 40534563 and its unchanged RunRecord hash without
requesting another extraction. An independent HPC3 reinspection also returned
58586013 and its unchanged canonical RunRecord hash. This fresh test extends the retained-job evidence
above to automatic submission of a new allocation.[^fresh]

The first verification audit, `182e4ebb`, rejected A2's evidence: counting queue
rows did not exercise repeat submission, and a worker-written comparison file
did not independently read the admission journal. The served replay case now
calls `dispatch_submit` twice through the live proxy, requires the same terminal
receipt and native digest, and finds exactly one queue row. Three reads through
the installed controller's `status` command retain all five rows with the same
full-journal digest `727b765412d7ce3b19edfc71b45311135f06d72c1486e307a641f094f5bd9ab6`.
The original four rows match the pinned pre-run snapshot's normalized digest
`adec96603bc29d395be58a79d5f6c731972b42295b55b521f4d44a133a9f95ca`.
This case passed; the full served run remained red on three separate cases:
dispatch waiting, minutes evidence and transcriber writes. A case pass does not
make the whole suite or the board audit a pass.[^replay]

### Column discovery for the monthly adapter

Metadata-only WRDS job **40538087** read 331 columns across fourteen
`crsp_a_stock` relations. It reused the existing grid upload, submit and verified
artifact-download functions, saving the exact read-only SQL and checksummed CSV
privately. No security observations were downloaded.[^schema]

The candidate inputs are `stkdlysecuritydata` for daily values and return flags,
`stksecurityinfohist` for dated security classifications, `metaexchangecalendar`
for calendar fields and `stkdelists` for delisting fields. These are schema
findings, not a completed data contract. The next step is to verify CRSP's field
and return conventions, including whether a selected return already incorporates
delisting, before defining the extraction joins. The monthly adapter must also
retain missing-return flags and historical classification intervals; column
existence alone does not establish how to interpret them.[^schema]

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
[^queue]: `artifacts/research-queue-verification-20261028/provenance.json`, fields `runtime`, `scheduler_runs` and `pilots`. Queue receipts were read through live Corvis tools, scheduler rows from production `hub_runner_runs`, and runtime hashes from local build metadata. API `tools/fleet/src/fleet/cli/research_queue.py` and `core/research_hpc.py` / `research_wrds.py` at f4064ef451f9dc66ecbd09632f0bfca22fd3718e implement the existing lease and native-command integration. MCPs `scripts/fleet/hub-jobs.json` at a09431951c0cfa1eb5b67e2ce11c2bc5e9938e6b supplies its schedule. These cross-package source files lie outside this wiki's workspaceRoot.
[^fresh]: `artifacts/research-queue-fresh-20261029/provenance.json`, fields `queue`, `scheduler_runs`, `admission_history`, `native`, `replay` and `wrds_reinspection`. Live Corvis receipts and program-owned native reports are distinguished from local journal comparisons; numerical observations and licensed source rows remain private.
[^replay]: MCPs served test and fleet job named in provenance. The test checks the existing terminal receipt before submission, hashes the saved pre-run snapshot and live controller rows, compares every full-journal digest, and verifies a single queue reference after both submissions. Fleet transcript lines 273-274 record the measured digests and passing case. Audit `182e4ebb` is the earlier rejection, not an acceptance of this repair.
[^schema]: `artifacts/research-queue-fresh-20261029/crsp-schema-discovery.json`, fields `job_id`, `query_sha256`, `artifact_sha256`, `relation_column_counts`, `candidate_fields` and `limits`. This is a local metadata measurement with verified remote bytes, not a vendor semantic specification.
