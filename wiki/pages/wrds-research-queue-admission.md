---
title: WRDS admission and shared queue ownership
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[crsp-monthly-holdings-validation]]", "[[wrds-proposer-isolation]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-resource-preflight-20261101/evidence.json"
  - "tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json"
  - "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json"
  - "tools/hpc3/artifacts/wrds-controller-handover-20261101/evidence.json"
  - "tools/fleet/src/fleet/core/research_handover.py"
  - "tools/fleet/tests/test_research_handover.py"
  - "tools/fleet/src/fleet/cli/research_native.py"
  - "tools/fleet/src/fleet/contracts/research_worker.py"
  - "tools/fleet/tests/test_research_native_process.py"
  - "tools/fleet/tests/test_research_wrds.py"
  - "tools/fleet/tests/test_research_control.py"
  - "tools/fleet/tests/test_research_hpc.py"
  - "tools/fleet/tests/test_research_hpc_final.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/platform_core/src/platform_core/event_journal.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_journal.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_records.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_execution.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "tools/fleet/src/fleet/cli/research_queue.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "tools/fleet/src/fleet/core/research_hpc.py"
  - "tools/fleet/src/fleet/core/research_wrds.py"
  - "tools/fleet/src/fleet/cli/tick.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py"
  - "libs/covenant_ml/tests/markets/test_crsp_holdings.py"
  - "libs/covenant_ml/tests/markets/test_crsp_study.py"
  - "libs/covenant_ml/tests/markets/test_research_trial.py"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-resource-preflight-20261101/evidence.json": "33c89daaca1b44b266192b687fd048fb2afc60f0"
  "tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json": "0fe368911bbfdd7de3d68047e27f13463e86593e"
  "tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json": "3850961553a9736d1533616f23ea2b534ac712e1"
  "tools/hpc3/artifacts/wrds-controller-handover-20261101/evidence.json": "a459d565961af14af29569de9287297e140fd53c"
  "tools/fleet/src/fleet/core/research_handover.py": "446e3e549dbd2b655a142f266f76bf9fe8e9c756"
  "tools/fleet/tests/test_research_handover.py": "949c9f364f842519ee3512543af90c246ad956f4"
  "tools/fleet/src/fleet/cli/research_native.py": "cb18c34d46e6e12e2aab0be7696f7ab8719cda86"
  "tools/fleet/src/fleet/contracts/research_worker.py": "316d6ef45c24e52b3a45ce402e925e9bf3b8cabe"
  "tools/fleet/tests/test_research_native_process.py": "a98984d33895becc39d22bf12635e35857084407"
  "tools/fleet/tests/test_research_wrds.py": "128047efa432f77d6b3b51259356f270f7c8cf99"
  "tools/fleet/tests/test_research_control.py": "bc44f22ae59cb3c33e60e25ea75411107abfa37e"
  "tools/fleet/tests/test_research_hpc.py": "b8bff1bc8cbd62078fa7bc9d8fd72b0494382504"
  "tools/fleet/tests/test_research_hpc_final.py": "f996d9db04d4cc9ad9f5afca643471fedca7e673"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "ad8b3f98f9c9cebf97327fcf513e352969d3327e"
  "libs/platform_core/src/platform_core/event_journal.py": "b0660ebf4ebe52161aed9b5e7c425064eac6782e"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "18c05453efb28a9ab7504e9dd1436bc9a135cfcc"
  "libs/covenant_ml/src/covenant_ml/markets/research_records.py": "4504084dce81b5cddbf86e861fef069108e753a5"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "be8e7dde20f59ac767b92309b670c1a98f65f4f6"
  "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py": "7b3f088839100bbfb58c4085bcd71bbfb4c140db"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "28faea8bff58470c1a8b73230949f45dba82be9d"
  "tools/fleet/src/fleet/cli/research_queue.py": "bc0f30e740470be34280d42eb0ccb266b5130dbd"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "tools/fleet/src/fleet/core/research_hpc.py": "053ee0e19441e00b3323196a5ffe8190b0451cc0"
  "tools/fleet/src/fleet/core/research_wrds.py": "b568e0005a3c960ba5d97d2e141611ea5a0d9a3b"
  "tools/fleet/src/fleet/cli/tick.py": "596da64a093028b74c9bda926b6035da5c64cb72"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "8db552627c0a8cf8962514d5c2a49b8d7dcd1416"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "824b9bad6e08e7d4955dad45f2b8a48bfa4ab6a9"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "325bc527818cfa1c6464541208c1cb15baa5967f"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "2778e68c44c20c724ccd8403312888f168499350"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "ac6baa65e78f71d8bed5728e162e05773551b81d"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "50dcf2341e759b4805b719757302d34165f68fa7"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "Operator request 2026-10-28: bounded extraction and replay, define substantive evaluation before expansion, then integrate agent-driven research into existing systems; board 35895aac-1b12-4cf5-b868-516196914789"
  - "Corvis live wiki and board discovery 2026-10-28: mcps-codebase agent-task-board-service, non-claude-agent-board-access and project-research; research room and WRDS task threads read before choosing the integration"
fact_checked: 2026-11-01
confidence: medium
---

# WRDS admission and shared queue ownership

- Research entry point: [[wrds-equity-research-protocol]].

The completion inspector now takes the exact artifact from the persisted
`run.json`, in addition to the native attempt and workspace host. An old queue
request still pins its original controller: repairing the journal through a
new runtime does not replace that terminal inspection. The November 1 recovery
and recorded controller handover are tracked in [[wrds-baseline-readiness]].[^handover]

[^handover]: `tools/fleet/src/fleet/core/research_hpc.py`, `poll`; `tools/fleet/tests/test_research_hpc.py`, `test_complete_attempt_requires_fresh_native_identity_and_record`. Board task `04c0cdbe-278f-41bb-8141-a170fc0e5324` retains original request and native allocation identities during controller replacement.

The published handover implementation adds an evaluator-owned, content-addressed
plan beside the original request. It permits changing only the interpreter;
policy, trial, script, cluster configuration, journal and attempt directory
must remain identical. The plan pins the recipe, checksums, expected native
job and result digest. The existing installed-controller inspection verifies
the replacement before the same completion verifier reads the native result.
Handover refuses unsettled validation and final-grant work before any submission
or lifecycle transition. A verified service report records both request and
controller identities; only then is an exclusive marker written under the
existing poll lock. Subsequent queue ticks repeat verification and retain the
plan digest in their observation.[^handover-implementation]

The first check stopped at lint (poll complexity and an unmarked regex).
The terminal verifier was lifted into `_verified_completion`; the repaired
check passed 2,627 tests with 100% statement and branch coverage, and independent
audit verified both repair criteria. The separate Linux execution lane passed
97 real host cases with zero skips before the fleet release advanced.[^handover-check]

The installed queue adapter now uses `wrds-queue-b53b0ea34`, built on Loki
from published source with 24 hash-pinned wheels. The maintained inspector
verified all 485 application files across its five first-party wheels, its
generated symbol/version check and dependency consistency. All four original
requests have verified handover markers targeting the measured
`wrds-selection-recovery-f38df09fe` proposal controller. The scheduled runner
closed all four original receipts through the same queue and resource lease;
each completion observation includes the handover digest and original native
allocation. The last closure was at 10:21:26 UTC.[^handover-live]

Installed CLI probes against isolated copies refused changed request bytes,
artifact checksum, native allocation and result digest without creating a
marker. The original request hashes and full journal output remained unchanged
across activation and those probes. Six unclaimed references were withdrawn
before any native submission, then replaced with corrected references to the
same admitted trials. This is still a synthetic operational rehearsal; no
licensed final grant was released.[^handover-live]

The first fresh normal request completed through that same scheduled path at
10:36:38 UTC. Queue receipt `eecf34ae` identifies native job `58719001` and the
same canonical result digest as the fresh native inspection, with a null
handover digest. The normal-path audit query matches exactly one receipt;
changed result, attempt and handover expectations each match zero. At the
published evidence revision, the fleet suite passed 2,634 tests with full
statement and branch coverage, and both updated wiki pages passed their
code-path audits without errors or warnings. Recovery passed independent audit
at 10:46:40 UTC. All ten synthetic queue receipts had passed by 11:18:37 UTC;
the maintained selector subsequently published winner and refusal evidence.
Selection passed independent audit `72d4a03d` at 12:27:53 UTC on November 1;
the recovery and selection milestones are now independently verified.
[[wrds-baseline-readiness]] records that verdict and the licensed allocation
preflight. CPU and RAM are mandatory evaluator-owned policy
fields, included in the journal identity and used by validation and final
recipes. Proposals cannot override them.[^normal-live][^selection-complete][^admission][^paths]

The frozen resource controller and matching image are now installed and
verified. Native preflight on November 1 at 13:30:03 UTC accepted the generated
four-CPU, 128-GiB, 121-minute recipe through the same controller configuration.
It used synthetic ciphertext and queued no evaluation. The existing queue
adapter remains installed; this new interpreter is an evaluator-owned setting
for future requests, and old requests retain their original runtime identities.
This measurement does not execute the licensed baseline or replace the prior
shared-queue rehearsal. Independent resource audit `967229c0` verified task
`42a8d507` at 13:44:22 UTC on November 1.[^resource-live]

[^resource-live]: `tools/hpc3/artifacts/wrds-resource-preflight-20261101/evidence.json`, frozen runtime, native image build, service-authored preflight, synthetic scope and exact input digests. Its retained audit query joins controller, image, staged request and scheduler reports; all six altered-evidence probes return zero.

[^selection-complete]: `tools/hpc3/artifacts/wrds-selection-recovery-20261101/selection.json`, recovery audit `eefb349e`, ten native/queue pairs and four service-authored selection outcomes. [[wrds-baseline-readiness]] records the remaining licensed-study prerequisites.

[^normal-live]: `tools/hpc3/artifacts/wrds-controller-handover-20261101/normal-completion.json`, native report `1caa931e`, queue receipt, verification jobs and audit queries; `audit-probes.json`, normal completion and three altered normal-evidence queries. Fleet check `90c37d46` at API `52345cc13` passed 2,634 cases and skipped 44 separate host-lane cases; the previously measured execution lane passed 97 cases with zero skips.

[^handover-live]: `tools/hpc3/artifacts/wrds-controller-handover-20261101/evidence.json`, service report `6322f701`, wheel identities, handover plans, preservation and refusal measurements. The evidence index distinguishes service-authored reports from session-captured probes and placement readbacks. Operational review is tracked by `04c0cdbe` separately from the verified lint repair.

[^handover-implementation]: `tools/fleet/src/fleet/core/research_handover.py`, `poll` and `record`; `tools/fleet/src/fleet/core/research_hpc.py`, `poll` and `_verified_completion`; `tools/fleet/src/fleet/cli/research_native.py`, `main`; `tools/fleet/src/fleet/contracts/research_worker.py`, `decode_observation`. `tools/fleet/tests/test_research_handover.py` covers unchanged original requests, altered inputs and runtime reports, a real failed controller process, exclusive marker/readback and publication failure. `tools/fleet/tests/test_research_hpc.py` rejects unsettled attempts and final-grant handover.

[^handover-check]: Live fleet job `cbcfa409-1a43-4242-9a92-06e69583f8f2` at API `3f3f51d2e8f3f9a4f596e9d403c061ba2b969866` stopped before tests with Ruff C901/RUF043. Repair task `9dc6f80c` passed independent audit `44198e74` at `2026-11-01T09:59:06.953Z`. Fleet jobs `b240c7d5` at `a9340a200` and `cb35e08b` at `b53b0ea34` each passed 2,627 tests, with 44 host cases skipped in the ordinary lane and 100% statement and branch coverage. Host execution job `69740785-701e-40ee-8eec-f3ff0de6598d` at `b53b0ea343de96df607ad7dbc877c07ec5220668` passed 97 cases, zero skips, at `2026-11-01T10:09:22Z`. These checks are distinct from the four real queue closures required by `04c0cdbe`.

The fleet research lane has an implementation in the existing fleet package:
it claims shared dispatch rows, uses existing resource leases during bounded
controller operations, and polls native jobs through maintained controllers.
Four requests may remain active, with the oldest renewed receipt polled each
tick. Native scheduler waiting does not occupy a local execution slot.
Board task `3dff6fbf` verified deployed automatic reconciliation, and task
`f5e2befd` verified the queue contract. Their audits closed VERIFIED on
2026-10-29 UTC. The hpc3 wiki page `wrds-installed-controller-verification`
holds the runtime and live receipts; this page describes the code seams.[^queue]

The queue reference contains backend, exact private specification digest and
attempt. The native adapter checks tenant binding and input hashes on each
poll. HPC3 keeps proposal admission and its attempt journal; WRDS keeps held
submission and its durable receipt. Transport failures retain those identities.
Completion requires fresh native evidence and verified RunRecord metadata;
no research rows or credentials are placed on the board.[^queue][^controllers]

Each native CLI observation measures the executing Python path and SHA-256.
The queue requires these fields, refuses a path different from its configured
placement, and retains both fields in progress or closing evidence. A runtime
installation report alone cannot establish which interpreter executed a poll;
the closing observation supplies that measurement for either backend.[^execution]

[^execution]: `tools/fleet/src/fleet/cli/research_native.py`, `main`; `tools/fleet/src/fleet/contracts/research_worker.py`, `decode_observation` and `encode_observation`; `tools/fleet/src/fleet/cli/research_queue.py`, `reconcile`; `tools/fleet/tests/test_research_native_process.py`, actual rendered-shell and child-process execution. Board task `da2a55dc-dc5a-4f4b-9bb7-81fbcf75f320`; audit `a814e463-b0ea-49b3-86a9-feb31b27c8e2` identified the missing closing-controller measurement. Fleet check `4c6aa1e0-0188-4fdd-82f1-d420407e4ea6` passed 2,578 tests at API `a0550032d5ea977d58528c45e0ec72aa610cb855`, with 100% statement and branch coverage.

WRDS submission, collection and final verification occupy separate polls.
Collection has a 900-second process deadline after a 60-second status call,
inside the existing 1,020-second native tick deadline. A successful collection
process still returns `running`; the next poll compares the native completion,
every collected output and the RunRecord before returning `complete`. An
existing partial directory fails verification and is never overwritten by the
poller. Explicit recovery must preserve that failed transfer before collecting
again; durable submission intent continues to prevent another native
submission.[^wrds-phases]

[^wrds-phases]: `tools/fleet/src/fleet/core/research_wrds.py`, `poll`, `_reserve` and `_verify_collected`; `tools/fleet/tests/test_research_wrds.py`, separate bounded polls for newly submitted and adopted jobs, incomplete receipts and changed files; `tools/fleet/tests/test_research_control.py`, actual controller subprocess success, nonzero exit and timeout; `tools/fleet/src/fleet/cli/research_queue.py`, `_script` and `reconcile`. Board task `da2a55dc-dc5a-4f4b-9bb7-81fbcf75f320`; fleet check `2c441d42-2aea-437a-806a-e19d76ca53da` at API `0ffb98731e70c234e4052d4319332faf7496a60e` passed 2,574 tests with 100% coverage on Pendragon.

The private HPC3 document requires explicit `phase: "validation"` with
`native_attempt`, or `phase: "final"` with `remote_manifest`. Old documents
without a phase are refused; rebuilding and re-pinning them is explicit.
Final polling calls the maintained `submit-final`/`reconcile-final` commands
against the delivered grant's fixed directory. Its exclusive `native` directory
records submission intent, and the cluster report task must match the shared
queue task. The queue grants no final authority: [[wrds-final-scoring]] describes
the original journal and consumed grant checks performed by those commands.
Both phases retain the same content-addressed queue reference and resource lease
mechanism.[^phase]

[^phase]: `tools/fleet/src/fleet/core/research_hpc.py`, `_require_phase` and `poll`; `tools/fleet/tests/test_research_hpc.py` and `test_research_hpc_final.py`, routing, strict fields, task binding and real rendered queue-shell process cases.

The controller's `python` executable and `workspace` configuration use absolute
native filesystem paths. HPC3 `root`, `entrypoint`, `image` and `access` use absolute POSIX
paths regardless of the controller's operating system. All six fields reject
parent-directory traversal. The execution command consumes the first two locally;
the frozen staging manifest and compute recipe consume the remote paths. The
access configuration has its own required digest in the native checksum set;
it names private evaluator keys without embedding their contents.[^paths]

`ResearchCluster.binds` is now required operator configuration, copied unchanged
into both validation and final image recipes. An empty list removes the former
implicit `/pub` dependency for home-only data; shared-data paths must be declared.
The shared launcher suppresses site bind-paths, hostfs and ambient mount variables.
Older frozen configurations are not rewritten or given a compatibility default.
Standard home and temporary mounts remain, so this is mount ownership, not the
holdout access boundary.[^mounts]

The implemented typed proposal/admission boundary accepts a proposal that
names its hypothesis, parent, protocol digest, allowed features, model parameters
and resource budget. Admission records the attempt **before** extraction or job
submission. Link proposal → snapshot/query → code/image → scheduler attempt →
RunRecord → review, including refusal, crash, timeout, null and negative results.
One candidate with five seeds is one hypothesis with five tracked jobs.[^request]

`research_records` reuses the benchmark configuration codec; `research_journal`
binds the frozen policy to `platform_core.event_journal`, lifted from the existing
execution rehearsal and used by both consumers. Raw object proposals are recorded
before decoding. Policy refusals remain visible, and admitted candidate/seed pairs
stay consumed after failure. The lifecycle records submission intent, unique job
identity and an identity-checked shared RunRecord. `scripts.research_proposals`
exposes init, propose, status, submit and reconcile operations. Submission uses the
maintained cluster staging and job commands. The submission boundary compares typed
proposal values, preserving raw admission bytes while accepting JSON key reordering;
changed proposal values remain refused. Reconciliation requires native Slurm
accounting and verified results through the cluster evidence command. It also
requires the measured encrypted bundle digest and native manifest checksum to
match admission, and the measured bundle partition to be train-validation. Transport
ambiguity remains unresolved until exactly one matching ledger row is found; no
automatic resubmission is allowed. The validation worker runs the monthly study
in [[crsp-monthly-holdings-validation]] with the existing trainers and RunRecord
publisher, after checking that the staged CRSP protocol is exactly the admitted
design; it publishes every arm's
monthly accounting, the fold counts and the paired intervals as observations.
These are provenance checks over owner-writable evidence, not protection against
an adversary controlling the account.[^execution][^holdings] This control layer refuses final-partition proposals;
the access boundary is the proposal process in [[wrds-proposer-isolation]].
The CLI's policy is evaluator-owned configuration, not a field agents may replace
in a proposal.[^admission]

[^request]: Board task 35895aac-1b12-4cf5-b868-516196914789, § "OPERATOR", submitted 2026-10-28T16:38:53Z; the study rules are this session's proposed design, introduced in API commit 3358191792a4a655825b5647694fc0430cf7bd33 and calendar-corrected in 78b009119, not quotations or empirical findings.

[^queue]: `tools/fleet/src/fleet/cli/research_queue.py` functions `tick` and `reconcile`; `tools/fleet/src/fleet/cli/tick.py` function `plan_tick`. Board task 3dff6fbf-e09a-430f-b13e-5dccb505ba80 tracks rollout separately from implementation.

[^controllers]: `tools/fleet/src/fleet/core/research_control.py` functions `read_specification` and `verified_input`; `tools/fleet/src/fleet/core/research_hpc.py` and `tools/fleet/src/fleet/core/research_wrds.py`, function `poll` in each.

[^admission]: `libs/covenant_ml/src/covenant_ml/markets/research_records.py` frozen contracts; `libs/covenant_ml/src/covenant_ml/markets/research_journal.py` admission and lifecycle reducers; `libs/platform_core/src/platform_core/event_journal.py` transactional persistence; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Board task 399048d8-b1d6-4e12-a4d3-a0ce2281c810.

[^execution]: `libs/covenant_ml/src/covenant_ml/markets/research_execution.py` submission and reconciliation; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py` `execute_trial`, the monthly validation worker since board task 8f7247a1; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission", execution settings and limitations. Board task 6d2d086c-5242-482b-ac81-8a61dd8e354d.

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. This was synthetic integration evidence; the later licensed attempt stopped before fitting, as recorded in [[wrds-baseline-20261101]].

[^paths]: `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py`, `decode_research_cluster` and `prepare_bundle`; `libs/covenant_ml/src/covenant_ml/markets/research_execution.py`, `_command`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission", encrypted access configuration. Board task 3261e090-1354-43ae-8703-a8cf5671f5f9 corrects the former use of POSIX validation for local controller paths.
[^mounts]: `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py`, `decode_research_cluster` and `prepare_bundle`; `libs/covenant_ml/docs/MARKET_RESEARCH.md`, § "Frozen research proposal admission"; HPC3 wiki `declared-image-mounts`, native probe identities and limits, board task `0a0e5913`.
