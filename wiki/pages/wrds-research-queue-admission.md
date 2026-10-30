---
title: WRDS admission and shared queue ownership
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[crsp-monthly-holdings-validation]]", "[[wrds-proposer-isolation]]"]
source_paths:
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
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "8cfeae2c102164539246bd6ea99b93051e8ccb2b"
  "libs/platform_core/src/platform_core/event_journal.py": "b0660ebf4ebe52161aed9b5e7c425064eac6782e"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "a36df5c2e20b454eabd26f3acfc75f07f1c3485e"
  "libs/covenant_ml/src/covenant_ml/markets/research_records.py": "3a5069fbde15c85b942237313970187f8bff7c00"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "92bab2e0f2d2c371a1d1003cddc1bbf3e0cd69ca"
  "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py": "b898232007d5c14a3aa6acb1f79899a944c1a986"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "92d38ea8211b08cf4480fa72a91e51761d6424bf"
  "tools/fleet/src/fleet/cli/research_queue.py": "aced6395998134dd8d9532dc7243b87745eab0c6"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "tools/fleet/src/fleet/core/research_hpc.py": "ac4a658ffe4ca6beae15ce56df08f712e7bddca2"
  "tools/fleet/src/fleet/core/research_wrds.py": "2130dac9b040de079efcdb5b5b238c6cb6022927"
  "tools/fleet/src/fleet/cli/tick.py": "596da64a093028b74c9bda926b6035da5c64cb72"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py": "c63fd95cfe672e9e198cd1b5cab895e97466dc77"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py": "47eaddf94090471d7dc84e188829de4d8cfaf1dd"
  "libs/covenant_ml/tests/markets/test_crsp_holdings.py": "b5081a606ba9f8d21673ef75c9a7da5a197c6dde"
  "libs/covenant_ml/tests/markets/test_crsp_study.py": "b797984c6422a9c99cfd4db6c09ef92a40c755f9"
  "libs/covenant_ml/tests/markets/test_research_trial.py": "448c595406899be546fe79b550df921a6761dcd8"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "Operator request 2026-10-28: bounded extraction and replay, define substantive evaluation before expansion, then integrate agent-driven research into existing systems; board 35895aac-1b12-4cf5-b868-516196914789"
  - "Corvis live wiki and board discovery 2026-10-28: mcps-codebase agent-task-board-service, non-claude-agent-board-access and project-research; research room and WRDS task threads read before choosing the integration"
fact_checked: 2026-10-30
confidence: medium
---

# WRDS admission and shared queue ownership

- Research entry point: [[wrds-equity-research-protocol]].

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

The controller's `python` executable and `workspace` configuration use absolute
native filesystem paths. HPC3 `root`, `entrypoint`, `image` and `access` use absolute POSIX
paths regardless of the controller's operating system. All six fields reject
parent-directory traversal. The execution command consumes the first two locally;
the frozen staging manifest and compute recipe consume the remote paths. The
access configuration has its own required digest in the native checksum set;
it names private evaluator keys without embedding their contents.[^paths]

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

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. The substantive monthly CRSP study has not run.

[^paths]: `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py`, `decode_research_cluster` and `prepare_bundle`; `libs/covenant_ml/src/covenant_ml/markets/research_execution.py`, `_command`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission", encrypted access configuration. Board task 3261e090-1354-43ae-8703-a8cf5671f5f9 corrects the former use of POSIX validation for local controller paths.
