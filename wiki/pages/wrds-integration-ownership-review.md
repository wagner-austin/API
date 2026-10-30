---
title: WRDS integration ownership review
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]", "[[crsp-monthly-holdings-validation]]", "[[wrds-proposer-isolation]]", "[[wrds-final-scoring]]"]
source_paths:
  - "CODE_STANDARDS.md"
  - "libs/platform_core/src/platform_core/stage_manifest.py"
  - "libs/covenant_ml/tests/markets/test_research_execution.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/paired.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/splits.py"
  - "libs/platform_core/src/platform_core/run_record.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/platform_core/src/platform_core/event_journal.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_journal.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_records.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py"
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
  "CODE_STANDARDS.md": "474681549e2caa9dc26a049f53c793ab17eaf830"
  "libs/platform_core/src/platform_core/stage_manifest.py": "661ec21bf8846370ad93aefa4460371d1981547e"
  "libs/covenant_ml/tests/markets/test_research_execution.py": "6c83658eb2e0447a2a06fd12d23e6c3b138c1d30"
  "libs/covenant_ml/src/covenant_ml/evaluation/resampling.py": "6d95079f002b370fa9a8920c3d361a4f803e6136"
  "libs/covenant_ml/src/covenant_ml/evaluation/paired.py": "930ff0876202a68c51454352dc63c0b7d80160c8"
  "libs/covenant_ml/src/covenant_ml/evaluation/splits.py": "1a5b1a1a94e4666d039ea21557480b66eb95bbac"
  "libs/platform_core/src/platform_core/run_record.py": "7d3ca55d81674311342b179cbf476a3d61b14dd6"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "e30492e1ef4ae30e3c3391560a066f51805d6250"
  "libs/platform_core/src/platform_core/event_journal.py": "b0660ebf4ebe52161aed9b5e7c425064eac6782e"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "a36df5c2e20b454eabd26f3acfc75f07f1c3485e"
  "libs/covenant_ml/src/covenant_ml/markets/research_records.py": "3a5069fbde15c85b942237313970187f8bff7c00"
  "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py": "bd2c9fd8c7810948c0702ac5f9d5a502c7054289"
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
  "libs/covenant_ml/tests/markets/test_research_trial.py": "2b0cd1e2362f58475205774e7dc320b2ae8991d2"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "Ownership review 2026-10-29: MCPs source snapshot 06512f6450c960e52409448f306c1254ed633fb6 is external to this API wiki's source_paths; live queue, installed controller and board reads are identified separately in the review footnotes."
fact_checked: 2026-10-30
confidence: medium
---

# WRDS integration ownership review

Research entry point: [[wrds-equity-research-protocol]].

The 2026-10-29 review traced the research submission, recovery and result path,
including its failure tests. It found no compatibility shim, alternate backend
fallback, passthrough alias or new type alias in the reviewed integration.
This is a bounded source review, not a claim about every file in either repository.
The named guard rules remain `hook-fallback-dispatch`,
`shim-compatibility-marker`, `passthrough-alias`, `qr-legacy-validators`
and `typealias-forbidden`; tests also exercise identity and recovery behavior.[^review]

| Responsibility | Authoritative implementation reused |
|---|---|
| Tenant-scoped submission and deduplication | MCPs `submitDispatchJob`, its research routing functions and the existing dispatch table's immutable research reference. |
| Controller capacity and polling | `fleet.cli.research_queue` uses shared queue operations and resource leases; native scheduler waits retain their receipts. |
| Proposal configuration and attempt history | `research_records` uses the benchmark codec; `research_journal` uses `platform_core.event_journal`. |
| Native execution and collection | HPC3/WRDS adapters invoke the maintained `hpc3` commands. Resumed WRDS collection rechecks local bytes against freshly verified native evidence. |
| Staging identity and numerical evidence | Shared `StageManifest` and `RunRecord` codecs; no research-specific replacement records. |

The review found one concrete provenance defect: the bundle writer hashed UTF-8
text but wrote it in text mode, allowing Windows newline translation to change
the emitted bytes. `prepare_bundle` now writes, hashes and sizes the same byte
payload. The existing execution test suite reads those files back and verifies
their exact hashes, lengths and checksum content through the shared manifest
codec. The repair adds no helper or second writer. Its full fleet check passed 4,525 tests
with 100% coverage. Windows CI also passed 4,525 tests with 100% statement and
branch coverage; the independent staging-repair audit closed VERIFIED.[^bundle]

A fresh read of live queue job `fb62c111-f5c8-48f6-a790-899e68f76920` retained
native job `58586013` and RunRecord digest
`49b7a44e64b76b7d153a6a3fb1a3f1805ea328720fbc855ca6e0bc892d75af17`.
The installed controller still reports four completed trials and one refusal;
its canonical full-report digest remains
`727b765412d7ce3b19edfc71b45311135f06d72c1486e307a641f094f5bd9ab6`.
This read consumed no trial budget and submitted no allocation.[^live-review]

The earlier controller-path repair passed its Windows check and independent
audit. The native replay case also passed, but its parent board task
`8b19319c` remains waiting on the unrelated minutes-corpus root-cause task
`671c2919`; that parent audit is not complete. Label-overlap exclusion and
paired moving-block primitives are integrated into the monthly holdings study
described in [[crsp-monthly-holdings-validation]], and the final-period boundary
is documented in [[wrds-proposer-isolation]] and [[wrds-final-scoring]].[^live-review][^sampling][^splits][^holdings]

[^review]: `CODE_STANDARDS.md`, named guard-rule table; API functions cited in [^queue], [^controllers], [^admission] and [^records]. External MCPs source snapshot 06512f6450c960e52409448f306c1254ed633fb6: `packages/db/src/fleet-dispatch/research.ts` functions `requireResearchRouting` and `existingResearchJob`, and `packages/db/migrations/877_fleet_dispatch_research_requests.sql` immutable reference constraints. Scope and findings: board task 79c9a370-763e-4fd0-a609-cd1a64080b19.

[^bundle]: `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py` function `prepare_bundle`; `libs/platform_core/src/platform_core/stage_manifest.py` codecs; `libs/covenant_ml/tests/markets/test_research_execution.py` case `test_bundle_manifest_identifies_exact_written_bytes`. API source commit 6f98b519968ea5435119aec0d21105f6e3cb2997 and regression commit 1eb02136baa6b86938b73dbad3395e0a41d9ce5c; fleet job 748608d6-faf0-4cb2-ac45-50fe75e9a977 passed at the latter tip. Windows GitHub check `check`, run 40820758612, passed 4,525 tests in 209.66 seconds with 100% coverage. Board repair db0ddd19-bd02-4941-ac09-f4a88a8361f0 closed VERIFIED through audit 0fa4a1c5-7a6f-4975-8682-28c33448e5e2 at 2026-10-29T05:05:50Z.

[^live-review]: Live `dispatch_get` and installed `research_proposals.py status` reads on 2026-10-29, preserved in board task 79c9a370 thread post 1a29cf1d-3ed9-4d47-a6a6-8a23c7c08bd3. `task_get 3261e090` records audit ed3e8e63 VERIFIED; `task_get 8b19319c` records its remaining wait on 671c2919. Earlier served replay case: MCPs `execution/tests/served/research-queue-replay.test.ts`, two submissions and three controller reads in served run c469a100-f103-4e26-a246-396d81b5ee04; the entire served suite was not green.

[^queue]: `tools/fleet/src/fleet/cli/research_queue.py` functions `tick` and `reconcile`; `tools/fleet/src/fleet/cli/tick.py` function `plan_tick`. Board task 3dff6fbf-e09a-430f-b13e-5dccb505ba80 tracks rollout separately from implementation.

[^sampling]: `libs/covenant_ml/src/covenant_ml/evaluation/resampling.py`, `build_resampling`, `bootstrap_weights` and `moving_block_weights`; `libs/covenant_ml/src/covenant_ml/evaluation/paired.py`, `paired_mean_difference` and `compare_moving_block`. The monthly study's use of them is cited in the holdings footnote.

[^splits]: `libs/covenant_ml/src/covenant_ml/evaluation/splits.py` class `ChronologicalGroupedSplits`, methods `__call__` and `validation_split`.

[^controllers]: `tools/fleet/src/fleet/core/research_control.py` functions `read_specification` and `verified_input`; `tools/fleet/src/fleet/core/research_hpc.py` and `tools/fleet/src/fleet/core/research_wrds.py`, function `poll` in each.

[^records]: `libs/platform_core/src/platform_core/run_record.py:309` `agree_across_runs`.

[^admission]: `libs/covenant_ml/src/covenant_ml/markets/research_records.py` frozen contracts; `libs/covenant_ml/src/covenant_ml/markets/research_journal.py` admission and lifecycle reducers; `libs/platform_core/src/platform_core/event_journal.py` transactional persistence; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Board task 399048d8-b1d6-4e12-a4d3-a0ce2281c810.

[^holdings]: `libs/covenant_ml/src/covenant_ml/markets/crsp_holdings.py`, `evaluate_crsp_holdings` and `crsp_window_path`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `run_crsp_validation` and `crsp_study_rows`; `libs/covenant_ml/src/covenant_ml/markets/research_trial_records.py`, `decode_research_evaluation`; `libs/covenant_ml/tests/markets/test_crsp_holdings.py`, `test_crsp_study.py` and `test_research_trial.py`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Monthly holdings and validation study". Initial integration evidence: fleet job 48a2d9c8-b9f6-4c6b-8a65-917bd3828f37 passed make check for libs/covenant_ml at API fd3f7723bd6eb7f5fbc5caa3157efed306bf3949: 4743 tests, 100% coverage. Board task 8f7247a1-aa1a-41d7-be0e-f2b8e97c1d2d. The substantive monthly CRSP study has not run.
