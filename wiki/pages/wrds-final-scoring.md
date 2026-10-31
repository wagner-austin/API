---
title: Evaluator-owned WRDS final scoring
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/private-delivery.json"
  - "libs/covenant_ml/src/covenant_ml/markets/research_delivery.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_native_final.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_execution.py"
  - "libs/covenant_ml/tests/markets/test_research_native_final.py"
  - "libs/covenant_ml/tests/markets/test_research_delivery.py"
  - "tools/hpc3/src/hpc3/core/private_delivery.py"
  - "tools/hpc3/src/hpc3/core/_private_receive.py"
  - "tools/hpc3/tests/test_private_delivery.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_release.py"
  - "libs/covenant_ml/tests/markets/test_research_release.py"
  - "libs/covenant_ml/tests/markets/test_research_trial_cli.py"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/research_journal.py"
  - "libs/covenant_ml/tests/markets/test_research_selection.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_final.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py"
  - "libs/covenant_ml/scripts/research_proposals.py"
  - "libs/covenant_ml/scripts/research_trial.py"
  - "libs/covenant_ml/tests/markets/test_research_final.py"
  - "libs/covenant_ml/tests/markets/test_crsp_final.py"
source_git_blobs:
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/private-delivery.json": "0a92649b4cc3a607199c258c0521bee3cdd0c4f6"
  "libs/covenant_ml/src/covenant_ml/markets/research_delivery.py": "fed59a35e1266cdb2e4236539056a5758d84f358"
  "libs/covenant_ml/src/covenant_ml/markets/research_native_final.py": "ffe230f47ba39421a668a7d6b6299b50cb075e9a"
  "libs/covenant_ml/src/covenant_ml/markets/research_cluster.py": "db45435892a37a949f96b86048ae22546a48c15c"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "0e60db673db8e13dc84907d9cbdd2ce460b58235"
  "libs/covenant_ml/tests/markets/test_research_native_final.py": "c2e7bfd5c740d172d2e6232a6d799581ea06138d"
  "libs/covenant_ml/tests/markets/test_research_delivery.py": "d38d19577a78209d5204a6f6dd2bd020f938f639"
  "tools/hpc3/src/hpc3/core/private_delivery.py": "cf24939bab931ab1db07220f0b59203c84773285"
  "tools/hpc3/src/hpc3/core/_private_receive.py": "4816c514cce4bab08f4dd3647443dfff5e72bf48"
  "tools/hpc3/tests/test_private_delivery.py": "46ddf1497889a55c90e2719540ae75e4c1a7da38"
  "libs/covenant_ml/src/covenant_ml/markets/research_release.py": "03ce1fd2f4c5a63a2fb0c4379f9ab305a688724c"
  "libs/covenant_ml/tests/markets/test_research_release.py": "c0ceda2139bf2a8286bdcccfa5dd93df5a90eb13"
  "libs/covenant_ml/tests/markets/test_research_trial_cli.py": "00143abf439078afa60a0dc8c8cb72fa02f2538e"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "9686350a06789e4f05d869da140948b6a56d0d19"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "a36df5c2e20b454eabd26f3acfc75f07f1c3485e"
  "libs/covenant_ml/tests/markets/test_research_selection.py": "b2939285c746d390261861ed80edae3455000a84"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "92d38ea8211b08cf4480fa72a91e51761d6424bf"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_final.py": "ffcbb67a5d68f2040f003deb67d3b1831f445c3f"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py": "fa810eb315b13925c5664c35b883b80be0afee2d"
  "libs/covenant_ml/scripts/research_proposals.py": "f23c7f9e828d129ffc1a1d55188094dc998aea50"
  "libs/covenant_ml/scripts/research_trial.py": "11fbcce7a4f7721db03c3f09677e9d06cce6f895"
  "libs/covenant_ml/tests/markets/test_research_final.py": "426b4a91c5644f4b3db51cac54727545b870ccf1"
  "libs/covenant_ml/tests/markets/test_crsp_final.py": "67af9e640314ef33871e9a182d6511fb75dc0abc"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
fact_checked: 2026-10-30
confidence: medium
---

# Evaluator-owned WRDS final scoring

- Research entry point: [[wrds-equity-research-protocol]].

The existing `research_journal` accepts evaluator-owned `select` and `release_final`
transitions. Selection requires settled validation work, binds a completed
candidate and its validation payload to an exact final manifest, and seals
further admissions. Release is consumed once against those identities while
preserving the validation record and its `complete` execution status for queue
reconciliation; `selection` records the separate final-release state.
These are state invariants, not caller authentication: the journal must remain
evaluator-owned.[^selection]

`research_final` issues them. `research_proposals select` binds the named
completed attempt's journal candidate and payload to the final manifest's
SHA-256. The controller's `research_proposals release` checks admission and
protocol, commits `release_final`, and then acquires the final key and writes
a private grant bound to the canonical request and encrypted manifest digests.
A failure during key access or grant creation leaves the release consumed.
The candidate, seed and features come from the journal's own admitted
request; a proposal naming another partition is refused at admission and can
never be selected, so no request field authorizes final scoring.
`crsp_final.run_crsp_final` fits once on every decision whose exit precedes
`holdout_start` and scores every decision from `holdout_start` through
`holdout_end` with the validation study's shared fold, arm and record code. Changing every outcome
after the last final decision changes no fit or selection in the tests.
Choosing which attempt to select remains an evaluator decision.[^final]

The worker uses `research_trial --final-grant G --final-manifest M --scratch S`
with its request and output paths. It requires no journal or long-lived key
directory. After verifying the grant bindings, it exclusively creates and fsyncs
the consumption marker and its directory, then opens and removes the private
key. It checks actual code/image lineage before decrypting. Concurrent consumers
and replays are refused; runtime drift or a later data failure leaves permission
consumed. Verified plaintext exists within the private decryption context, and
the shared result publisher retains ciphertext and plaintext identities.[^encrypted]

The grant is an evaluator-private capability, not a second admission journal.
Native delivery must preserve one grant location and its spent marker; copying
it into independently consumable locations defeats single use. An evaluator
administrator remains privileged. Real subprocess tests score synthetic data
after removing the controller journal and long-lived key, reject replay, and
allow exactly one of two competing consumers. This tests the worker boundary;
deployment and native rehearsal evidence remain in [[wrds-baseline-readiness]].[^grant]

The controller's `deliver --trial T --final-manifest M --cluster C --directory G
--destination D` consumes local grant `G` before transport. It sends the release
metadata and identity over SSH stdin using the existing stage-manifest codec.
The receiver requires a private mode-0700 parent and creates destination `D`
exclusively; each file is mode 0600, flushed and read back against its digest.
No key appears in command arguments or the delivery transcript. An ambiguous
transfer leaves the local grant spent, and replay cannot overwrite the receiver.
This assumes a trusted evaluator account and SSH endpoint, not protection from
administrators who can copy capabilities.[^delivery]

The same receiver accepted a synthetic payload on both HPC3 and WRDS over
authenticated SSH, with independent digest and mode readback; replay was refused
without changing the received file. This verifies private transfer on those
hosts, not encrypted execution through their native schedulers. WRDS used the
shared core API; the private-stage CLI currently reads an HPC3 workspace.[^transfer-probe]

[^transfer-probe]: `tools/hpc3/artifacts/wrds-final-runtime-20261030/private-delivery.json`, `measurements`, payload identity, transport path and limitations; independent readbacks retained in its private evidence directory.

`submit-final --trial T --cluster C --directory G --remote-manifest M` requires
the original journal's released selection, consumed local grant and verified
delivery receipt. It reserves `G/native` once, then reuses the validation
recipe, stage and submit machinery with a distinct `:final` attempt identity.
`reconcile-final` uses the same native observation and measured-dataset checks,
requiring the final partition, frozen manifest, final record label, protocol
and image. It returns queue metadata while preserving the journal's original
validation result. Submission failure retains the exclusive directory and never
permits resubmission. [[wrds-research-queue-admission]] connects these commands
to the existing queue.[^native]

[^delivery]: `libs/covenant_ml/src/covenant_ml/markets/research_delivery.py`, `deliver_final_grant`; `tools/hpc3/src/hpc3/core/private_delivery.py`, `deliver_private_bundle`; `tools/hpc3/src/hpc3/core/_private_receive.py`, receiver program; `libs/covenant_ml/tests/markets/test_research_delivery.py`, spent-grant and real transport failure cases; `tools/hpc3/tests/test_private_delivery.py`, real receiver, replay and process failures.

[^native]: `libs/covenant_ml/src/covenant_ml/markets/research_native_final.py`, `submit_final_attempt` and `reconcile_final_attempt`; `libs/covenant_ml/src/covenant_ml/markets/research_cluster.py`, `prepare_bundle`; `libs/covenant_ml/src/covenant_ml/markets/research_execution.py`, `stage_and_submit`, `observe_attempt` and `require_measured_dataset`; `libs/covenant_ml/tests/markets/test_research_native_final.py`, original-journal preservation, lineage refusals and real failed launch.

[^encrypted]: `libs/covenant_ml/src/covenant_ml/markets/research_final.py`, `release_final` and `execute_final`; `libs/covenant_ml/scripts/research_trial.py`, `main`; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `publish_monthly_record`; `libs/covenant_ml/tests/markets/test_research_final.py`, `test_final_key_is_acquired_only_after_irrevocable_release`, `test_failed_grant_creation_cannot_restore_journal_permission` and `test_native_runtime_drift_consumes_the_grant_without_scoring`; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission".

[^grant]: `libs/covenant_ml/src/covenant_ml/markets/research_release.py`, `write_final_grant` and `consume_final_grant`; `libs/covenant_ml/tests/markets/test_research_release.py`, `test_two_real_processes_cannot_redeem_the_same_grant`; `libs/covenant_ml/tests/markets/test_research_trial_cli.py`, `test_real_final_worker_needs_neither_journal_nor_long_lived_keys`. These are Linux process tests using synthetic data, not native scheduler runs.

[^selection]: `libs/covenant_ml/src/covenant_ml/markets/research_journal.py`, `_transition` and `_final_identity`; `libs/covenant_ml/tests/markets/test_research_selection.py`, real SQLite selection, release and atomic refusal cases; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission".

[^final]: `libs/covenant_ml/src/covenant_ml/markets/research_final.py`, `select_candidate` and `execute_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_final.py`, `run_crsp_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_fold`, `crsp_arms` and `crsp_known_panel`; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `require_trial_lineage` and `publish_monthly_record`; `libs/covenant_ml/scripts/research_proposals.py` and `libs/covenant_ml/scripts/research_trial.py`; `libs/covenant_ml/tests/markets/test_research_final.py` and `test_crsp_final.py`, real journals, extracts and trainers; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Fleet job 379d582f-130b-47c4-8579-56ec1be8d586 passed make check for libs/covenant_ml at API 06636dbbebd56771e170b45a7f190309f396cde1 on Colossus at 2026-10-30T05:45:28Z: 4,786 tests, 100.00% coverage. The earlier job 2e9095f9 at bddfcab33 ran every test green but failed the test-quality guard on one substring assertion, repaired in 06636dbbe. On main at 85e408c1c08cd3ff5faaf88416f4d62965ed64a3, job 8054c71b passed libs/covenant_ml on Colossus at 2026-10-30T05:57:25Z: 4,786 tests, 100.00% coverage. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.
