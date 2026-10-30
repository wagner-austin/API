---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "tools/fleet/tests/_host.py"
  - "tools/fleet/tests/test_research_proposer_host.py"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json"
  - "libs/covenant_ml/scripts/research_storage_probe.py"
  - "libs/covenant_ml/tests/markets/test_research_storage_probe.py"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json"
  - "tools/hpc3/artifacts/wrds-final-runtime-20261030/provenance.json"
  - "libs/covenant_ml/src/covenant_ml/markets/research_release.py"
  - "libs/covenant_ml/tests/markets/test_research_release.py"
  - "libs/covenant_ml/tests/markets/test_research_trial_cli.py"
  - "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/cleanup.json"
  - "libs/covenant_ml/tests/markets/test_encrypted_process.py"
  - "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/rehearsal.json"
  - "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/provenance.json"
  - ".github/workflows/covenant-ml.yml"
  - "libs/covenant_ml/src/covenant_ml/markets/research_keys.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_execution.py"
  - "libs/covenant_ml/src/covenant_ml/evaluation/evidence.py"
  - "libs/covenant_ml/tests/markets/test_research_keys.py"
  - "libs/covenant_ml/tests/markets/test_research_execution.py"
  - "libs/covenant_ml/src/covenant_ml/markets/encrypted_records.py"
  - "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py"
  - "libs/covenant_ml/tests/markets/test_encrypted_storage.py"
  - "libs/covenant_ml/tests/markets/test_encrypted_records.py"
  - "tools/hpc3/artifacts/wrds-monthly-runtime-20261030/provenance.json"
  - "tools/hpc3/artifacts/wrds-controller-verification-20261028/provenance.json"
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/research_trial.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py"
  - "libs/covenant_ml/src/covenant_ml/markets/research_final.py"
  - "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py"
  - "libs/covenant_ml/scripts/research_proposals.py"
  - "libs/covenant_ml/scripts/research_trial.py"
  - "libs/covenant_ml/tests/markets/test_research_final.py"
  - "libs/covenant_ml/tests/markets/test_crsp_final.py"
  - "tools/fleet/src/fleet/core/research_proposal.py"
  - "tools/fleet/src/fleet/cli/research_propose.py"
  - "tools/fleet/tests/test_research_proposal.py"
  - "tools/fleet/tests/test_research_proposal_host.py"
source_git_blobs:
  "tools/fleet/tests/_host.py": "3f7c3145ff37f9e74f0ad77d9195265e23beacd7"
  "tools/fleet/tests/test_research_proposer_host.py": "894256824c924e03236a5c64160a753f96015579"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json": "5d9afac1cd73df5108a17c49127f811dd8ae8a05"
  "libs/covenant_ml/scripts/research_storage_probe.py": "8dc13973868bdefb3e3cf26c452871ce6d072235"
  "libs/covenant_ml/tests/markets/test_research_storage_probe.py": "ce083a45f3100c51e049c85eae865fde345c8034"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json": "2ef67b9d8ea85e09cf92c99d7ec56b96b07cb609"
  "tools/hpc3/artifacts/wrds-final-runtime-20261030/provenance.json": "1d1fd48609fe278db5adbd9ded31746c01c94df4"
  "libs/covenant_ml/src/covenant_ml/markets/research_release.py": "03ce1fd2f4c5a63a2fb0c4379f9ab305a688724c"
  "libs/covenant_ml/tests/markets/test_research_release.py": "c0ceda2139bf2a8286bdcccfa5dd93df5a90eb13"
  "libs/covenant_ml/tests/markets/test_research_trial_cli.py": "00143abf439078afa60a0dc8c8cb72fa02f2538e"
  "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/cleanup.json": "976cfc40fbaca4955084948061d6a2fcd90d54ab"
  "libs/covenant_ml/tests/markets/test_encrypted_process.py": "63a3a2f34e7188ee307ce1ea1b178f44f1995854"
  "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/rehearsal.json": "6f58fc955b6a9086564ea6a36f3028fcd2ac0ed6"
  "tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/provenance.json": "11769f95c43f98fe1e6df6dd071eb308a0c90a8b"
  ".github/workflows/covenant-ml.yml": "3dd4022a0f4e46cddaccec3be16188fbdc275dae"
  "libs/covenant_ml/src/covenant_ml/markets/research_keys.py": "8cb56c464b77f760375b512fd17159c3a2d9a9f3"
  "libs/covenant_ml/src/covenant_ml/markets/research_execution.py": "0e60db673db8e13dc84907d9cbdd2ce460b58235"
  "libs/covenant_ml/src/covenant_ml/evaluation/evidence.py": "9bc1459b035fcedeab41f91b84960faebfbe73a2"
  "libs/covenant_ml/tests/markets/test_research_keys.py": "ceb7f3b58753e99d7a559b8e471c46d1abb28067"
  "libs/covenant_ml/tests/markets/test_research_execution.py": "397dc3169486f1625090aa488dcf686fb13d7db4"
  "libs/covenant_ml/src/covenant_ml/markets/encrypted_records.py": "6b3826b16f265301ec9923070cc89322427f8bc6"
  "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py": "d53e02a8e707198a6d5c35f214d0fd557ee995e3"
  "libs/covenant_ml/tests/markets/test_encrypted_storage.py": "de361bb8f31d40eecacea21a0e493d3eba2abe30"
  "libs/covenant_ml/tests/markets/test_encrypted_records.py": "2cfcf565e6b24f7b252059891a4cf9e545329923"
  "tools/hpc3/artifacts/wrds-monthly-runtime-20261030/provenance.json": "48a1ceff729c1af46bbf6a0748614eec152e2659"
  "tools/hpc3/artifacts/wrds-controller-verification-20261028/provenance.json": "b678f473462c24db2cbd1e510ffe6c034e36a9bb"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "284621906d62f97da2080ca823bdc971aa9a0454"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "92d38ea8211b08cf4480fa72a91e51761d6424bf"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_final.py": "ffcbb67a5d68f2040f003deb67d3b1831f445c3f"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py": "fa810eb315b13925c5664c35b883b80be0afee2d"
  "libs/covenant_ml/scripts/research_proposals.py": "f23c7f9e828d129ffc1a1d55188094dc998aea50"
  "libs/covenant_ml/scripts/research_trial.py": "11fbcce7a4f7721db03c3f09677e9d06cce6f895"
  "libs/covenant_ml/tests/markets/test_research_final.py": "426b4a91c5644f4b3db51cac54727545b870ccf1"
  "libs/covenant_ml/tests/markets/test_crsp_final.py": "67af9e640314ef33871e9a182d6511fb75dc0abc"
  "tools/fleet/src/fleet/core/research_proposal.py": "dd0d9f77373e490d69aef66d305768a959b6f525"
  "tools/fleet/src/fleet/cli/research_propose.py": "1c5e8bf6bab8bd79564235a292cc875a54a71fe2"
  "tools/fleet/tests/test_research_proposal.py": "e3032d998c3d7e4b0951b2db7d1be643978ac2c5"
  "tools/fleet/tests/test_research_proposal_host.py": "177c817e1992334468ed07142a01df4f71908aa6"
provenance:
  - "2026-10-30 independent audit 02c6e4a1-669d-43db-99c8-bedfc0f9754a reopened encryption task e6729a16: the submitted A2 mapping omitted the real container execution lane, and A3 read only queue status. Read-only native-report query against demure mcp-postgres at 21:29 UTC returned 1 for the service-authored report joined to queue 120da295 by RunRecord digest and ciphertext input identities; changed bundle digest and deliberately excluding a present sha256 field each returned 0. These query checks bind the recorded synthetic run and detect an age-key marker, not arbitrary secret formats or licensed-row leakage."
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "2026-10-30 readiness recheck, board cda5d102-9f26-4c2a-bfbe-cfdbe17b33c6: live task_get reports 1e555057, 7c71808d and 364a4ba8 done after independent audit; dispatch_list at API e12eb1c8312b9d4ae57e38906ca787e02e0e094c reports research check b37025f3 (4786 passed, 100%), fleet check a4f8cdc8 (2495 passed, 100%) and real host check 16b62830 (79 passed). Local-file inspection of /home/corvis/research-control/wrds-validation-bda8ca2af/lib/python3.11/site-packages/covenant_ml/markets/research_trial.py still finds load_wrds_evaluation at lines 18 and 58, SHA-256 98e11f10c1a28dcaeaeec9fde9e21c22f93be0c8c8cd0add5c50e23e4805d271. Its entrypoint/scripts/research_proposals.py exposes init/status/propose/submit/reconcile and no select, SHA-256 ed2272d101c645375ada9cc46caa4f2c288f9cefd196c634827a81ca7d0785ef. These are local installed-file readings, not a new cluster execution."
fact_checked: 2026-10-30
confidence: medium
---

# WRDS substantive baseline readiness


- Research entry point: [[wrds-equity-research-protocol]].

The encrypted-bundle implementation passed independent audit `acd2c6c4` on
October 30 at 21:38 UTC: all three criteria were verified. The earlier audit
`02c6e4a1` reopened its evidence mapping because it omitted the real
container-isolation lane and read queue status without retained input
identities. The accepted mapping uses `tools/fleet-execution-linux` for
proposer isolation, evaluator tests for dataset binding, and the service-generated
native report joined to the queue's RunRecord digest. This accepts the encrypted
synthetic workflow; the substantive baseline remains unrun.[^audit-mapping]

[^audit-mapping]: Live board audit `acd2c6c4-6534-4144-ad5a-f7e68a216436`, computed VERIFIED result at 2026-10-30T21:38:54Z, supersedes rejected audit `02c6e4a1-669d-43db-99c8-bedfc0f9754a` and check review `9312e8df-7856-4b3a-b77b-d15f31d43cb7`; service-written native report `548e2973-d926-44bb-98fc-016ab7f375b7` for WRDS job `40550033` and queue `120da295-8f12-4b90-ba0e-a2af14d5d3b6`. Fresh fleet execution job `ecf1339b` passed 95 actual host cases with zero skips at API `a813b1ebe`. `tools/fleet/tests/_host.py`, execution-only marker selection; `tools/fleet/tests/test_research_proposer_host.py`, `test_actual_proposer_denies_private_data_credentials_policy_and_network` and real timeout/cleanup cases. `libs/covenant_ml/tests/markets/test_research_execution.py`, `test_changed_dataset_cannot_complete`, refuses mismatched bundle digests, verified inputs and partitions. The ordinary fleet suite skips the rootless Docker cases; only its execution lane runs them.

The implementation and independent audits are complete for the process primitive,
its admission integration and evaluator-owned one-shot final controls. At API
`e12eb1c8312b9d4ae57e38906ca787e02e0e094c`, live fleet records show 4,786 research
tests and 2,495 fleet tests passing with 100% coverage, plus 79 actual host cases.
That establishes the tested code revision. It does not establish that the pilot's
frozen controller or compute image contains that revision.[^readiness]

The October 30 runtime upgrade freezes API
`92719c0601c27d31fcd4338e9fd3e8fc3b85c656`. The separate installed controller is
`/home/corvis/research-control/wrds-monthly-92719c060/bin/python`; its proposal CLI
exposes `select`, and `pip check` passes. Native image build `58629088` completed
with exit `0:0` in 83 seconds on `hpc3-15-20`, using eight CPUs on `free`.
The generated symbol/version selfcheck passed both during construction and as
the unprivileged user, followed by a passing dependency check. The image at
`/pub/wagnera3/images/research-monthly-92719c060/research-monthly.sif` has SHA-256
`5072a7a24b6dd2ae22e05c68d3691be400c49a8bc4a2be8ca27fc997b9e0a5fc`.
The old pilot remains unchanged. These measurements establish the new runtime,
not the complete queue rehearsal or a substantive baseline.[^monthly-runtime]

The operator additionally requires encrypted, checksum-verified data bundles,
evaluator-only decryption into private job storage, keys absent from images,
logs and proposer-visible files, and a separately sealed final bundle. This is
a prerequisite for licensed staging, tracked in board task `e6729a16`.[^encryption]
The encrypted evaluator is installed separately at
`/home/corvis/research-control/wrds-encrypted-44fa91cea/bin/python`.
Its matching image freezes API `44fa91cea1607660a45f29172d8564a465d40744` at
`/pub/wagnera3/images/research-encrypted-44fa91cea/research-encrypted.sif`, SHA-256
`58b87a0ef12df5a1e061b9924353884aa31b5d70b5117fa7c911466d1426eba6`.
Native build `58629212` completed with exit `0:0` in 78 seconds on
`hpc3-22-05`; both generated selfchecks and the dependency check passed.
This establishes runtime construction; the subsequent encrypted validation
rehearsal also completed through the shared queue.[^encrypted-runtime][^rehearsal]

The shared library now implements age-encrypted files, separately named
train-validation/final bundles, and both ciphertext and plaintext SHA-256
identities using the existing staging records. Extraction needs only the public
recipient. Decryption accepts an in-memory identity and yields the extraction
manifest only after every file authenticates and matches its frozen identity.
Plaintext lives in a new mode-0700 temporary directory, with a declared size
limit, and is removed on ordinary exit or exception. HPC3 validation and final
scoring, and WRDS-native validation storage, are measured below. WRDS final
scoring and forced-kill scratch reclamation remain unverified.[^bundle][^final-rehearsal][^wrds-native]

The evaluator now requires encrypted inputs and separate private partition keys.
Validation works without a final key. Final scoring acquires its key only after
the existing journal commits its one-shot release; a missing key consumes that
release rather than permitting a retry. Both paths reuse the CRSP reader and
shared result publisher. Results retain ciphertext and plaintext identities,
and reconciliation refuses a dataset digest or partition inconsistent with the
admitted request and native checksum evidence.[^integrated]

Key directories and decrypted storage must actually have mode 0700; keys must
be private regular files opened without following symlinks. These are Linux
execution requirements, not Windows ACL guarantees. Same-user or administrative
processes remain privileged. The new controller `release` command consumes the
authoritative journal before creating a private request-bound grant. The worker
consumes that grant durably and removes its delivered key before decryption,
without a journal copy. Real subprocess tests cover successful synthetic scoring
without the journal or long-lived key, replay refusal and competing consumers.
Native grant transport and final-job submission/reconciliation now reuse the
existing controllers and shared queue, as described in [[wrds-final-scoring]].
The controller at `/home/corvis/research-control/wrds-final-85f1d4d75/bin/python`
and image freeze runtime source `85f1d4d7521d4a189836f8ecca805f39c5c366df`.
All 859 installed wheel code files match their frozen wheels. Image build
`58635077` completed on `hpc3-19-14` in 113 seconds, exit `0:0`, and passed
both selfchecks and dependency verification. The image is explicitly stored at
`/data/homezvol1/wagnera3/research-images/final-85f1d4d75/research-final.sif`,
SHA-256 `eeedc89bbde5fde0e49c497da124526a51d31a1d7cb6fd605924697ac0813710`;
this home-backed build followed observed `/pub` metadata stalls on two login
nodes. At `5ae042ef3`, a test-only correction over that runtime source, the full
gates passed 4,874 evaluator and 2,528 fleet tests with 100% coverage, typing
and guards clean. The subsequent synthetic final-scoring rehearsal completed
through the shared queue; the encrypted-workflow adequacy audit subsequently
passed as recorded above.
The image build and execution receipts establish different parts of that
claim.[^final-runtime][^final-rehearsal][^grant]

Fresh validation job `58635333` and final job `58635492` each completed in nine
seconds with exit `0:0` on `hpc3-l18-05`. Shared queue entries `9ee4c5a8` and
`50a1fb4e` closed passed after fresh native inspection verified 27 inputs per
allocation, the correct partition and canonical RunRecord. The original
validation request and record stayed byte-identical through final release and
execution. The delivered final key was absent afterward, with its durable
consumed marker present; delivery and submission replays both failed before
another allocation. Two prelaunch failures remain recorded, including the
refused older entrypoint pin. This is a synthetic operational measurement,
not a substantive baseline or a scientific comparison.[^final-rehearsal]

WRDS Cloud also passed encrypted validation-storage jobs `40550023` and
`40550033` on `wrds-sas46-w` and `wrds-sas34-w`. The existing shared queue closed
both passed after native accounting, input/runtime verification and verified
collection. Both authenticated ten files into mode-0700 plaintext directories
under the scheduler's mode-0755 scratch, measured normal-exit removal, and
reproduced the HPC3 validation plaintext-manifest digest. Their observation
files were byte-identical; host-specific receipts and RunRecords differ.
Resubmitting the completed queue reference returned the same job. Failed job
`40550007` remains recorded: the initial probe unnecessarily required the parent
scratch directory to be 0700 and stopped before decryption.[^wrds-native]

This WRDS deployment uses a private Python 3.11.16 runtime, with 488 application
files compared against six frozen wheels at installation. The native worker
checks interpreter bytes/version and 61 distribution versions before and after
each allocation; it does not rehash all installed package files. The probe
reuses `EvaluatorKeys`, `decrypted_bundle` and shared observations. It establishes
normal validation-storage execution, not WRDS final-grant scoring, forced-kill
scratch reclamation, or execution of the HPC3 Apptainer image. The published
probe passed 4,877 evaluator tests at 100% coverage with typing and guards clean.
No licensed rows were involved.[^wrds-native]

The forced-kill probe decrypted and verified ten synthetic files in a mode-0700
directory on `hpc3-22-05`, then intentionally ended native job `58629286` with
`FAILED`, exit `137:0`. A later allocation, `58630827`, required that same node
and verified the earlier scratch path was absent, completing with `0:0`.
The observation occurred several hours later: it establishes eventual absence
for this one probe, not immediate reclamation, physical erasure or cleanup after
node failure. Real-process regression cases separately confirm normal exit
removes plaintext and SIGKILL leaves it for the scheduler.[^cleanup]

[^cleanup]: `tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/cleanup.json`, native accounting, receipts, recipe hashes and limits; `libs/covenant_ml/tests/markets/test_encrypted_process.py`, `PROBE_PROGRAM` and `test_real_process_exit_and_sigkill_have_distinct_cleanup`. Fleet job 42b48bdc-c31d-4436-b670-6efe4860c8f1 passed all 4,838 evaluator tests with 100% coverage at API 8dfc76f75003cdefdaa124e2a3b1e5f95670ce5a on Lavender, ending 2026-10-30T12:48:59Z. GitHub runs 41058285948 and 41058285858 also succeeded. The native runtime remains 44fa91cea; the probe program is extracted from the tested constant at 8dfc76f.

The real isolated proposer produced an allowed synthetic proposal, which was
transported manually to the installed admission controller and admitted into a
fresh journal. Shared queue `a332dd13-11b8-470f-9871-211490a67a8b` was then
claimed by the scheduled `fleet-research-diphtheria` runner. It submitted native
job `58629261`, verified its result, and closed the queue entry automatically at
2026-10-30T12:30:36Z. Native accounting reports `COMPLETED`, exit `0:0`, 54 seconds,
one CPU on `free`, node `hpc3-22-05`. Sixteen input checksums were verified,
including the image, ciphertexts, bundle metadata, access configuration and
entrypoint. The result retains the admitted ciphertext-bundle identity and
plaintext-manifest identity. This is a synthetic validation measurement, not a
substantive study or a native final-scoring rehearsal.[^rehearsal]

The live rootless proposer probe on Colossus was denied reads of a real canary
age identity and synthetic final CSV, including through `/proc/1/root`, access
to its host Docker socket, and network connectivity. It ran as uid 65534 with only
`LC_CTYPE` and `PATH`; the host files existed, were readable by the evaluator,
and remained byte-identical. This verifies that sandbox boundary against host
canaries, not protection against the evaluator's administrators.[^key-probe]

[^key-probe]: Live Colossus run on 2026-10-30 using `PROBE_PROGRAM` from `tools/fleet/tests/test_research_proposal_host.py` through installed `fleet.core.research_proposer.run_proposer`, with pinned image sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337. Private evidence `/home/corvis/research-data/wrds/runtime/encrypted-44fa91cea/isolation-probe/result.json` has SHA-256 21eb46b323576db1d33f651b347a4c68c10624e43123e23c43f44e6ca80086e9; program SHA-256 127315f96d43024be4880c08266a7aec6d3d1cd563768c3bfd786c4266f5cd32. All four reads returned errno 2; network access returned errno 101. The separately generated canary identity was not the study's final key, and no secret bytes were emitted.

[^grant]: `libs/covenant_ml/src/covenant_ml/markets/research_final.py`, `release_final` and `execute_final`; `libs/covenant_ml/src/covenant_ml/markets/research_release.py`, `write_final_grant` and `consume_final_grant`; `libs/covenant_ml/tests/markets/test_research_release.py`, competing real consumers and failure cases; `libs/covenant_ml/tests/markets/test_research_trial_cli.py`, `test_real_final_worker_needs_neither_journal_nor_long_lived_keys`; `libs/covenant_ml/tests/markets/test_research_final.py`, `test_failed_grant_creation_cannot_restore_journal_permission`. Full fleet gate f428dec3-aeb3-4bc4-9b5e-e7f9f3e1acd9 at API 9783dcd84dbc6ba6a5033e05fb1ca6c1085d1f66 passed on Serendipity at 2026-10-30T17:33:05Z: 4,850 tests, 100% coverage, 191 seconds of 300. Earlier acb42784 at 66b17108f passed all tests but failed env-access-banned in the subprocess test; 9783dcd84 uses the existing shared environment reader and passes the complete gate. The authoritative journal and existing numerical evaluator are reused; no native grant transport is claimed by these process tests.

The remaining sequence follows these measured gaps:[^final-rehearsal][^wrds-native][^integrated][^cleanup][^grant]

1. Automate the measured cross-host proposer-to-admission handoff through the
   existing controller. Verify WRDS forced-kill scratch reclamation, which the
   normal-exit measurements do not establish; native final scoring is currently
   rehearsed on HPC3 only.
2. Materialize the frozen four-relation CRSP extraction with exact SQL, vintage,
   checksums and calendar coverage, then stage its verified manifest to HPC3.
   Keep final-period evidence confined to the evaluator.
3. Declare candidate/seed aggregation, the selection criterion and tie handling
   before inspecting study results. Run the bounded baseline and validation
   comparison first; seal the chosen candidate and use the one-shot final release
   afterward. The selection API enforces identity, not which candidate is best.

[^encrypted-runtime]: `tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/provenance.json`, source archive, controller and wheel identities, native image accounting, selfchecks and private evidence hashes. This artifact records the runtime build; subsequent queue evidence is in [^rehearsal]. Only the validation identity was delivered to HPC3; the final identity remains on the evaluator host. No licensed observations were staged.

[^final-runtime]: `tools/hpc3/artifacts/wrds-final-runtime-20261030/provenance.json`, frozen source archive, installed wheel readback, native build accounting, image digest, fleet gates and private evidence hashes. Source `85f1d4d75` and tested revision `5ae042ef3` differ only in one test assertion. This is a construction and installation measurement, not a final-scoring execution receipt.

[^wrds-native]: `tools/hpc3/artifacts/wrds-final-runtime-20261030/wrds-native.json`, `runtime`, `dataset`, `measurements`, `repeatability`, `preserved_failure`, `verification` and hashed private evidence; `libs/covenant_ml/scripts/research_storage_probe.py`, `main`; `libs/covenant_ml/tests/markets/test_research_storage_probe.py`, real encrypted files and subprocess success/refusal. Queue jobs `836d7d24-ac43-42eb-8e1d-f4723a061532` and `120da295-8f12-4b90-ba0e-a2af14d5d3b6` closed passed on 2026-10-30. Probe publication `bbd989810` passed fleet job `8ed8468c`; its source bytes match the earlier tested snapshot used by both allocations. This session explicitly invoked existing queue ticks for submission and reconciliation.

[^final-rehearsal]: `tools/hpc3/artifacts/wrds-final-runtime-20261030/rehearsal.json`, `validation`, `final`, `grant`, `recorded_failures` and hashed private evidence. Queue closure digests are `bd9fcef7219a2d9b4e6840324c41e327a24e8ca1e08b3aff18d0da9e86416139` and `43ba8b73ca203fb5bf6843cacd92f444337516a1fdaa7e96de88395dd806cd86`. The scheduled runner claimed the requests and submitted the final allocation; this session explicitly invoked existing ticks for validation submission and prompt reconciliation. The proposer ran in the existing Colossus sandbox, with its output transported by this session before admission; a fully automated cross-host proposal handoff is not established.

[^rehearsal]: `tools/hpc3/artifacts/wrds-encrypted-runtime-20261030/rehearsal.json`, queue closure, native accounting, verified_files, dataset identities, proposer isolation and private evidence hashes. Live dispatch_get on 2026-10-30 reported queue a332dd13-11b8-470f-9871-211490a67a8b passed with canonical RunRecord SHA-256 cfe6049eaf1431f68e73f8bc62c98dbf2da7a86f8a9ea04eb86b2c91b2d0087a. The native record file separately hashes to 8122346701a00789c9d17f626bca454481f7d39d2315e07d78f803d9c15a79da; these distinguish canonical record encoding from stored file bytes. Private observe-1.log SHA-256 3d49ad6aee7d1abafa6891063160ed73aa11474999cb2ebcdd166aac9cfbb550 retains the native evidence returned to the existing reconciliation controller. The source revision remains 44fa91cea; documentation publication at f17f7ba3 passed full fleet checks 2b00dfd6 (2,006 HPC3 tests) and e9a5322b (4,836 evaluator tests), both at 100% coverage.

[^final]: `libs/covenant_ml/src/covenant_ml/markets/research_final.py`, `select_candidate` and `execute_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_final.py`, `run_crsp_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_fold`, `crsp_arms` and `crsp_known_panel`; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `require_trial_lineage` and `publish_monthly_record`; `libs/covenant_ml/scripts/research_proposals.py` and `libs/covenant_ml/scripts/research_trial.py`; `libs/covenant_ml/tests/markets/test_research_final.py` and `test_crsp_final.py`, real journals, extracts and trainers; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Fleet job 379d582f-130b-47c4-8579-56ec1be8d586 passed make check for libs/covenant_ml at API 06636dbbebd56771e170b45a7f190309f396cde1 on Colossus at 2026-10-30T05:45:28Z: 4,786 tests, 100.00% coverage. The earlier job 2e9095f9 at bddfcab33 ran every test green but failed the test-quality guard on one substring assertion, repaired in 06636dbbe. On main at 85e408c1c08cd3ff5faaf88416f4d62965ed64a3, job 8054c71b passed libs/covenant_ml on Colossus at 2026-10-30T05:57:25Z: 4,786 tests, 100.00% coverage. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.

[^readiness]: `tools/hpc3/artifacts/wrds-controller-verification-20261028/provenance.json`, controller_source_commit, controller_python and attempts; the October 30 local-file and live-tool reads recorded in this page's provenance. Implementation and test sources are cited in [^installed] and [^final]. Remaining steps are operational prerequisites, not completed jobs or performance findings.

[^monthly-runtime]: `tools/hpc3/artifacts/wrds-monthly-runtime-20261030/provenance.json`, source_commit, source_archive_sha256, controller_python_sha256, controller_entrypoint_hashes, first_party_wheel_sha256, image and evidence_sha256. Its private evidence includes native accounting, the generated build transcript, captured environment and staging manifests. At that exact source revision, fleet jobs 2eed7059, 7d1ae3f0 and 20312c9e passed 4,786 evaluator, 2,006 HPC3 and 2,510 fleet tests respectively, each with 100% coverage. Board task 474c840b.

[^encryption]: Board task e6729a16-3d8b-4f0f-9d68-73ea5d56b95d, October 30 operator requirement and acceptance criteria. This paragraph specifies the required workflow; it reports no completed encryption or key-custody measurement.

[^bundle]: `libs/covenant_ml/src/covenant_ml/markets/encrypted_records.py`, `decode_encrypted_bundle`; `libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py`, `encrypt_bundle`, `encrypted_stage_manifest` and `decrypted_bundle`; `libs/covenant_ml/tests/markets/test_encrypted_storage.py` and `test_encrypted_records.py`. Fleet job 76a1c77b-cd31-40d1-ad3d-2ae4b447f219 passed the full libs/covenant_ml make check at API e8c4358f29a834d8c98cc098a082f7b99ce3a0db on Loki, ending 2026-10-30T11:45:25Z: 4,815 tests, 100% coverage, lint and strict typing passed. Its 29 new cases cover the existing CRSP reader, wrong keys, altered/truncated ciphertext, traversal, output bounds and cleanup after an evaluator exception. This is a library measurement, not native key delivery, final authorization, forced-kill reclamation or a completed queue rehearsal. Board task e6729a16.

[^installed]: `tools/fleet/src/fleet/core/research_proposal.py`, `propose`; `tools/fleet/src/fleet/cli/research_propose.py`, `main`; `tools/fleet/src/fleet/core/research_control.py`, `read_pinned` and `invoke`; `tools/fleet/tests/test_research_proposal.py`, admitted and refused recording, no evaluator path in any daemon argument, drifted-input and isolation refusals; `tools/fleet/tests/test_research_proposal_host.py`, `test_installed_proposer_cannot_reach_evaluator_assets`. Host lane job fc8bc3c5-ad1b-479e-b44c-ddfcd40152ad passed tools/fleet-execution-linux at API 06636dbbebd56771e170b45a7f190309f396cde1 on Sedona at 2026-10-30T05:46:34Z: 79 cases ran, 0 skipped, 0 failed. Full check b01963a6-2105-49a3-a81a-8551982ca23d passed tools/fleet at the same commit on Sedona at 2026-10-30T05:43:08Z: 2,479 tests, 100.00% coverage. Commit 06636dbbe was staged on a pushed-tests ref and rebased unchanged onto main as part of 85e408c1c08cd3ff5faaf88416f4d62965ed64a3; there, host lane job 6833f095-6fe8-48c5-a464-548cf228597b passed on Colossus at 2026-10-30T05:58:34Z (79 cases, 0 failed, 0 skipped) and full check f7581b88 passed tools/fleet on Loki at 2026-10-30T05:55:25Z (2,495 tests, 100.00% coverage). The admission controller in the host case is a recording stand-in for `research_proposals propose`; the real controller's recording of proposals and refusals is the covenant_ml suite's. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.

[^integrated]: `libs/covenant_ml/src/covenant_ml/markets/research_keys.py`, `EvaluatorKeys.acquire` and `create_evaluator_keys`; `libs/covenant_ml/src/covenant_ml/evaluation/evidence.py`, `require_private_directory`; `libs/covenant_ml/src/covenant_ml/markets/research_execution.py`, `reconcile_attempt`; the trial/final implementations and `test_research_keys.py`, `test_research_execution.py`, and `test_research_final.py` cited in source_paths. API 2e3a5cfebea38db752aa4bfb281104337cb6baee passed fleet job f8020224-a82c-42d7-9b3e-cf2fc084689a on Loki: 4,836 tests, 100% coverage. The permission check in 44fa91cea1607660a45f29172d8564a465d40744 passed the full suite again at API 90b03d7c8b6bc5c1ca1620d6fb4c4b13d9148994, job cd0d062a-c5b0-4a7d-acb5-f860b6a546de on Lacrymal, ending 2026-10-30T12:15:55Z: 4,836 tests, 100% coverage, lint, typing and guards passed. These are Linux suite measurements, not a completed native encrypted-data rehearsal. GitHub run 41050434226 at the earlier e8c4358f revision failed its Windows 0700 assertion; the workflow now targets Linux without omitting security cases. GitHub Ubuntu run 41054078619 at 90b03d7c also passed all 4,836 tests at 100% coverage on 2026-10-30.
