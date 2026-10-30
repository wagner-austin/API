---
title: WRDS substantive baseline readiness
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
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
  "libs/covenant_ml/src/covenant_ml/markets/encrypted_records.py": "6b3826b16f265301ec9923070cc89322427f8bc6"
  "libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py": "fad9820f53a021aa41df1d50b176f4af7d41a7bc"
  "libs/covenant_ml/tests/markets/test_encrypted_storage.py": "8f7e4479acae5a2ff6f38d88fb9368ceac7811b5"
  "libs/covenant_ml/tests/markets/test_encrypted_records.py": "2cfcf565e6b24f7b252059891a4cf9e545329923"
  "tools/hpc3/artifacts/wrds-monthly-runtime-20261030/provenance.json": "48a1ceff729c1af46bbf6a0748614eec152e2659"
  "tools/hpc3/artifacts/wrds-controller-verification-20261028/provenance.json": "b678f473462c24db2cbd1e510ffe6c034e36a9bb"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "e30492e1ef4ae30e3c3391560a066f51805d6250"
  "libs/covenant_ml/src/covenant_ml/markets/research_trial.py": "09aad4f0cf5563d59f0b431ffed6e3c5911bd42a"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_study.py": "2288c853df8b360d8d8654b9eb2578af2fb3982e"
  "libs/covenant_ml/src/covenant_ml/markets/research_final.py": "854dd095f76364bc1a00ca0e9e35ca2e06ce88f7"
  "libs/covenant_ml/src/covenant_ml/markets/crsp_final.py": "fa810eb315b13925c5664c35b883b80be0afee2d"
  "libs/covenant_ml/scripts/research_proposals.py": "f62f4f85739a00b2fbdc41f1e8724dd59188632f"
  "libs/covenant_ml/scripts/research_trial.py": "3e8ead7063b2e1835bb59131d73b9df0095b25d5"
  "libs/covenant_ml/tests/markets/test_research_final.py": "81e4a8f1f5220b502de59369c9a8ac1868f8ab77"
  "libs/covenant_ml/tests/markets/test_crsp_final.py": "67af9e640314ef33871e9a182d6511fb75dc0abc"
  "tools/fleet/src/fleet/core/research_proposal.py": "dd0d9f77373e490d69aef66d305768a959b6f525"
  "tools/fleet/src/fleet/cli/research_propose.py": "1c5e8bf6bab8bd79564235a292cc875a54a71fe2"
  "tools/fleet/tests/test_research_proposal.py": "e3032d998c3d7e4b0951b2db7d1be643978ac2c5"
  "tools/fleet/tests/test_research_proposal_host.py": "177c817e1992334468ed07142a01df4f71908aa6"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "2026-10-30 readiness recheck, board cda5d102-9f26-4c2a-bfbe-cfdbe17b33c6: live task_get reports 1e555057, 7c71808d and 364a4ba8 done after independent audit; dispatch_list at API e12eb1c8312b9d4ae57e38906ca787e02e0e094c reports research check b37025f3 (4786 passed, 100%), fleet check a4f8cdc8 (2495 passed, 100%) and real host check 16b62830 (79 passed). Local-file inspection of /home/corvis/research-control/wrds-validation-bda8ca2af/lib/python3.11/site-packages/covenant_ml/markets/research_trial.py still finds load_wrds_evaluation at lines 18 and 58, SHA-256 98e11f10c1a28dcaeaeec9fde9e21c22f93be0c8c8cd0add5c50e23e4805d271. Its entrypoint/scripts/research_proposals.py exposes init/status/propose/submit/reconcile and no select, SHA-256 ed2272d101c645375ada9cc46caa4f2c288f9cefd196c634827a81ca7d0785ef. These are local installed-file readings, not a new cluster execution."
fact_checked: 2026-10-30
confidence: medium
---

# WRDS substantive baseline readiness

- Research entry point: [[wrds-equity-research-protocol]].

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
a prerequisite for licensed staging, tracked in board task `e6729a16`.
It is not a capability of the installed runtime.[^encryption]

The shared library now implements age-encrypted files, separately named
train-validation/final bundles, and both ciphertext and plaintext SHA-256
identities using the existing staging records. Extraction needs only the public
recipient. Decryption accepts an in-memory identity and yields the extraction
manifest only after every file authenticates and matches its frozen identity.
Plaintext lives in a new mode-0700 temporary directory, with a declared size
limit, and is removed on ordinary exit or exception. The same API is intended
for WRDS Cloud and HPC3; neither native integration is verified yet.[^bundle]

This library does not supply key custody or grant final access. Cluster key
delivery must enforce the journal's existing sealed selection and single final
release. Same-user or administrative processes remain privileged; a private
directory alone is not an isolation boundary against them. Cluster scratch
reclamation after a forced kill also needs a native probe. The remaining
sequence is:[^bundle]

1. Integrate encrypted bundle delivery with existing staging, native execution
   and result verification. Preserve verifiable plaintext identities after
   private job storage is removed, and measure the key-access boundary.
2. Rehearse an allowed proposal through the real admission controller, shared
   queue, native job and reconciled RunRecord in one chain. The existing installed
   isolation test uses a recording admission stand-in; its result cannot stand
   for that complete chain. Exercise final release only in a separate rehearsal
   journal with synthetic final data, not the substantive holdout.
3. Materialize the frozen four-relation CRSP extraction with exact SQL, vintage,
   checksums and calendar coverage, then stage its verified manifest to HPC3.
   Keep final-period evidence confined to the evaluator.
4. Declare candidate/seed aggregation, the selection criterion and tie handling
   before inspecting study results. Run the bounded baseline and validation
   comparison first; seal the chosen candidate and use the one-shot final release
   afterward. The selection API enforces identity, not which candidate is best.

[^final]: `libs/covenant_ml/src/covenant_ml/markets/research_final.py`, `select_candidate` and `execute_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_final.py`, `run_crsp_final`; `libs/covenant_ml/src/covenant_ml/markets/crsp_study.py`, `crsp_fold`, `crsp_arms` and `crsp_known_panel`; `libs/covenant_ml/src/covenant_ml/markets/research_trial.py`, `require_trial_lineage` and `publish_monthly_record`; `libs/covenant_ml/scripts/research_proposals.py` and `libs/covenant_ml/scripts/research_trial.py`; `libs/covenant_ml/tests/markets/test_research_final.py` and `test_crsp_final.py`, real journals, extracts and trainers; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission". Fleet job 379d582f-130b-47c4-8579-56ec1be8d586 passed make check for libs/covenant_ml at API 06636dbbebd56771e170b45a7f190309f396cde1 on Colossus at 2026-10-30T05:45:28Z: 4,786 tests, 100.00% coverage. The earlier job 2e9095f9 at bddfcab33 ran every test green but failed the test-quality guard on one substring assertion, repaired in 06636dbbe. On main at 85e408c1c08cd3ff5faaf88416f4d62965ed64a3, job 8054c71b passed libs/covenant_ml on Colossus at 2026-10-30T05:57:25Z: 4,786 tests, 100.00% coverage. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.

[^readiness]: `tools/hpc3/artifacts/wrds-controller-verification-20261028/provenance.json`, controller_source_commit, controller_python and attempts; the October 30 local-file and live-tool reads recorded in this page's provenance. Implementation and test sources are cited in [^installed] and [^final]. Remaining steps are operational prerequisites, not completed jobs or performance findings.

[^monthly-runtime]: `tools/hpc3/artifacts/wrds-monthly-runtime-20261030/provenance.json`, source_commit, source_archive_sha256, controller_python_sha256, controller_entrypoint_hashes, first_party_wheel_sha256, image and evidence_sha256. Its private evidence includes native accounting, the generated build transcript, captured environment and staging manifests. At that exact source revision, fleet jobs 2eed7059, 7d1ae3f0 and 20312c9e passed 4,786 evaluator, 2,006 HPC3 and 2,510 fleet tests respectively, each with 100% coverage. Board task 474c840b.

[^encryption]: Board task e6729a16-3d8b-4f0f-9d68-73ea5d56b95d, October 30 operator requirement and acceptance criteria. This paragraph specifies the required workflow; it reports no completed encryption or key-custody measurement.

[^bundle]: `libs/covenant_ml/src/covenant_ml/markets/encrypted_records.py`, `decode_encrypted_bundle`; `libs/covenant_ml/src/covenant_ml/markets/encrypted_storage.py`, `encrypt_bundle`, `encrypted_stage_manifest` and `decrypted_bundle`; `libs/covenant_ml/tests/markets/test_encrypted_storage.py` and `test_encrypted_records.py`. Fleet job 76a1c77b-cd31-40d1-ad3d-2ae4b447f219 passed the full libs/covenant_ml make check at API e8c4358f29a834d8c98cc098a082f7b99ce3a0db on Loki, ending 2026-10-30T11:45:25Z: 4,815 tests, 100% coverage, lint and strict typing passed. Its 29 new cases cover the existing CRSP reader, wrong keys, altered/truncated ciphertext, traversal, output bounds and cleanup after an evaluator exception. This is a library measurement, not native key delivery, final authorization, forced-kill reclamation or a completed queue rehearsal. Board task e6729a16.

[^installed]: `tools/fleet/src/fleet/core/research_proposal.py`, `propose`; `tools/fleet/src/fleet/cli/research_propose.py`, `main`; `tools/fleet/src/fleet/core/research_control.py`, `read_pinned` and `invoke`; `tools/fleet/tests/test_research_proposal.py`, admitted and refused recording, no evaluator path in any daemon argument, drifted-input and isolation refusals; `tools/fleet/tests/test_research_proposal_host.py`, `test_installed_proposer_cannot_reach_evaluator_assets`. Host lane job fc8bc3c5-ad1b-479e-b44c-ddfcd40152ad passed tools/fleet-execution-linux at API 06636dbbebd56771e170b45a7f190309f396cde1 on Sedona at 2026-10-30T05:46:34Z: 79 cases ran, 0 skipped, 0 failed. Full check b01963a6-2105-49a3-a81a-8551982ca23d passed tools/fleet at the same commit on Sedona at 2026-10-30T05:43:08Z: 2,479 tests, 100.00% coverage. Commit 06636dbbe was staged on a pushed-tests ref and rebased unchanged onto main as part of 85e408c1c08cd3ff5faaf88416f4d62965ed64a3; there, host lane job 6833f095-6fe8-48c5-a464-548cf228597b passed on Colossus at 2026-10-30T05:58:34Z (79 cases, 0 failed, 0 skipped) and full check f7581b88 passed tools/fleet on Loki at 2026-10-30T05:55:25Z (2,495 tests, 100.00% coverage). The admission controller in the host case is a recording stand-in for `research_proposals propose`; the real controller's recording of proposals and refusals is the covenant_ml suite's. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.
