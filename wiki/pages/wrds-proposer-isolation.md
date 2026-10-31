---
title: WRDS proposal-program isolation
tags: [wrds, research, evaluation, provenance]
hubs: [libs]
related: ["[[wrds-equity-research-protocol]]"]
source_paths:
  - "libs/covenant_ml/docs/MARKET_RESEARCH.md"
  - "libs/covenant_ml/src/covenant_ml/markets/research_journal.py"
  - "libs/covenant_ml/tests/markets/test_research_selection.py"
  - "tools/fleet/src/fleet/core/research_control.py"
  - "tools/fleet/src/fleet/core/research_proposer.py"
  - "tools/fleet/tests/test_research_proposer.py"
  - "tools/fleet/tests/test_research_proposer_host.py"
  - "tools/fleet/src/fleet/core/research_proposal.py"
  - "tools/fleet/src/fleet/cli/research_propose.py"
  - "tools/fleet/tests/test_research_proposal.py"
  - "tools/fleet/tests/test_research_proposal_host.py"
  - "tools/fleet/tests/test_research_proposer_remote.py"
  - "tools/fleet/tests/test_research_proposer_hub.py"
  - "tools/fleet/tests/test_research_proposal_report.py"
source_git_blobs:
  "tools/fleet/tests/test_research_proposer_remote.py": "746f23412b9c7c5aff8838bc5a35b59f04b25742"
  "tools/fleet/tests/test_research_proposer_hub.py": "323f96f26ccf0cb43fa78903f1a2ac541d55942a"
  "tools/fleet/tests/test_research_proposal_report.py": "a3711895f09c8266571b621e9ebdb18e116a64f0"
  "libs/covenant_ml/docs/MARKET_RESEARCH.md": "284621906d62f97da2080ca823bdc971aa9a0454"
  "libs/covenant_ml/src/covenant_ml/markets/research_journal.py": "a36df5c2e20b454eabd26f3acfc75f07f1c3485e"
  "libs/covenant_ml/tests/markets/test_research_selection.py": "b2939285c746d390261861ed80edae3455000a84"
  "tools/fleet/src/fleet/core/research_control.py": "ba591e0282354ffd6b6293717384ada763077f05"
  "tools/fleet/src/fleet/core/research_proposer.py": "62a07acd52b848b65014155466fe268beecfe247"
  "tools/fleet/tests/test_research_proposer.py": "04d28830b7b0fdfbf20b478f7e7fdeef25083d35"
  "tools/fleet/tests/test_research_proposer_host.py": "41ddec456e01f4b2b10bb180fa0cebe67f6ce45e"
  "tools/fleet/src/fleet/core/research_proposal.py": "e4276cfec0006ed26aa8fb3c7b8fb8cdcb93a179"
  "tools/fleet/src/fleet/cli/research_propose.py": "f6f9b337556473eddc55eb72f1d8cc4ce708bd65"
  "tools/fleet/tests/test_research_proposal.py": "414e637e3c4dc1a8656951f049c42dbd52bb1f28"
  "tools/fleet/tests/test_research_proposal_host.py": "6c9656d052b58508be9ee7461010dba26bc457d1"
provenance:
  - "Lifted from wrds-equity-research-protocol at API 14b60b8610c98f17357869422cf2239ef4fddbf8 on 2026-10-30; dated measurements retain their original scope. Board task daca401b-a9cc-4cc8-8626-8ee760bedad5."
  - "Operator request 2026-10-28: bounded extraction and replay, define substantive evaluation before expansion, then integrate agent-driven research into existing systems; board 35895aac-1b12-4cf5-b868-516196914789"
fact_checked: 2026-10-30
confidence: medium
---

# WRDS proposal-program isolation

- Research entry point: [[wrds-equity-research-protocol]].

The coordinator can assign proposer, runner and reviewer work using its current
tasks. The evaluator enforces dates, costs, permitted changes and trial limits.
The proposer receives development/validation evidence only. A prompt telling an
agent with unrestricted SSH to avoid a directory is not isolation, so a
proposer is a program, not a session: the boundary on this page is what a proposer
holds. An agent session that holds the evaluator's account, SSH or WRDS
credentials is privileged administrative access, outside any boundary here,
and must not be described as a restricted proposer.[^request][^installed]

`fleet.core.research_proposer.run_proposer` executes an offline Python proposal
program on the existing rootless Docker lane. The trusted caller supplies an
already-present immutable image ID and approved context. The process receives no
host mounts, an empty environment, no network, a read-only root, no capabilities,
and a non-root user. The decoder requires rootless operation and cgroup-v2 memory,
CPU and process limits; its JSON field is `CpuCfsQuota`, and an info carrying
only another spelling of that key (`CPUCfsQuota`, `cpuCfsQuota`, `CpuCFSQuota`,
`cpu_cfs_quota`) is refused, so no fallback can pass. Limits are one CPU,
256 MiB, 64 processes, a caller-selected program deadline of 1–120 seconds,
and 8,192 bytes each for source, context and returned JSON. Daemon operations
have separate 30-second deadlines, so the program deadline is not an overall
request deadline. An invocation-specific label scopes removal, followed by a
query requiring no matching container to remain.[^proposer]

This component returns an untrusted object. `fleet.core.research_proposal`
is its one caller and the installed path: `python -m fleet.cli.research_propose
--root R --digest D` reads the evaluator's `R/D/proposal.json` by digest, verifies
the pinned program, context, policy and controller bytes, runs the program,
writes the returned object to `R/D/proposals/<sha256>.json` and passes that file
to the maintained `research_proposals propose`. Admission records it before
decoding, keeps refusals (exit 2 is accepted as a recorded refusal), and an
admitted attempt then reaches HPC3 through the existing queue unchanged; there
is no new queue or public submission tool. A drifted input or an isolation
refusal writes no proposal and invokes no admission.[^installed]

The private specification now requires explicit placement: `host: null` uses
the caller's rootless daemon; an SSH alias such as `colossus` uses that host's
existing `execdocker` lane. The shared SSH transport receives individually
quoted arguments and the operation's deadline. Policy, journal, controller and
data stay with the evaluator. A transport failure propagates, including a
failure that prevents confirming cleanup; placement never retries on another
host. `--report-task` optionally publishes the admission metadata under the
`fleet-research-proposer` service identity. If reporting fails after admission,
inspect the journal before rerunning.[^remote]

On October 30 at 23:59 UTC, the installed CLI ran the same pinned proposal
program twice on Colossus from Ziggurat and returned its output directly to the
frozen local admission controller. A fresh synthetic journal recorded attempt 1
as `admitted` and attempt 2, requesting the final partition, as `refused` with
`holdout_forbidden`. The two program-authored reports bind the specification,
image, candidate and returned proposal hashes. A read-only query found both
reports and only one after changing the expected validation specification hash.
No licensed rows or keys were involved; no queue submission or new cluster
allocation was part of this handoff measurement.[^remote]

The real hub lane subsequently passed all ten cases with no skips, including
literal quotes and shell metacharacters across SSH, and a real timed-out
proposal whose owned container was removed. The Linux isolation lane passed
all 95 cases. An earlier hub run passed the two remote cases but failed an
unrelated user-service identity assumption; the corrected test starts and
inspects a bounded user service instead of treating the check's system service
as a user service. No production tick restriction was relaxed.[^remote-gates]

[^remote-gates]: `tools/fleet/tests/test_research_proposer_hub.py`, `test_actual_remote_proposal_and_timeout_cleanup`. Live fleet job `b2285d48-6c1f-4ee8-b1b9-3c38c6926a6c` at API `f62c99f16582589365902255c6e6bc658d6e0c75`, Diphtheria, 2026-10-31T00:03:37Z: 10 passed, 0 skipped, both remote readings report `remaining=0`. Full package job `b01f1bec-8498-4428-9b49-5caf9be7ab5a` at the same commit passed 2,561 tests with 100% coverage on Loki. Linux job `7f4fb878-467f-4220-b5aa-0c580a483663` at API `ca178d11ab74ab55eb24a3e7dbf69bdd6fa75207` passed 95 cases on Lavender. Earlier hub job `a92ec475-ad65-4313-867e-18c1cef68f25` at that earlier commit failed with 9 passed and 1 failed; it is not a passing suite receipt.

[^remote]: `tools/fleet/src/fleet/core/research_proposer.py`, `_checked` and `run_proposer`; `tools/fleet/src/fleet/core/research_proposal.py`, `propose`; `tools/fleet/src/fleet/cli/research_propose.py`, `main`; `tools/fleet/tests/test_research_proposer_remote.py`, quoting, transport failures, cleanup and local admission cases; `tools/fleet/tests/test_research_proposer_hub.py`, `test_actual_remote_proposal_and_timeout_cleanup`; `tools/fleet/tests/test_research_proposal_report.py`, service identity and outcome binding. API `ca178d11ab74ab55eb24a3e7dbf69bdd6fa75207` passed full check `672d626f-e949-44a2-921b-1df16de480b0` on Colossus: 2,561 tests, 100% statement/branch coverage. Read-only source installation `/home/corvis/research-control/proposer-remote-ca178d11a` runs with the existing final controller Python; it does not replace the frozen evaluator wheels. Private journal `/home/corvis/research-data/wrds/experiments/remote-proposer-20261030/journal.sqlite`. Reports `c389d77a-cd5b-4a2a-82e8-b5f80f4ad938` and `a823892d-3e21-45a1-adc3-a011d491a535` on task `45ff6e73-ec8c-41ff-aae4-39d00423dddb`, author `fleet-research-proposer`. Validation spec SHA-256 `3ee8eac1611d368a65b9238806a43382d233a1c186c4daa385ef7898b2daac70`; refused final spec `815fade7b1f1c04aaea6baf4e8f777b7e344a90cef70783f13ffe931b569ad92`.

Enforced boundary, by what the proposer holds: the program text and approved
context only; no host path (the evaluator's policy, journal, final extract,
controller code, daemon socket and `~/.ssh` are absent inside the process), no
writable root, no inherited environment, so no WRDS, SSH or database credential,
no network route, and no capability or root user. It cannot replace evaluator
code or policy: those files are absent where it runs, and the evaluator rejects
any changed bytes by digest before running. Outside it: an administrator or an
agent session on the evaluator's account, kernel or daemon compromise, and the
trusted image and context the evaluator chooses, which must exclude held-out
data.[^proposer][^installed]

The actual host suite passed on Colossus at API
`a9be5c1ee955b9bc576268b8183df3b6f619a836`: 78 cases, zero failures and zero
skips. Its five proposer cases verify missing host-private paths, denied root
writes, absent inherited credentials, unreachable network, exact cgroup limits
and an allowed JSON result, plus cleanup after nonzero exit, timeout, malformed
JSON and oversized output. That was component evidence.[^proposer]

At that same commit the full `tools/fleet` check passed 2,463 tests with
100% statement and branch coverage, strict typing and guards clean. The earlier
selection/release correction also passed independent audit `27406acc` on
2026-10-30.[^proposer][^selection]

Measured denial on the installed path, API `06636dbbe`, host lane on Sedona as
the `execdocker` user: the installed CLI ran a probe program against real
evaluator files at their host paths. Reading the policy, journal, final-period
CSV, a WRDS `.pgpass`, the daemon socket, `~/.ssh` and the final CSV through
`/proc/1/root` each failed with errno 2; overwriting the controller code and the
policy failed with errno 2; connecting out failed with errno 101; the process
ran as uid 65534 with only `LC_CTYPE` and `PATH` in its environment, so a
`PGPASSWORD` canary set in the evaluator's environment was absent. All five
host files kept their exact bytes. Allowed path, same run: the returned object
was written content-addressed and handed to the controller, whose transcript
records one invocation, exit 2 accepted as a recorded refusal.[^installed]

[^selection]: `libs/covenant_ml/src/covenant_ml/markets/research_journal.py`, `_transition` and `_final_identity`; `libs/covenant_ml/tests/markets/test_research_selection.py`, real SQLite selection, release and atomic refusal cases; `libs/covenant_ml/docs/MARKET_RESEARCH.md` § "Frozen research proposal admission".

[^installed]: `tools/fleet/src/fleet/core/research_proposal.py`, `propose`; `tools/fleet/src/fleet/cli/research_propose.py`, `main`; `tools/fleet/src/fleet/core/research_control.py`, `read_pinned` and `invoke`; `tools/fleet/tests/test_research_proposal.py`, admitted and refused recording, no evaluator path in any daemon argument, drifted-input and isolation refusals; `tools/fleet/tests/test_research_proposal_host.py`, `test_installed_proposer_cannot_reach_evaluator_assets`. Host lane job fc8bc3c5-ad1b-479e-b44c-ddfcd40152ad passed tools/fleet-execution-linux at API 06636dbbebd56771e170b45a7f190309f396cde1 on Sedona at 2026-10-30T05:46:34Z: 79 cases ran, 0 skipped, 0 failed. Full check b01963a6-2105-49a3-a81a-8551982ca23d passed tools/fleet at the same commit on Sedona at 2026-10-30T05:43:08Z: 2,479 tests, 100.00% coverage. Commit 06636dbbe was staged on a pushed-tests ref and rebased unchanged onto main as part of 85e408c1c08cd3ff5faaf88416f4d62965ed64a3; there, host lane job 6833f095-6fe8-48c5-a464-548cf228597b passed on Colossus at 2026-10-30T05:58:34Z (79 cases, 0 failed, 0 skipped) and full check f7581b88 passed tools/fleet on Loki at 2026-10-30T05:55:25Z (2,495 tests, 100.00% coverage). The admission controller in the host case is a recording stand-in for `research_proposals propose`; the real controller's recording of proposals and refusals is the covenant_ml suite's. Board task 364a4ba8-0ff0-461c-ba7f-dfe7a4b039ba.

[^proposer]: `tools/fleet/src/fleet/core/research_proposer.py`, `run_proposer`, `_runtime` and `_remove_owned`; `tools/fleet/tests/test_research_proposer.py`, policy and lifecycle cases; `tools/fleet/tests/test_research_proposer_host.py`, actual process denial and failure cases. Component task 7c71808d-4328-471e-81c2-0fb40453d561; verification repair 1e555057-c0cf-40a0-8642-781406b4df6c. Initial strict check 4e65d678 rejected discarded denial errors; initial real-host check df645d67 rejected the incorrect CPU JSON key before launching any proposal. Neither is passing evidence. Corrected real host check 60942856-9290-41e2-aa4f-086df60599a4 passed at the cited API commit on 2026-10-30T03:11:41Z, 78 passed, zero skipped, 30 seconds of a 300-second budget; the host lane intentionally runs without coverage because the separate complete package check owns that requirement. Full check a74cf4a3-93bb-4bdf-ab4e-0d6ea640ac36 passed at the same commit on Pendragon on 2026-10-30T03:16:17Z, 2,463 passed, 100.00% coverage, 38 seconds of 300. Its audit found no check that would fail if an alternate key spelling were accepted, so `test_only_dockers_json_cpu_quota_key_is_accepted` was added at API 629be28f6; there, check de6506a9-360f-45ba-a153-cfd4818b9097 passed on Loki at 2026-10-30T05:24:31Z (2,467 passed, 100.00% coverage) and host check 557b9c97-ddc7-4c41-a205-776356c32546 passed on Serendipity at 2026-10-30T05:24:06Z (78 passed, 0 failed).

[^request]: Board task 35895aac-1b12-4cf5-b868-516196914789, § "OPERATOR", submitted 2026-10-28T16:38:53Z; the study rules are this session's proposed design, introduced in API commit 3358191792a4a655825b5647694fc0430cf7bd33 and calendar-corrected in 78b009119, not quotations or empirical findings.
