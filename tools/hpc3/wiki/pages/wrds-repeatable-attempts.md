---
title: WRDS experiments need durable attempts and verified results
tags: [wrds, reproducibility, operations]
hubs: [operations]
related: ["[[one-ledger-per-repository]]", "[[wrds-filesystem-catalog]]", "[[node-local-scratch]]"]
source_paths:
  - "src/hpc3/contracts/wrds_run.py"
  - "src/hpc3/core/wrds_jobs.py"
  - "src/hpc3/core/wrds_results.py"
  - "src/hpc3/core/wrds_scheduler.py"
  - "src/hpc3/core/wrds_attempt_report.py"
  - "src/hpc3/cli/wrds.py"
  - "src/hpc3/wrds/worker.py"
  - "specs/cleargbm-paired-image.json"
  - "runs/bench-wrds-synthetic-a.json"
  - "runs/bench-wrds-synthetic-b.json"
  - "runs/bench-wrds-image-verify.json"
  - "runs/wrds-synthetic-stage.json"
  - "runs/wrds-entrypoint-stage.json"
  - "artifacts/wrds-synthetic-2a50e93ca/agreement.json"
  - "artifacts/wrds-synthetic-2a50e93ca/configuration-comparison.json"
  - "artifacts/wrds-crsp-14593-2019-2020-1028/provenance.json"
source_git_blobs:
  "src/hpc3/contracts/wrds_run.py": "00660ff78017fe0dd51c344ba8f3f0b51ae0b23f"
  "src/hpc3/core/wrds_jobs.py": "e79b573ab5a1f145599018c38c95fcde511d9d7b"
  "src/hpc3/core/wrds_results.py": "758c87f30dfd8cc00dc71e0c89b0e393af2f157a"
  "src/hpc3/core/wrds_scheduler.py": "4b558fcd7669735fbdebe6dbc735c527b1b1d50d"
  "src/hpc3/core/wrds_attempt_report.py": "bfc35caed2d3825c6bce7664f4e6617746d1b7f7"
  "src/hpc3/cli/wrds.py": "aa834308825091ae3c0fe78ba74d0c4be91cb0d9"
  "src/hpc3/wrds/worker.py": "e8a91def08eb391ac271f04c5cf3dbc3561470a5"
  "specs/cleargbm-paired-image.json": "abe322bc23bb1771d3c248b299de61f1248fdd71"
  "runs/bench-wrds-synthetic-a.json": "48a25a0e3b5624bf5dfc2a0fbe457d925b566d25"
  "runs/bench-wrds-synthetic-b.json": "d3df02db2cf4c5743ed9a4de8352dfb6491f1bcd"
  "runs/bench-wrds-image-verify.json": "e0b7c5c599f69f0e01cef31478ca3c97aa10fb63"
  "runs/wrds-synthetic-stage.json": "a9aaa744576049beec872f55cf24fde5a0be44a5"
  "runs/wrds-entrypoint-stage.json": "d9b5d637c391388e0e782212f123402ea5be5309"
  "artifacts/wrds-synthetic-2a50e93ca/agreement.json": "4554212fd3f74b2d40e7c8844607c395fd51f9f2"
  "artifacts/wrds-synthetic-2a50e93ca/configuration-comparison.json": "4c03455a561841511de0c13141c7bfe871c96a11"
  "artifacts/wrds-crsp-14593-2019-2020-1028/provenance.json": "300beff429aa112b7cd21fde60f0c8985953a6fb"
provenance:
  - "2026-10-28 bounded CRSP pilot: WRDS 40534563 produced 505 rows for PERMNO 14593 in 2019–2020; hpc3.cli.wrds report posted measured accounting and hashes as load-report-35895aac on board 35895aac"
  - "HPC3 real-data replays 58575271 and 58575282 completed 0:0 on hpc3-17-01 in 11s and 10s; native accounting, checksum-success lines and traces preserved beside the metadata-only provenance record; numerical outputs remain private"
  - "Board task d43697b3: operator requested the existing HPC system for repeatable WRDS experiments"
  - "2026-10-27 America/Los_Angeles: WRDS jobs 40530016 and 40530018, same synthetic input/code/runtime/worker/seed, collected through hpc3.cli.wrds"
  - "Ziggurat /home/corvis/research-data/wrds/experiments/system-probe-20261027/repeatability-proof.json pins both completion records and common RunRecord files; metrics SHA256 60dc457a811582434c73d554c0df58942856247aac8874abcab1942e86d52756; observed sum 45195"
  - "Board task d43697b3, posts by load-report-d43697b3 at 2026-10-28T04:18:48Z, 04:19:54Z, 04:20:58Z and 04:21:34Z: hpc3.cli.wrds report at API f87c0758f measured jobs 40530016, 40530018, 40530005 and 40530008 on WRDS"
  - "Job 40530005's payload raised SystemExit(17); the worker refused it with WRDS_PAYLOAD_FAILED, so qacct records failed 0, exit_status 1 on wrds-sas24-w. Job 40530008 was cancelled through the CLI; qacct records failed 100, exit_status 137 on wrds-sas20-w. Neither attempt directory holds completion.json"
  - "Earlier job 40529992 failed at its worker checksum because Grid Engine's shell wrapper altered escaping; job 40529996 completed after -shell n"
  - "Board task eadd19d5: operator moved numerical evaluation compute to HPC3; no licensed WRDS rows were extracted for these checks"
  - "HPC3 image build 58566045: COMPLETED 0:0 in 85s on hpc3-20-20; SIF /pub/wagnera3/images/cleargbm-paired-2a50e93ca/cleargbm-paired.sif, SHA256 f5d16ad3732cb1ba196b2c56aa56fa3465dcc183c75fc73924a5aecf455fb16b"
  - "HPC3 synthetic replays 58566380 and 58574881: COMPLETED in 13s on hpc3-22-05 and 6s on hpc3-20-20; each 2 CPUs, 4 GiB, free partition, 0 SU"
  - "HPC3 image verification 58574938: COMPLETED 0:0 in 5s on hpc3-17-13; generated selfcheck SHA256 9db790abc4b8fb0bedaeb7c132276635a4806188bf18706479792ef66c108efb checked the strengthened committed spec; selfcheck and pip check passed"
  - "Ziggurat /home/corvis/research-data/wrds/runtime/2a50e93ca preserves source archive, six wheel digests, original captured spec, rendered build, synthetic input and query, run documents and collected logs; source commit 2a50e93ca06fcc82896c143a37ee247187cd2b12, archive SHA256 02e542cdfc0909cda9a228bf6cf535397e8c395502619979bb4b5defade3d2d8"
  - "Remote and collected replay files SHA256: both manifests db9c0d4cd183655762f49ad7940db8e68b62e90e66409da0bc30afb23243ab05; both RunRecords 6a317031cd5ba5cafbedbc8ab0db75a6d59acd81e0a034a931b9721abc3f1fac"
  - "API libs/covenant_ml/README.md section WRDS-shaped evaluation inputs and libs/platform_core/src/platform_core/run_record.py agree_across_runs, outside this wiki workspaceRoot"
fact_checked: 2026-10-28
confidence: high
---

# WRDS experiments need durable attempts and verified results

WRDS execution lives in the existing HPC package, with a Grid Engine adapter
and the shared research `RunRecord`. The run identity includes code and input
digests, Python binary/version and distribution versions, seed, arguments,
resources and outputs. Local checkout paths are removed from that identity.
An explicit attempt label distinguishes repeats of the same experiment.[^identity]

Submission exclusively creates the attempt directory, verifies uploaded
bytes and asks the scheduler to validate the resource request. It submits
held, persists the job ID remotely and in the local ledger, then releases.
An interrupted release is recovered with `resume`; reusing a directory never
silently creates another job. The staged source and metadata are made
read-only to prevent accidental edits, not to isolate hostile payloads.[^submit]

The worker hashes declared inputs before and after execution and rejects
runtime drift, nonzero exits, missing outputs and nonfinite metrics. It writes
its completion record only after those checks pass. Collection additionally
requires successful scheduler accounting, matching run/attempt/job/worker
identity and verified transferred output hashes. Queue absence alone is
`accounting-pending`, never completion.[^worker][^collect][^status]

A session's note restating job numbers and digests is not evidence an audit
can accept, so `report` measures one attempt on WRDS and posts the reading on
its board task as `load-report-<task>`. The post carries qacct's execution
host, failure code and exit status, the spec's code, input and runtime
digests, and either the attested output and run-record digests collect would
write or collect's own refusal and whether `completion.json` exists. The
completed path calls the same attestation check collect uses, so a report
cannot call an attempt good that collect would refuse.[^report][^cli]

The synthetic fixture ran twice with the same worker, code, five-number
input, runtime and seed. Jobs **40530016** and **40530018** ran on two
different compute nodes, `wrds-sas35-w` and `wrds-sas27-w`, each with failed
0 and exit status 0. They produced identical metrics and identical shared
run-record bytes; the sum was **45195**. The deliberate failure (job
40530005: the payload exited 17, so the worker exited 1) and the cancellation
(job 40530008: failed 100, exit status 137) were both refused by collect and
neither wrote a completion record. This establishes the tested execution
path, not determinism for arbitrary research programs or permission to export
particular licensed datasets.[^live]

Runtime pins cover the interpreter binary and installed distribution
versions; they do not freeze all package contents into a container. Payloads
must use the supplied seed, declare their data dependencies and write finite
metrics. Current limits are individual Python CPU jobs, up to eight slots,
48 GiB total and seven days; scheduler admission remains authoritative.
UCI's Slurm sweep, chain and container commands are not WRDS commands.[^identity][^worker]

The first checksum wrapper also exposed a scheduler-specific failure: passing
a shell command through Grid Engine's default extra shell altered escaping.
The adapter now requests `-shell n` and explicitly invokes one shell with
quoted arguments. The failed job was recorded as failed rather than being
accepted because it had left the queue.[^scheduler][^live]

## Frozen numerical evaluation on HPC3

The operator chose HPC3 for the numerical workload. The existing paired
evaluation entrypoint now accepts a declared WRDS-shaped extraction manifest,
using the same ClearGBM, LightGBM and XGBoost arms, chronological splits and
shared RunRecord. The first deployment used the existing synthetic test
fixture: **no query downloaded WRDS rows**. Its 160 input rows yield 159
next-observation examples; this is an integration check, not a trading
backtest or evidence of predictive value.[^paired]

The frozen SIF contains six first-party wheels from source commit
`2a50e93ca06fcc82896c143a37ee247187cd2b12` and 52 pinned third-party
requirements. The existing image-capture and image-build workflow produced
it; each replay declares its content hash. The committed spec later gained
three additional symbol assertions. Job **58574938** ran the newly rendered
self-check against the original immutable SIF and passed, followed by
`pip check`. That job verifies the stronger checks without claiming the
original build contained them.[^image]

Jobs **58566380** and **58574881** ran on different HPC3 nodes using the same
synthetic manifest, image and seeds 42/43. The full result manifests and
RunRecords were byte-identical. The existing `agree_across_runs` reports
**45/45 identical observations**, no unmatched observations; the shared
configuration comparator reports `identical`. This establishes agreement
for these two runs only. The host fingerprint records OS, architecture and
logical core count, not CPU model, so that verdict has that resolution.[^replay]

The committed `bench-wrds-synthetic-a.json` and `-b.json` are the executed
recipes. For another attempt, copy one and give it a new name plus new
`--out` and `artifact` paths; then use the existing `hpc3-preflight`,
`hpc3-submit` and `hpc3-watch` commands with `runs/hpc3.json`. Input and
entrypoint stage manifests preserve their hashes. The entrypoint lives
outside the image and is separately tied to the source archive: the image
commit alone does not attest it. These earlier recipes use synthetic data;
the separately scoped real-data extraction and replay follow below.[^recipe]

## Real-data pilot and the next study

WRDS job 40534563 extracted 505 daily rows from `crsp_a_stock.dsf` for
PERMNO 14593, inclusive 2019-01-01 through 2020-12-31, with a 600-row cap.
The maintained worker saved the exact read-only query, timestamps and digests;
the existing collector verified the outputs. Five files were hash-verified
on HPC3. Replays 58575271 and 58575282 used the same snapshot and frozen SIF,
checking seven data/code/image files before evaluation. Their 45 observations,
result manifests and RunRecords matched exactly. Native scheduler records,
compute checksum lines, recipes and metadata hashes are retained; licensed
rows and numerical results stay under the private evidence root.[^crsp]

This is an engineering pilot. It does not establish profitability, historical
vendor vintage, or a monthly strategy. The proposed substantive protocol and
agent integration boundary are in the API wiki's
[`wrds-equity-research-protocol`](../../../../wiki/pages/wrds-equity-research-protocol.md).
The earlier frozen-runtime audit was reopened for insufficient machine-readable
live-run verification. Its repaired closure references a program-authored
native replay report; both the extraction and frozen-runtime tasks passed
independent audit on 2026-10-28. Installed
validation-controller runs and the subsequent repair are recorded in
[[wrds-installed-controller-verification]].[^crsp][^audit]

[^crsp]: `artifacts/wrds-crsp-14593-2019-2020-1028/provenance.json`; native `accounting-and-digests.txt`, `watch.txt`, `trace-a.txt`, `trace-b.txt` and `compute-checksums.json` beside it; service-authored WRDS report named in provenance.
[^audit]: Board task eadd19d5-6d5e-4c05-a8fe-1867337fdbef, original audit rejection 2026-10-28T16:35:56.756Z, repaired closure 2026-10-28T19:37:41.061Z and VERIFIED 2026-10-28T19:52:23.242Z; extraction task 35895aac-1b12-4cf5-b868-516196914789 VERIFIED 2026-10-28T19:52:01.056Z. Native report by hpc-report-14d9e920 on task 14d9e920-dd4c-41f7-8460-ccf4623844c7.
[^identity]: `src/hpc3/contracts/wrds_run.py:230` `decode_wrds_run` and `:345` `run_identity`.
[^submit]: `src/hpc3/core/wrds_jobs.py:245` `submit`, `:220` `resume` and `:170` `record_event`.
[^worker]: `src/hpc3/wrds/worker.py:92` `execute_run`.
[^collect]: `src/hpc3/core/wrds_results.py:53` `verified_completion`, `:74` `attested_completion` and `:127` `collect`.
[^status]: `src/hpc3/core/wrds_scheduler.py:181` `status` and `:113` `decode_accounting`.
[^scheduler]: `src/hpc3/core/wrds_scheduler.py:25` `job_argv`.
[^report]: `src/hpc3/core/wrds_attempt_report.py:70` `render` and `:25` `completion_presence`.
[^cli]: `src/hpc3/cli/wrds.py:80` `main`.
[^live]: The four posts by `load-report-d43697b3` on board task d43697b3 listed in the frontmatter provenance section, and Ziggurat `/home/corvis/research-data/wrds/experiments/system-probe-20261027/repeatability-proof.json:1`, fields `jobs`, `identity`, `worker_sha256`, `matching_outputs`, `observations` and `artifact_sha256`.
[^paired]: API `libs/covenant_ml/README.md` WRDS input contract; executed manifests under `artifacts/wrds-synthetic-2a50e93ca/`, fields `dataset`, `seeds`, `results` and `seed_comparisons`; synthetic provenance in `runs/wrds-synthetic-stage.json`.
[^image]: `specs/cleargbm-paired-image.json`; `runs/bench-wrds-image-verify.json`; job 58566045 build logs and job 58574938 verification logs identified in `provenance`.
[^replay]: `artifacts/wrds-synthetic-2a50e93ca/agreement.json` and `configuration-comparison.json`; scheduler placement and independently checked remote/collected file hashes in `provenance`; host fields in the preserved RunRecords.
[^recipe]: `runs/bench-wrds-synthetic-a.json`, `runs/bench-wrds-synthetic-b.json`, `runs/wrds-synthetic-stage.json` and `runs/wrds-entrypoint-stage.json`.
