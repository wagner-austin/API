---
title: Image jobs declare their shared-data mounts
tags: [images, storage, provenance]
hubs: [cluster-facts]
related: ["[[node-local-scratch]]", "[[image-ledger-lessons]]"]
source_paths:
  - "src/hpc3/core/image_exec.py"
  - "src/hpc3/core/sbatch.py"
  - "src/hpc3/contracts/image.py"
  - "tests/test_image_mount_shell.py"
  - "artifacts/wrds-declared-mounts-20261102/evidence.json"
  - "artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json"
source_git_blobs:
  "src/hpc3/core/image_exec.py": "5d6a6241b29c5974f02e56efb12ca07eafd96463"
  "src/hpc3/core/sbatch.py": "bf884f896e0eaaeafd595a034030368467f04026"
  "src/hpc3/contracts/image.py": "382e3133643bb10c191337c0e9eb507d170002d6"
  "tests/test_image_mount_shell.py": "345bd899563c0f7be1c833e3ecb1cbc24e1aee91"
  "artifacts/wrds-declared-mounts-20261102/evidence.json": "85aa658a3d949e054048fa129d41a541d16f3281"
  "artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json": "fe2f9f8774c053a9e07826ace70b9187ae9b0ef0"
provenance:
  - "Board task 0a0e5913-7be2-4be5-8225-2833d21981f8; controller source abb117967be4f07194c748c9ad1d3e9eb6d3e676."
  - "API libs/covenant_ml/src/covenant_ml/markets/research_cluster.py, outside this wiki workspaceRoot: required operator binds copied into validation and final recipes."
fact_checked: 2026-11-02
confidence: high
---

# Image jobs declare their shared-data mounts

`ImageReference.binds` owns required data mounts. The shared `image_exec` owner
disables implicit site bind-paths and hostfs with `--no-mount bind-paths,hostfs`,
clears the three `APPTAINER_*` and three `SINGULARITY_*` mount variables, and
quotes each declared same-path bind. Preflight, the batch payload and image
commit inspection reuse that owner. Standard home, temporary and device mounts
remain; this is not an access-control sandbox. GPU jobs retain `--nv`.[^owner]

The research recipe requires an explicit mount list and no longer inserts
`/pub`. Home-only requests use `[]`; shared-data requests declare the needed
directory. Existing frozen requests and installed controllers remain historical:
the native probes used a pinned source overlay over the existing frozen
interpreter, not a fleet-wide replacement.[^deployment]

## Why the dependency mattered

On November 2, a home-image preflight still depended on BeeGFS through explicit
`/pub` and site mounts. It waited in `FhgfsOps_revalidateIntent` and reached the
600-second controller deadline. A diagnostic with site mounts disabled answered;
the unchanged canonical request subsequently passed when storage responded.
Removing that unnecessary dependency does not repair BeeGFS itself.[^incident]

## Native verification and limits

Both CPU jobs requested one CPU, 4 GiB, five minutes, the free partition and
no requeue. They reused image `369ae415…78fde` and the maintained stage,
preflight, submit and execution-inspection commands.[^native]

| Job | Mounts observed inside image | Result |
|---|---|---|
| `58776025`, `hpc3-l18-05` | No BeeGFS mounts | Home fixture verified; completed `0:0` in 6 seconds. |
| `58776026`, `hpc3-l18-04` | Exactly the declared `/pub/wagnera3/research/mount-probe-0a0e5913-v2-20261102` | Home and shared fixtures verified; completed `0:0` in 7 seconds. |

Both used `$TMPDIR` under `/tmp/wagnera3/<jobid>`, created a mode-0700 private
directory, verified a write/read and removed it normally. Native inspection
verified four and five input/runtime hashes respectively, result and RunRecord
identities, and artifact timestamps within the allocations. This proves neither
forced-termination cleanup nor hostile-account isolation. No licensed data or
final keys were used; no new GPU allocation was tested.[^native]

Earlier jobs `58775808` and `58775824` failed because the diagnostic pinned CPU
determinism after importing NumPy. The guard refused a misleading record. Their
requests and failures remain archived; new requests fixed import order. Full
controller and evaluator checks passed 2,201 and 5,381 tests with 100% coverage;
the shell tests also verify literal paths and propagation of nonzero exits.[^checks]

[^owner]: `src/hpc3/core/image_exec.py`, `APPTAINER_EXEC`, `mount_arguments`, `run_inside_image` and `gpu_arguments`; `src/hpc3/core/sbatch.py`, `payload_lines` and `code_provenance_export`; `src/hpc3/contracts/image.py`, `ImageReference`.
[^deployment]: `artifacts/wrds-declared-mounts-20261102/evidence.json`, `controller`, `image`, `effective_flags`, `cleared_variables`, `freeze_posts` and `limits`; API `research_cluster.py`, `decode_research_cluster` and `prepare_bundle`, external source identified in frontmatter.
[^incident]: `artifacts/wrds-protocol-v3-rehearsal-20261102/evidence.json`, `preflight_incident`; [[node-local-scratch]], § "The login node stalls on /pub too, and nothing on the hub can bound it".
[^native]: `artifacts/wrds-declared-mounts-20261102/evidence.json`, `resources`, `native_service_posts` and `files`; hash-addressed sibling `home-native-evidence.json`, `shared-native-evidence.json`, `home-result.json`, `shared-result.json` and both `.sbatch` files, accounting, verified inputs, result mounts and scratch observations.
[^checks]: `artifacts/wrds-declared-mounts-20261102/evidence.json`, `earlier_failure`, `fleet` and `files`; sibling failed native reports and stderr; `tests/test_image_mount_shell.py`, `test_shell_preserves_paths_removes_ambient_mounts_and_propagates_exit`.
