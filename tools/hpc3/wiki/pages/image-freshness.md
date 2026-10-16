---
title: An image older than its command is refused at preflight
tags: [images, preflight, provenance]
hubs: [images-and-staging]
related: ["[[image-build-flow]]", "[[submission-rules]]", "[[spec-symbol-drift]]", "[[known-answers]]", "[[ledger-closures]]"]
source_paths:
  - "src/hpc3/core/entrypoint.py"
  - "src/hpc3/core/drift.py"
  - "src/hpc3/contracts/drift.py"
  - "src/hpc3/contracts/ledger.py"
  - "src/hpc3/core/preflight.py"
  - "src/hpc3/core/array_submit.py"
  - "src/hpc3/cli/trace.py"
  - "tests/test_entrypoint.py"
  - "tests/test_drift.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/core/entrypoint.py": "728bfb2f5678f0fc5cdd8333e1245f0d83338994"
  "src/hpc3/core/drift.py": "0d6242944b907a4c1a9b5028377d76894b661993"
  "src/hpc3/contracts/drift.py": "495319bda979c2eaa43db991174e74145d6bda7f"
  "src/hpc3/contracts/ledger.py": "b200073d657643908de5491eca70e1fabaeebb4c"
  "src/hpc3/core/preflight.py": "940f3f7520a93221b6718b3310e0eaa651f96465"
  "src/hpc3/core/array_submit.py": "205fb40432570438368f18ff26bcf9cbd8bee13a"
  "src/hpc3/cli/trace.py": "1984bfe5a85fd601ab9dd37e59bfbcd299e4ecdb"
  "tests/test_entrypoint.py": "05b02dcc8b56ba5ab9f8b5418da36a423403f699"
  "tests/test_drift.py": "df99de375093bcc2970a59b784f5c53619150684"
  "README.md": "ac67efe201119ac5992ff0f63881cbe5ac01d528"
provenance:
  - "board task 2c8161a3 (MCPs board), filed 2026-09-04: the registered mi image predates cartridge_benchmark and cartridge_composition_sweep"
  - "board task 7aab32ef (MCPs board), 2026-10-16: a module the image carries but whose source moved was admitted with no word"
  - "apptainer exec <image> cat /opt/env/GIT_COMMIT on login-i17, 2026-10-16, for all nine registered images"
  - "the entrypoint probe run against /pub/wagnera3/images/v23/abl.sif on login-i17, 2026-10-16"
  - "hpc3-preflight from ziggurat, 2026-10-16T17:55Z-17:58Z, HEAD 3dc31db77 with the source_paths rule later committed as API ac572060d in the working tree: runs/ka-probe-v23-v100.json and runs/cartridge-traits-gpt2-pilot-cpu-v56.json with runs/hpc3-mi.json, and one member of runs/sweep-cleargbm-p6-rung1.json with repo_commit set to the cluster checkout's 80221ea, with runs/hpc3.json; the run at 17:53:58Z without that rule printed 195 files for the first"
  - "the hub's ledger, C:/Users/Test/PROJECTS/API/tools/hpc3/runs/ledger.jsonl, backfilled 2026-10-16T17:51Z: 16064 rows given image_drift null, sha256 bf047d1d... before and 25291ed1... after, the original kept beside it as ledger.jsonl.pre-image-drift-20261016"
  - "Hpc3ErrorCode ENTRYPOINT_NOT_INSTALLED, ENTRYPOINT_PROBE_UNREADABLE, ENTRYPOINT_IMAGE_UNSTAMPED, ENTRYPOINT_IMAGE_COMMIT_UNKNOWN and ENTRYPOINT_DRIFT_UNREADABLE, libs/platform_core/src/platform_core/error_codes_tooling.py, outside this wiki's workspaceRoot"
fact_checked: 2026-10-16
confidence: high
---

# An image older than its command is refused at preflight

Onboarding is enforced (`PROJECT_UNIMAGED`) and identity is enforced (a
`.sif` is pinned by sha256), but until 2026-10-16 whether an image was new
enough for the command a run named was convention. The registered `mi` image,
v23, predates `cartridge_benchmark` and `cartridge_composition_sweep`, and
only a session declining to commit a run document kept one from asserting a
command its image cannot run.[^task]

## Every image says which commit it is, or is refused

The build writes the commit it was built from to `<env_path>/GIT_COMMIT`
inside the image. Read on 2026-10-16 from all nine registered images, every
one carries it: v23, the `mi` image, says `fab1d991`, and v56, the
`cartridge-qa` image, says `0e9f177f`. The stamp inside the image is the
bytes' own answer, so the check reads it there rather than a copy of it kept
in the registry, which would be one more record to drift.[^stamps]

Preflight reads it for every image run, in the same round trip as the module
check, and an image whose stamp cannot be read, or reads empty, is refused
`ENTRYPOINT_IMAGE_UNSTAMPED` with the exit status and the text the read gave.
Without it nothing could say which code the run executes. A directory
environment is not asked: no build identifies it.[^unstamped]

## The environment is asked whether it can start the module, and from where

Preflight, and array preflight for a sweep, find every `python -m <module>`
the command runs and ask the environment's own interpreter, inside the image,
whether it can start each one. Resolution follows `python -m`: from the
directory the command enters with `cd` (cleargbm runs `scripts.optimize`
from a cluster checkout, in no wheel), else the job's script directory. Only
the top-level name goes through `importlib.util.find_spec`, which imports
nothing, and the rest is a file test, so a package whose `__init__` loads
torch costs the probe nothing. For a module it can start, the probe also
says where the top-level name resolved: under the interpreter's own prefix,
which is the code the image was built with, or outside it, which is the
command's own checkout.[^probe]

A module the environment cannot start is refused `ENTRYPOINT_NOT_INSTALLED`,
naming the image and the commit it was built from, before anything is
uploaded or queued. An answer the check cannot read is
`ENTRYPOINT_PROBE_UNREADABLE`, and a directory the command enters that does
not exist fails the probe's own `cd`.[^refusal]

## A module that moved on since its image is warned about and recorded

A module the image HAS is admitted, because reproducing an old command under
its old image must stay legal. But the code it runs is the image's, and when
that module's source has changed since the image's commit, the numbers the
run produces come from older code than the person reading them is likely to
assume. So its top-level package's source is found in the project's `repo`,
at a source root (a same-named module inside another package is not it), and
`git diff` between the image's commit and HEAD names what changed. A module
that resolved outside the environment, a package the repo has no source for,
and a run with no image are each recorded with the reason nothing was
compared, so an empty change list never reads as a comparison that was not
made. An image commit the repo does not hold is refused
`ENTRYPOINT_IMAGE_COMMIT_UNKNOWN`, and git failing once it does is
`ENTRYPOINT_DRIFT_UNREADABLE`.[^drift]

Measured through the real cluster on 2026-10-16:[^measured]

| run | image commit | what preflight printed |
| --- | --- | --- |
| `ka-probe-v23-v100`, `model_trainer.cli.known_answer_probe` | `fab1d991` (v23) | admitted, `WARNING`: 194 files under `services/Model-Trainer/src/model_trainer` changed since the image's commit |
| `cartridge-traits-gpt2-pilot-cpu-v56`, `model_trainer.cli.cartridge_trait_sweep` | `0e9f177f` (v56) | admitted, `WARNING`: 25 files changed |
| `p6-rung1`, `scripts.optimize` | `80221ea1` (cleargbm) | admitted, not compared: resolved from `/pub/wagnera3/api/services/covenant-radar-api/scripts`, outside the image's environment |

The first run of this comparison also counted
`libs/platform_core/src/platform_core/config/model_trainer.py` as part of
`model_trainer`, 195 files rather than 194, which is why the source must sit
at a root.[^firstrun]

Every submission records the result in its ledger row's `image_drift`, per
array member for the modules that member starts, `hpc3-submit`, `hpc3-sweep`,
`hpc3-campaign` and `hpc3-chain` print it under what they submitted, and
`hpc3-trace` prints it beside the image's digest. The field is required on
every row: the hub's 16,064 earlier rows were backfilled to explicit null,
which `hpc3-trace` renders as "code unrecorded", never as unchanged.[^ledger]

[^task]: Board task 2c8161a3's spec (see `provenance`); `src/hpc3/core/entrypoint.py` § module docstring, "THE HOLE".
[^stamps]: `apptainer exec` reads of `/opt/env/GIT_COMMIT` on login-i17, 2026-10-16 (see `provenance`); `src/hpc3/core/entrypoint.py` § `probe_command`, which reads the same file.
[^unstamped]: `src/hpc3/core/entrypoint.py` § `probe_command` (the stamp step, only when `read_stamp`) and `check_entrypoints`, which raises `ENTRYPOINT_IMAGE_UNSTAMPED`; `tests/test_entrypoint.py` § `TestTheRegisteredMiImage`, the absent and the empty stamp.
[^probe]: `src/hpc3/core/entrypoint.py` § `invocations`, `_PROBE_SOURCE` and `check_entrypoints`; `src/hpc3/core/preflight.py` § `preflight`; `src/hpc3/core/array_submit.py` § `array_preflight`; `tests/test_entrypoint.py` § `TestTheProbeRunsForReal`, where an installed package resolves inside the interpreter's prefix and a checkout's copy outside it.
[^refusal]: `src/hpc3/core/entrypoint.py` § `parse_probe` and `check_entrypoints`; `tests/test_entrypoint.py` § `TestReadingTheAnswer` and `TestBothSubmissionPathsAskBeforeTheScheduler`; `README.md`, the paragraph after the `repo_commit` one.
[^drift]: `src/hpc3/core/drift.py` § module docstring, `source_paths` and `measure_drift`; `src/hpc3/contracts/drift.py` § `ModuleDrift` and `ImageDrift`; `tests/test_drift.py` § `TestMeasuringAgainstRealHistory`, against a real two-commit repository, and `TestWhereAPackageLives`.
[^measured]: The three `hpc3-preflight` runs, finished at 2026-10-16T17:55:18Z (v23), 17:56:05Z (v56) and 17:58:09Z (cleargbm), from ziggurat (see `provenance`).
[^firstrun]: The `hpc3-preflight` run finished at 2026-10-16T17:53:58Z, before the rule (see `provenance`); `src/hpc3/core/drift.py` § `source_paths`, "ONLY AT A SOURCE ROOT"; `tests/test_drift.py` § `TestWhereAPackageLives` and the real repository's `libs/core/model_trainer.py`.
[^ledger]: `src/hpc3/contracts/ledger.py` § `LedgerEntry`'s `image_drift` attribute and `_require_drift_or_null`; `src/hpc3/core/array_submit.py` § `submit_array`, which narrows the drift per member; `src/hpc3/cli/trace.py` § `main`; the hub ledger's backfill (see `provenance`).
