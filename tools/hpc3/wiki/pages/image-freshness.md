---
title: An image older than its command is refused at preflight
tags: [images, preflight, provenance]
hubs: [images-and-staging]
related: ["[[image-build-flow]]", "[[submission-rules]]", "[[spec-symbol-drift]]", "[[known-answers]]"]
source_paths:
  - "src/hpc3/core/entrypoint.py"
  - "src/hpc3/core/preflight.py"
  - "src/hpc3/core/array_submit.py"
  - "tests/test_entrypoint.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/core/entrypoint.py": "f6646a5d62823d3eff828085a5af7d77ff166ffd"
  "src/hpc3/core/preflight.py": "ed718a167420cb85ac203f43872b023c0c172765"
  "src/hpc3/core/array_submit.py": "5391cb2c3b8505f23ed25fc0b91fdff332c06729"
  "tests/test_entrypoint.py": "ffd6c5b1eacfe839659861e9f98f548b517b4b89"
  "README.md": "c3962b94b036071c6e52e96babff021a11f1b5d8"
provenance:
  - "board task 2c8161a3 (MCPs board), filed 2026-09-04: the registered mi image predates cartridge_benchmark and cartridge_composition_sweep"
  - "apptainer exec <image> cat /opt/env/GIT_COMMIT on login-i17, 2026-10-16, for all nine registered images"
  - "the entrypoint probe run against /pub/wagnera3/images/v23/abl.sif on login-i17, 2026-10-16"
  - "Hpc3ErrorCode.ENTRYPOINT_NOT_INSTALLED and ENTRYPOINT_PROBE_UNREADABLE, libs/platform_core/src/platform_core/error_codes_tooling.py, outside this wiki's workspaceRoot"
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

## Every image already says which commit it is

The build writes the commit it was built from to `<env_path>/GIT_COMMIT`
inside the image. Read on 2026-10-16 from all nine registered images, every
one carries it: v23, the `mi` image, says `fab1d991`, and v56, the
`cartridge-qa` image, says `0e9f177f`. The stamp inside the image is the
bytes' own answer, so the check reads it there rather than a copy of it kept
in the registry, which would be one more record to drift.[^stamps]

## The environment is asked whether it can start the module

Preflight, and array preflight for a sweep, find every `python -m <module>`
the command runs and ask the environment's own interpreter, inside the image,
whether it can start each one. Resolution follows `python -m`: from the
directory the command enters with `cd` (cleargbm runs `scripts.optimize`
from a cluster checkout, in no wheel), else the job's script directory. Only
the top-level name goes through `importlib.util.find_spec`, which imports
nothing, and the rest is a file test, so a package whose `__init__` loads
torch costs the probe nothing.[^probe]

A module the environment cannot start is refused `ENTRYPOINT_NOT_INSTALLED`,
naming the image and the commit it was built from, before anything is
uploaded or queued. An answer the check cannot read is
`ENTRYPOINT_PROBE_UNREADABLE`, and a directory the command enters that does
not exist fails the probe's own `cd`. A command that starts no module, such
as `python train.py`, is examined as zero modules with no round trip.[^refusal]

## Old commands under old images stay legal

A module the image HAS passes even when its source has changed since the
image's commit, because that is what reproducing an old command under the
old image is. Measured against v23 on 2026-10-16:
`model_trainer.cli.cartridge_composition_sweep` was refused, the message
reading "built from fab1d991", and `model_trainer.cli.known_answer_probe`,
the command v23's known-answer probes ran, was admitted with that commit
reported.[^measured]

[^task]: Board task 2c8161a3's spec (see `provenance`); `src/hpc3/core/entrypoint.py` § module docstring, "THE HOLE".
[^stamps]: `apptainer exec` reads of `/opt/env/GIT_COMMIT` on login-i17, 2026-10-16 (see `provenance`); `src/hpc3/core/entrypoint.py` § `probe_command`, which reads the same file.
[^probe]: `src/hpc3/core/entrypoint.py` § `invocations`, `_PROBE_SOURCE` and `check_entrypoints`; `src/hpc3/core/preflight.py` § `preflight`; `src/hpc3/core/array_submit.py` § `array_preflight`.
[^refusal]: `src/hpc3/core/entrypoint.py` § `parse_probe` and `check_entrypoints`; `tests/test_entrypoint.py` § `TestTheProbeRunsForReal` and `TestBothSubmissionPathsAskBeforeTheScheduler`; `README.md`, the paragraph after the `repo_commit` one.
[^measured]: The probe run against v23 on login-i17 (see `provenance`); `tests/test_entrypoint.py` § `TestTheRegisteredMiImage`, which replays both answers.
