---
title: A capture source drifts from the image it produced, and re-capturing silently ships the drift
tags: [identity, environments, images]
hubs: [images-and-staging]
related: ["[[environment-pins]]", "[[image-build-flow]]", "[[image-ledger-lessons]]", "[[spec-symbol-drift]]"]
source_paths:
  - "src/hpc3/cli/image_capture.py"
  - "src/hpc3/core/image_capture.py"
  - "specs/abl-image.json"
source_git_blobs:
  "src/hpc3/cli/image_capture.py": "af0766bc1d4da403e2e566d6f0540bc3e1766f40"
  "src/hpc3/core/image_capture.py": "dc635e24b966209acc57e44e0f130c55015a9417"
  "specs/abl-image.json": "fa51c3bd9abe519a9ccc54fdc38e5e532b0f0cc5"
provenance:
  - "cluster environment /pub/wagnera3/envs/abl-pinned (not in this repo)"
fact_checked: 2026-09-03
confidence: high
---

# A capture source drifts from the image it produced, and re-capturing silently ships the drift

`hpc3-image-capture` mirrors an environment. That is its whole value on the
onboarding path and its whole hazard afterwards, because the environment it
mirrors keeps changing after the image is sealed and the image does not.

Measured 2026-09-01, comparing `/pub/wagnera3/envs/abl-pinned` against
`specs/abl-image.json`, the spec v31 was built from:

```
env distributions                 89
spec requirements                 82   (+ 5 first-party wheels)
in spec, ABSENT from the env      cupy-cuda12x
```

The image has it and the capture source does not. Verified directly inside
the sealed image:

```
$ apptainer exec /pub/wagnera3/images/v31/abl.sif /opt/env/bin/python \
    -c "import importlib.util as u; print(bool(u.find_spec('cupy')))"
True
```

So a re-capture off `abl-pinned` today would emit a spec without cupy, and
the next image would drop the NVRTC path `ordered_kernels` compiles through.
Nothing in the build would object: the spec would be internally consistent,
the self-check would pass, and the loss would surface as an import error in
whatever job first needed it.

## The probe follows the image, not the host

Once a project declares an image the capture stops reading the host
environment at all. `probe_command(source["env_path"])` builds the probe, and
when `source["image"]` is not None it is wrapped in `run_inside_image`
(`cli/image_capture.py`, section `capture_spec`). `env_path` is `/opt/env`
for an imaged project, which exists only inside the `.sif`.

**This paragraph used to end "that means the drift above is not reachable by
re-running capture as-is", and it is no longer true.** `--env-path` was added
on 2026-09-03 to unblock onboarding, and it does exactly the thing this page
said was unreachable: it names a host directory to probe and skips the
registry entirely (`cli/image_capture.py`, section `_environment_to_probe`,
and its module docstring's "TWO ROUTES"). So the host env CAN be re-captured
now.

What has not changed is that the two routes answer different questions. The
registered route probes what the IMAGE contains; `--env-path` probes what a
host directory contains. A capture that switches routes to pick up one added
dependency also picks up everything the host env has drifted away from, which
is the failure this page opened with.

## Three fields capture cannot carry forward

`system_packages` and `smoke_commands` are emitted empty unconditionally, and
`required_symbols` comes only from the `--symbols` flag
(`cli/image_capture.py`, section `capture_spec`). Both empties are deliberate
and documented at the assignment: capture reads what an environment CONTAINS,
while an OS package list and a smoke command state what the image must be
able to DO, which cannot be probed off a package listing.

For the abl spec as committed that is 34 smoke commands and 58 required
symbols. A re-capture that does not re-supply them produces a spec that
builds an image asserting nothing about itself.

The symbols are no longer only hand-maintained, though: they are re-checked
against this monorepo's source by a test, so a rename cannot leave them
naming something that has moved ([[spec-symbol-drift]]).

## What the source says the practice already is

The comment above the probe records it plainly. The steady-state path "is the
version bump, which was always the recurring job and was being done by
hand-editing `git_commit` in the generated spec"
(`cli/image_capture.py`, section `capture_spec`).

So the surgical edit is not a shortcut around the tool. It is what every
version bump has actually done -- including the v32 build on 2026-09-03 --
and for a single added dependency it is also the only edit that preserves
cupy, the smoke commands and the symbol assertions in one step.

## Application notes

Before re-capturing an imaged project, diff the intended capture source
against the spec the current image was built from. A capture that adds one
package and removes another is indistinguishable, in the emitted spec, from
a capture that adds one package.

The reverse check is cheap and worth doing after any build: read the sealed
image with `apptainer exec` and confirm the distributions the spec claims are
present. The image is the artifact runs cite; the env is a staging area
nobody pins.
