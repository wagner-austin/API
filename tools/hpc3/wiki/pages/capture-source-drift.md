---
title: A capture source drifts from the image it produced, and re-capturing silently ships the drift
tags: [identity, environments, images]
hubs: [images-and-staging]
related: ["[[environment-pins]]", "[[image-build-flow]]", "[[image-ledger-lessons]]"]
source_paths:
  - "src/hpc3/cli/image_capture.py"
  - "src/hpc3/core/image_capture.py"
  - "specs/abl-image.json"
source_git_blobs:
  "src/hpc3/cli/image_capture.py": "00ac8efad2d702a41d1e7a5bf3cc8d004de47991"
  "src/hpc3/core/image_capture.py": "05e0a12d2c07600a9485563a52c0a78005e0f106"
  "specs/abl-image.json": "923b4e8b0b9dba0e5f09f81976ec6549ced5489d"
provenance:
  - "cluster environment /pub/wagnera3/envs/abl-pinned (not in this repo)"
fact_checked: 2026-09-01
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
environment at all. `probe_command(config["env_path"])` builds the probe, and
when `config["image"]` is not None the probe is wrapped in
`run_inside_image` (`cli/image_capture.py` L 155-158). `env_path` is
`/opt/env` for an imaged project, which exists only inside the `.sif`.

That means the drift above is not reachable by re-running capture as-is. To
capture an environment carrying a new dependency, the config handed to
capture has to point at the host env with no image declared, which is the
onboarding shape, not the steady-state one.

## Three fields capture cannot carry forward

`system_packages` and `smoke_commands` are emitted empty unconditionally
(`cli/image_capture.py` L 176, L 190), and `required_symbols` comes only from
the `--symbols` flag (L 184). Both empties are deliberate and documented in
the source: capture reads what an environment CONTAINS, while an OS package
list and a smoke command state what the image must be able to DO, which
cannot be probed off a package listing.

For the abl spec as committed that is 29 smoke commands and 46 required
symbols. A re-capture that does not re-supply them produces a spec that
builds an image asserting nothing about itself.

## What the source says the practice already is

The comment above the probe records it plainly: the command "was written for
the onboarding case and worked exactly once per project; every version bump
since has hand-edited `git_commit` in the generated spec instead"
(`cli/image_capture.py` L 150-154).

So the surgical edit is not a shortcut around the tool. It is what every
version bump has actually done, and for a single added dependency it is also
the only edit that preserves cupy, the smoke commands and the symbol
assertions in one step.

## Application notes

Before re-capturing an imaged project, diff the intended capture source
against the spec the current image was built from. A capture that adds one
package and removes another is indistinguishable, in the emitted spec, from
a capture that adds one package.

The reverse check is cheap and worth doing after any build: read the sealed
image with `apptainer exec` and confirm the distributions the spec claims are
present. The image is the artifact runs cite; the env is a staging area
nobody pins.
