---
title: The image build flow, and why a GPU project must declare an image
tags: [images, identity, gpu]
hubs: [images-and-staging]
related: ["[[image-ledger-lessons]]", "[[environment-pins]]", "[[known-answers]]"]
source_paths:
  - "src/hpc3/contracts/image_spec.py"
  - "src/hpc3/cli/image.py"
  - "src/hpc3/cli/image_build.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/image_spec.py": "ef0e7c294cd0c8cca9e515605e0be486a6d77f78"
  "src/hpc3/cli/image.py": "6961a7e21a8bf6427f7191e07c4c91e784c5a7ad"
  "src/hpc3/cli/image_build.py": "31e8cc8c5ff389857c000e60e31d11cf8534eb90"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
fact_checked: 2026-09-01
confidence: high
---

# The image build flow, and why a GPU project must declare an image

**A project that requests a GPU must declare an `image`.** This is enforced
when the workspace is decoded, so a GPU project cannot be registered without
one, and a run may override the image but may not set it to `null`.

The reason is not tidiness. A GPU run's numbers are decided by the whole stack
above the card — CUDA runtime, cuDNN, the torch build, every library that
touches a tensor. An image pins all of it and gives the run a **content
digest**, which is the `image_digest` axis of the run fingerprint. A directory
environment pins nothing and has no digest, and it can be edited in place
while `pinned_packages` is edited to match — which is exactly what happened on
2026-08-28, in under an hour, with every check still passing.

CPU-only projects may omit it. `cleargbm` has no card and no driver stack; its
arithmetic is pinned by BLAS thread count instead.

## Four commands, in order

1. **`hpc3-image-capture`** reads the project's live environment and writes
   the spec. Do NOT hand-write one: pip-freeze-and-paste is unrepeatable and
   silently incomplete, which is how the first spec got made.
2. **`hpc3-image`** renders the spec into definition, requirements,
   self-check and build script. Pure; builds nothing. `--project` and
   `--name` COMPOSE the job name — there is no `--job-name`, so a name whose
   project half no workspace declares cannot be rendered
   ([[image-ledger-lessons]]).
3. **`scp` the rendered files AND the first-party wheels into the IMAGE
   directory** — `build.sbatch` does `cd <image-dir>` and runs `build.sh`
   there. Putting them anywhere else fails with
   `build.sh: No such file or directory`.
4. **`hpc3-image-build`** preflights, submits the rendered build, and writes
   the ledger row — like every other job. ~25 minutes on `free`, which is
   `PreemptMode=CANCEL`, so the rendered `--requeue` is inert there and a
   preempted build is simply gone; re-run the command, it is idempotent apart
   from the job id.

Then put the built image's path and `sha256` in the project's `image`, set
`env_path` to the in-image prefix (`/opt/env`), and preflight: the pinned
packages are then checked **inside the image** rather than against a
directory someone can edit.

## CPU-only is a single spelling

`"gpu": null` is how a CPU-only job is stated, and it is the only way — there
is no zero-count request, because two spellings of one state is how they
drift apart. The partition must agree in both directions
([[submission-rules]]).
