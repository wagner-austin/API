---
title: Twenty-one unledgered image builds, and how the path that made them was closed
tags: [images, ledger, incidents]
related: ["[[image-build-flow]]", "[[triage-conditions]]", "[[ledger-closures]]"]
source_paths:
  - "src/hpc3/cli/image_build.py"
  - "src/hpc3/core/image_build.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/cli/image_build.py": "31e8cc8c5ff389857c000e60e31d11cf8534eb90"
  "src/hpc3/core/image_build.py": "2b93eab7d3f31f62a523d8078b8ffc4c440d90a8"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
provenance:
  - "image ebb61ed0 (the 23rd unledgered build)"
fact_checked: 2026-09-01
confidence: high
---

# Twenty-one unledgered image builds, and how the path that made them was closed

**Image builds were submitted by raw `ssh hpc3 'cd … && sbatch build.sbatch'`
until 2026-08-28**, and that is why twenty-one builds hold no ledger row:
`hpc3-trace` cannot say which job built an image, `hpc3-watch` was never given
the id, and `hpc3-triage` reports the build as `unclaimed` for the two hours
it runs — correctly, because from this machine's records it is a stranger
holding eight cores. The reverse-direction triage check found the
twenty-second on its first run, and `hpc3-image-build` is the answer to it:
the finding was closed by closing the path that produced it.

## The twenty-third, and why --job-name died

A build ran the old way on the same day the command landed. The build of image
`ebb61ed0…` was submitted by raw `sbatch` hours after `hpc3-image-build`
existed, so it holds no ledger row either. The cause was the render step, not
the submit step: it had been rendered with `--job-name img.abl-sif-v22`, a
name whose project half is `img`, which no workspace declares — and
`hpc3-image-build` **refuses** exactly that, so the malformed name pushed its
author onto the path that records nothing. A refusal at submit time is one
step too late when the thing it refuses is what the previous step invited you
to write. That is why `--job-name` is gone: the renderer composes the name
from a declared `--project`, the submitter derives it by the same rule, and
the two cannot disagree. A build that would break the ledger cannot be
rendered.

## Why --project and --name are stated, not read from the script

The ledger's name is the qualified `<project>.<name>` and **the project half
is the part a script cannot tell you**: v22 was rendered `img.abl-sif-v22`,
which reads as a project called `img` that no workspace declares. The
script's own `#SBATCH -J` is then required to match what will be recorded —
so the ledger row and `squeue` cannot disagree — and the refusal prints the
`hpc3-image --job-name` that would fix it. The partition is read from the
script rather than assumed from `BUILD_PARTITION`: that constant says what
this package renders *today*, and the file on the cluster is what actually
runs.

## What the build's row records

`deterministic: false` and an empty `image_digest` — a build is not a
numerical run, and it *produces* the image rather than running inside one —
and the `.sif` as its `artifact`, which is what lets `hpc3-trace --match`
answer "which job built this image".
