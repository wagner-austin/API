---
title: An image spec cites the repository, and a rename invalidates the citation twenty-five minutes later
tags: [images, identity, provenance, refactoring]
hubs: [images-and-staging]
related: ["[[capture-source-drift]]", "[[image-build-flow]]", "[[image-ledger-lessons]]", "[[known-answers]]"]
source_paths:
  - "tests/test_committed_specs.py"
  - "specs/abl-image.json"
  - "src/hpc3/contracts/image_spec.py"
source_git_blobs:
  "tests/test_committed_specs.py": "14558a4bc10fbd6de87f13cda123f97b8c135046"
  "specs/abl-image.json": "d77aef35d445ae03105934250b7a1823031e63df"
  "src/hpc3/contracts/image_spec.py": "26354bdfe7ae4f19d4bac5fa6936f851010f93e2"
provenance:
  - "services/Model-Trainer/src/model_trainer/cli/_measurement_hooks.py -- the module `probed_shapes_hook` moved to, added in commit 5bea978c and outside this wiki's workspaceRoot"
  - "jobs 55736151 and 55736405, mi.image-v32 on the free partition, 2026-09-03 -- the build this was found while preparing"
fact_checked: 2026-09-03
confidence: high
---

# An image spec cites the repository, and a rename invalidates the citation twenty-five minutes later

An image spec's `required_symbols` and `smoke_commands` name **Python module
paths in this monorepo**. The spec lives in `tools/hpc3/specs/`; the modules
live in five other packages; between them is a wheel build and a container.
Nothing re-checked the citation, so a correct refactor could leave a spec that
was already broken and looked fine.

## What happened

`model_trainer.cli._test_hooks` passed the 600-line ceiling and gave up its
measurement tables — the ladder, the gemm shape sets, the cost sweeps — to a
sibling `_measurement_hooks`. `probed_shapes_hook` moved with them, which is
correct: the module's own docstring says it holds "the seams that need real
weights and a real GPU", and a table of matrix shapes is not that.

Five places in `specs/abl-image.json` still named the old home: one
`required_symbols` entry and four `smoke_commands`. Every one of them runs
**inside the built image**, in `%post`, after pip has installed eighty-three
pinned requirements and five wheels. The build takes about twenty-five
minutes, and the self-check is the last thing in it.

So the failure mode was: refactor correctly, rebuild, wait, and learn at the
end that the recipe was stale. Nothing earlier in the chain objects — the
spec decodes, the renderer renders, `sbatch` accepts, apt and pip succeed.

## Why the existing checks do not cover it

Three things look like they should and do not.

**The image's own self-check** is the right check in the wrong place. It runs
the assertions, which is exactly what should happen, and it runs them at the
far end of the expensive part. `known-answers` makes the same distinction for
a different failure: an image that still builds is not an image that still
computes.

**`capture-source-drift`** covers the other direction — an environment that
moves away from the spec built from it. This is the repository moving away
from it. Capture cannot help here either way, because
`required_symbols` comes only from `--symbols` and `smoke_commands` is
emitted empty unconditionally; both are hand-maintained after capture.

**The wiki's own `source_git_blobs`** is the mechanism that solves this
problem, for wiki pages. A path that still resolves proves nothing across a
rewrite; a blob pin catches drift. A spec has no such field, and adding one
would pin file contents when what is being cited is a *name*.

## The check that exists now

`tests/test_committed_specs.py` maps every module this monorepo provides —
by walking `*/*/src` — and refuses any spec that names a symbol which no
longer exists. It parses rather than imports, because importing
`model_trainer.core.services` to answer a question about a name pulls torch
into the test process, and a spec is most usefully checked in an environment
that does not have the image's dependencies at all.

**The attribute half is the load-bearing half.** A module-only check would
have PASSED here: `model_trainer.cli._test_hooks` still exists, and only the
attribute left it. The suite carries that exact case as a negative control
rather than as a story, alongside its counterpart — that the symbol the spec
names today does resolve.

Two things it deliberately does not do:

- **`smoke_commands` stay unchecked.** They are arbitrary Python one-liners
  carrying imports, asserts and digest comparisons. Static analysis of them
  would be a parser nobody trusts or a regex that passes on the cases it was
  not written for. The image's self-check remains the thing that runs them.
- **A module whose root package this repository does not provide is
  skipped**, and that is a fact rather than an exemption. `turkic-lstm` lives
  in `~/PROJECTS/LSTM`, so `char_lstm` is not here to check. What stops that
  becoming a hole is a count assertion: a rule that silently scans nothing
  passes forever.

## Application note

A rename that crosses a module boundary is a change to the image recipe,
whether or not the author knows a recipe exists. Run `make check` in
`tools/hpc3` after one — the check is under two seconds — rather than
discovering it from a `%post` failure at the end of a build.
