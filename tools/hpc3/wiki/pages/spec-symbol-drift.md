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
  "tests/test_committed_specs.py": "6de85126912111f818440366c667a96af7f78472"
  "specs/abl-image.json": "fa51c3bd9abe519a9ccc54fdc38e5e532b0f0cc5"
  "src/hpc3/contracts/image_spec.py": "26354bdfe7ae4f19d4bac5fa6936f851010f93e2"
provenance:
  - "services/Model-Trainer/src/model_trainer/cli/_measurement_hooks.py -- the module `probed_shapes_hook` moved to, added in commit 5bea978c and outside this wiki's workspaceRoot"
  - "jobs 55736151, 55736405 and 55736689, mi.image-v32 on the free partition, 2026-09-03 -- the build this was found while preparing; the first two were preempted and cancelled"
fact_checked: 2026-09-03
confidence: high
---

# An image spec cites the repository, and a rename invalidates the citation twenty-five minutes later

An image spec's `required_symbols` and `smoke_commands` name **Python module
paths in this monorepo**. The spec lives in `tools/hpc3/specs/`; the modules
live in five other packages; between them is a wheel build and a container.
Nothing re-checked the citation, so a correct refactor could leave a spec that
was already broken and looked fine[^1].

## What happened

`model_trainer.cli._test_hooks` passed the 600-line ceiling and gave up its
measurement tables — the ladder, the gemm shape sets, the cost sweeps — to a
sibling `_measurement_hooks`. `probed_shapes_hook` moved with them, which is
correct: the module's own docstring says it holds "the seams that need real
weights and a real GPU", and a table of matrix shapes is not that[^2].

Five places in `specs/abl-image.json` still named the old home: one
`required_symbols` entry and four `smoke_commands`. Every one of them runs
**inside the built image**, in `%post`, after pip has installed eighty-three
pinned requirements and five wheels. The build takes about twenty-five
minutes, and the self-check is the last thing in it[^3].

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
emitted empty unconditionally; both are hand-maintained after capture[^4].

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
that does not have the image's dependencies at all[^5].

**The attribute half is the load-bearing half.** A module-only check would
have PASSED here: `model_trainer.cli._test_hooks` still exists, and only the
attribute left it. The suite carries that exact case as a negative control
rather than as a story, alongside its counterpart — that the symbol the spec
names today does resolve[^6].

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

## What the derived rule then found on its own

The transcribed list it replaced was forty-six pairs, and its own docstring
recorded fourteen recurrences of going stale plus `make check` red on `main`
across nine consecutive commits. That list said the generator was
unreachable, "because this package cannot: `hpc3` is its own poetry project
and `import model_trainer` fails in its venv." The premise was wrong in one
word: the check does not have to IMPORT[^7].

Once it was derived rather than transcribed, a second invariant became
cheap — **every first-party wheel an image installs must be named by at least
one required symbol**, or a stale one of them cannot be caught. That found
twelve gaps nobody had looked for[^8]:

| spec | wheels nothing named |
|---|---|
| `abl` | `platform_workers` |
| `cleargbm` | `covenant_domain`, `covenant_nn`, `covenant_persistence`, `covenant_radar_api`, `platform_core`, `platform_workers` |
| `rusted` | `platform_core`, `hpc3` |
| `tankpit` | `monorepo_guards`, `platform_core` |

All closed. Where the image's already-asserted code imported something from
the missing package, that import was the choice; where it did not, the
package's principal export was.

**What this invariant does NOT claim.** A symbol only detects a stale wheel
if it names something newer than the last build. Naming `get_logger` proves
a wheel was installed, not that it is current. The rule closes "nothing names
this wheel at all", which is a different and weaker guarantee, and the
difference is worth stating rather than blurring.

The pyproject read goes through `config_test_hooks.tomllib_loads`, because
the env guard bans `import tomllib` outright. The guard has an allow-list;
reaching for it when a supported seam already exists would have been an
exemption rather than a fix[^9].

## Application note

A rename that crosses a module boundary is a change to the image recipe,
whether or not the author knows a recipe exists. Run `make check` in
`tools/hpc3` after one — the check is under two seconds — rather than
discovering it from a `%post` failure at the end of a build[^10].

[^1]: `specs/abl-image.json` fields `required_symbols` and `smoke_commands`, against `src/hpc3/contracts/image_spec.py` section `ImageSpec`, which types them as free strings.
[^2]: `services/Model-Trainer/src/model_trainer/cli/_measurement_hooks.py` module docstring, and `_test_hooks.py`'s, in commit 5bea978c (see `provenance`).
[^3]: `specs/abl-image.json` `requirements` and `wheels`, and job 55736689's elapsed time (see `provenance`).
[^4]: `src/hpc3/cli/image_capture.py` section `capture_spec`, at the `required_symbols` and `smoke_commands` assignments.
[^5]: `tests/test_committed_specs.py` sections `_module_files` and `_top_level_names`.
[^6]: `tests/test_committed_specs.py` sections `test_an_attribute_that_moved_out_of_a_surviving_module_is_caught` and `test_the_attribute_that_moved_resolves_in_its_new_home`.
[^7]: `tests/test_committed_image_spec.py` class docstring, which quotes the removed list's own reasoning.
[^8]: `tests/test_committed_specs.py` section `test_every_first_party_wheel_is_asserted_by_at_least_one_symbol`, run against the four specs before they were amended.
[^9]: `tests/test_committed_specs.py` section `_wheel_packages`; the ban is `libs/monorepo_guards`' env rule, outside this wiki's workspaceRoot.
[^10]: `tests/test_committed_specs.py` -- the suite runs in under three seconds, measured 2026-09-03.
