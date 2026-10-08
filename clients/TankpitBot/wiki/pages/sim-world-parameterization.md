---
title: Sim World Parameterization
tags: [sim, architecture, decisions, hpc, determinism, measurement]
related:
  - "[[feature-corpus-provenance]]"
  - "[[coding-standards]]"
  - "[[game-economy]]"
source_paths:
  - "src/tankpit_bot/sim/world_seed.py"
  - "src/tankpit_bot/sim/run_boot.py"
  - "src/tankpit_bot/sim/run.py"
  - "src/tankpit_bot/sim/scenarios.py"
  - "src/tankpit_bot/sim/cli_args.py"
  - "src/tankpit_bot/runtime_artifacts.py"
  - "src/tankpit_bot/sim/field_run.py"
source_git_blobs:
  "src/tankpit_bot/sim/world_seed.py": "29197885d73c089b101f28a6d4088281d6fe7193"
  "src/tankpit_bot/sim/run_boot.py": "5b53afaf54c67be123391762dd756a746e8ddea4"
  "src/tankpit_bot/sim/run.py": "e0686e2c6ef53ee4aa0849eed6ca0f4a48317a6d"
  "src/tankpit_bot/sim/scenarios.py": "cc1bb400bcc8082a869b0f7726bc810b75ebfeca"
  "src/tankpit_bot/sim/cli_args.py": "179ec5e69c3d9ea92b2623f5cac78b2f1ea9565b"
  "src/tankpit_bot/runtime_artifacts.py": "062e2e5984680df96a1d2e17e8897bd9899c6238"
  "src/tankpit_bot/sim/field_run.py": "1bdf096f5e4c32ea8ba64535178a6e5caea06340"
provenance:
  - "Board task b008ab91, note 2026-09-03T01:36Z — the retracted saturation table that found the layout half"
  - "Board task b008ab91, note 2026-09-03T03:10Z — the population-seed half and the two-blockers-are-one finding"
fact_checked: "2026-09-02"
verified: 2026-10-08 (run_boot.py, run.py, cli_args.py and runtime_artifacts.py re-read against API HEAD for MCPs board task 3d71a8e1; the world resolution's move into run_boot.resolve_named_world and the HPC3 image added, the stamp-derived artifact paths and the sweep gate unchanged)
confidence: high
hubs: [architecture]
---

# Sim world parameterization: the stamp is a label, not an input

*Established 2026-09-02, as the precondition for running sim sweeps on
HPC3.*

## The defect

A run's stamp is its NAME — it dates the artifacts. It was also, silently,
an INPUT to what the run played. Three sites in `_boot` derived world state
from it:[^1]

| site | what the stamp decided |
|---|---|
| `select_practice_layout(stamp)` | which of 3 practice layouts — 36 bot spawns and the client spawn |
| `seed_field_population(seed=crc32(stamp))` | where **every container** lies |
| `select_practice_layout(stamp)` (atlas-forage) | the client spawn again |

So naming a run changed what it played. Locally that is a feature: successive
soaks see different rooms for free. In a measurement it is a confound, and it
cost a published result — a saturation table varied session depth and the
stamp together, so the layout moved with the variable, and the conclusion was
retracted.[^2]

**The container seed was the more dangerous half, and it was found second.**
The layout at least prints itself in a log line. The population seed prints
nowhere, so a stamp-varied larder moves any forage or economy number with
nothing in any artifact recording that it did.[^3]

## The two cluster blockers are one defect

They look independent and are not:

- stamp **varies** per array task → worlds vary by accident
- stamp **fixed** across tasks → artifact paths collide, because
  `archive_log_path`, `archive_events_path` and `sim-<stamp>.world.json` are
  all stamp-derived

There is no stamp policy that satisfies both. The stamp was doing double duty
as a label and as a world input; separating those fixes both at once.

This is not theoretical: the test pinning "two named layouts differ" first
failed because both sessions shared a stamp, wrote the same
`sim-<stamp>.world.json`, and the assertion compared a file with itself.[^4]

## The shape

**`_boot` stops deriving and starts taking.** `layout` and `population_seed`
are required keyword arguments — required rather than defaulted, because a
default is what let the stamp-derived value hide.

**The derivation moved up to the CLI, where it is visible.** `run_sim_session`
resolves an explicit `--layout` / `--population-seed` if given, and otherwise
falls back to the stamp exactly as before — so **interactive behaviour is
unchanged and existing soaks reproduce**. A sweep member names both, and the
stamp becomes inert. The resolved world is now logged either way, saying for
each value whether it was `named` or `derived from stamp`. Since the
multiplayer track (board task b008ab91, 2026-10-05) that resolution is one
function, `resolve_named_world` in `src/tankpit_bot/sim/run_boot.py:89`,
called by both `run_sim_session` (`run.py:175`) and the multi-client
`run_field_session` (`field_run.py:340`), so a field session names or
derives its world exactly as a one-bot session does.[^8]

**Naming is refused, not defaulted.** `layout_by_provenance` raises
`UnknownPracticeLayoutError` on an unknown name. A sweep that silently played
a different world than its member document names would produce numbers nobody
could interpret.

**`--runs-root` relocates the probe artifacts** off the fixed `runs/probe`,
which N array tasks on one node would otherwise share. `--out` already
existed for the capture and world archives.

The flag parsing moved out of `scenarios.py` into `cli_args.py` on the way:
that module had reached 593 lines against a hard 600-line ceiling, so the gate
went into a new module rather than being squeezed under the bar.

## What a sweep member must do

State the world, and give the task its own roots:

```
tankpit-sim-run --practice --rounds N \
  --layout <provenance> --population-seed <int> \
  --runs-root runs/task-$SLURM_ARRAY_TASK_ID \
  --out       runs/sim/task-$SLURM_ARRAY_TASK_ID
```

Omitting either flag is **refused** wherever the numbers will be compared.
`require_named_world` raises `UnnamedWorldError` naming the missing flags, on
either of two conditions:

- **`--sweep` was passed**, which DECLARES the intent to compare.
- **`SLURM_ARRAY_TASK_ID` is set**, which BETRAYS it. This is the arm that
  matters. The failure being guarded is a *forgotten* flag, and a gate you
  must remember to arm does not guard against forgetting — so the refusal
  fires on the cluster whether or not the member document remembered
  `--sweep`. It keys off the same variable `hpc3`'s own array contract reads,
  so it triggers on the real environment rather than a spelling of it
  invented here.[^6]

A half-named world is refused too: naming the layout and forgetting the seed
still moves the larder, and that is the half that prints nowhere.

Interactive runs are untouched — no flags, no array variable, no gate. A
set-but-empty `SLURM_ARRAY_TASK_ID` does not arm it either, since a shell that
exports the name without a value is not an array task, and gating every local
run is how a guard earns its way out of the tree.

## Status

**Registered 2026-09-02** as the `tankpit` project
(`tools/hpc3/runs/hpc3-tankpit.json`), with a section in `docs/RESEARCH.md`
as the enforcement requires. Free partition, no GPU, no charge account —
`hpc3` refuses a billing partition outright, so free is not a setting anyone
chose.

**Since 2026-09-03 the project runs from an image.** `b86990217`
(2026-09-02, repo API, "tankpit ships an image: built, self-checked, and
preflight passes") built the first one, and `30a0b65ea` (2026-09-03)
adopted v2, built from the self-contained wheel: `hpc3-tankpit.json` now
names `/pub/wagnera3/tankpit/images/v2/tankpit.sif` with its `sha256`,
binds `/pub/wagnera3`, and sets `env_path` to `/opt/env` inside the
image.[^7] The rest of this section is the state on the day of
registration, kept because it records why an image was the answer.

**Registered was not runnable, and preflight said so.** The refusal was then
`ENV_PATH_MISSING` rather than `WORKSPACE_PROJECT_UNKNOWN`: the workspace
resolves, the project is found, the partition and budget and preemption
clauses pass, the run's experiment identity validates, and it stops at
`/pub/wagnera3/envs/tankpit` not existing.[^5] The monorepo IS staged at
`/pub/wagnera3/api` (behind this tree), but the cluster's system Python is
3.9 where this package needs 3.11. The remaining work was provisioning.

**It needs an image, and the first registration said otherwise — that was
wrong.** The reason given was that four of five projects use a directory
environment and this payload is "pure Python". That is a popularity argument,
and it ignored the closest analogue: `rusted` is also CPU-only on `free` with
`requeue` and `deterministic`, and it carries an image.

An image answers both blockers a directory environment leaves open. It
bundles its own interpreter, so 3.9-vs-3.11 stops mattering; and it bakes
first-party code as wheels with `git_commit` recorded, so the payload is not
read from the mutable `/pub/wagnera3/api` checkout that was then behind
this tree. `image_digest` then becomes a real fingerprint axis instead of
`NO_VALUE`. The directory-environment hazard is documented, not theoretical:
one "can be edited in place while `pinned_packages` is edited to match --
which is exactly what happened on 2026-08-28, in under an hour, with every
check still passing."

On 2026-09-02 the image was **not yet built**, so the project declared `image: null` as a stated
gap rather than a decision — `ImageSpec` requires a real `sha256` and a
placeholder would be a fabricated digest. Building it faces a chicken-and-egg
the four-command flow does not cover for a new project:
`hpc3-image-capture` probes an existing environment **over SSH** at
`env_path`, so a project with no environment has nothing to capture. A
bootstrap environment must exist first, and only then can the image that
replaces it be captured and built.

[^1]: `src/tankpit_bot/sim/run_boot.py` before 2026-09-02 — lines 207, 221 and 238 of that revision.
[^2]: Board task `b008ab91`, note 2026-09-03T01:36Z: "I varied session depth and the run STAMP together. `select_practice_layout` returns `PRACTICE_LAYOUTS[crc32(stamp) % len(PRACTICE_LAYOUTS)]`."
[^3]: `src/tankpit_bot/sim/world_seed.py`, `population_seed_for_stamp` docstring; the seed reaches `seed_field_population`, which lays the dotted/hidden container field.
[^4]: `tests/sim/test_world_parameterization.py::test_two_named_layouts_differ_under_one_stamp` — the docstring records the first-run failure and why the separate archive directories are the point rather than test hygiene.
[^5]: `hpc3-preflight --config tools/hpc3/runs/hpc3-tankpit.json --run <a sim-session run doc>`, run 2026-09-02: `ENV_PATH_MISSING: /pub/wagnera3/envs/tankpit/bin does not exist on hpc3.`
[^8]: `src/tankpit_bot/sim/run_boot.py:89-125`, `resolve_named_world(stamp, layout, population_seed)`, whose docstring opens "The stamp -> world derivation lives HERE, in the open, rather than inside the boot where it used to hide"; `_boot` at `:380` still takes `layout` and `population_seed` as keyword arguments (`:385-386`). The block moved out of `run.py` in `815b645c3` (2026-10-05, repo API), the commit that added `field_run.py`.
[^7]: `tools/hpc3/runs/hpc3-tankpit.json` (repo API, outside this wiki's workspace), the `tankpit` project's `image` block and `env_path`, read 2026-10-08; its history is `d02078239` (registration), `b86990217`, `30a0b65ea`, `881ee3299` and `1106e49af`.
[^6]: `src/tankpit_bot/sim/cli_args.py:28` `ARRAY_TASK_ENV_VAR` and `:230` `require_named_world`; `tools/hpc3/src/hpc3/` reads the same variable in its array and payload contracts.
