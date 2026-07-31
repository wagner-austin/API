---
title: "Doctrine — A Gameplay Style as One File"
tags: [policy, doctrine, experiments, methodology]
related:
  - "[[policy-loop]]"
  - "[[policy-determinism]]"
  - "[[community-play-strategies]]"
source_paths:
  - "src/rw_bot/policy/doctrine.py"
  - "doctrines/default.doctrine"
  - "src/rw_bot/harness/sweep.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-29"
confidence: high
hubs: [bot-architecture]
---

# Doctrine — A Gameplay Style as One File

A doctrine is every gameplay knob in one required-field file: goals, worker
ceiling, wave mass, reserve, and one flag per behaviour (expansion, counter
tilt, interception, scouting, raiding).[^1] The entry point takes a doctrine
path; a sweep job line names one; trying a style is copying a file and
editing a line.[^2]

## What it replaced

The knobs used to be positional CLI slots — ten of them by the end, each new
question threading one more position through the entry point, the Makefile
and the sweep harness. The deeper style choices were not knobs at all:
spending priority was call order, unit identities were module constants, and
a composition experiment was a source edit, which is how an A/B stops being
an A/B ([[policy-loop]]).

## The one-field discipline

Two arms that differ in one field are an A/B; in two, an anecdote. The
shipped presets form chains pinned by test — `aa` → `aa-counter` →
`aa-counter-guard` → one arm per behaviour flag — so whatever moves between
adjacent arms moved because of that field and nothing else.[^3] Every field
is required: a doctrine file with a missing field is an error naming the
field, not a default quietly changing what the arm means.

## The interaction with measurement

A sweep batch is jobs × seeds over doctrines, and since runs do not reproduce
across batches ([[policy-determinism]]), **every batch carries its own
control arm** — a doctrine result is only comparable to a control that played
under the same conditions. The `default` preset is pinned byte-for-byte to
the in-code constant so the baseline cannot drift.[^3]

[^1]: `src/rw_bot/policy/doctrine.py` — the `Doctrine` TypedDict, `parse_doctrine_lines`, and the required-field validators.
[^2]: `src/rw_bot/harness/sweep.py` — `JOB_FIELDS = ("label", "seed", "doctrine", "samples")`; `scripts/play.py` takes `[doctrine-path]`.
[^3]: `tests/test_policy_doctrine.py` — `test_the_shipped_default_preset_matches_the_constant`, `test_the_duel_arms_form_a_one_field_chain`.
