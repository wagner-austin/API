# RustedWarfareBot

**System identification against an obfuscated binary.** Rusted Warfare ships as a
compiled Java program with no published internals and class names that change
silently between releases, so this project infers the engine's rules from the inside
and pins every claim to the build it was measured on.

A Java agent injected into the game's own JVM dispatches orders and serialises
simulation state; this Python package plans, evaluates, and verifies. The standing
goal is a 100% win rate against the built-in AI at Impossible and every rung below,
measured, with any champion match watchable live.

The game is a plain Java program and boots fully headless via `-nodisplay`, so there is no virtual framebuffer, no screen scraping, and no input synthesis.

Verification is structural rather than incidental: `mechanics/` holds the extracted
game facts, doctrines make each experiment differ from its control by exactly one
field, seeded matches are bit-identical per seed with a world digest that proves it,
and `make agent-selftest` patches the real jar so a moved obfuscated class fails at
the build gate instead of inside a live engine.

## Wiki

**Read `wiki/index.md` at the start of every session.** It is the single source of truth for engine internals, the headless harness, game mechanics, bot architecture, and multiplayer constraints. Navigate index → hub → page; `wiki/log.md` is the dated record of every measurement and verdict.

Claims about engine internals are pinned to a game build (`game_version` frontmatter) because the jar is obfuscated and class names change silently between releases.

## Build

```bash
make check          # lint + test + agent-selftest (the gate)
make lint           # guard + ruff + mypy over src, tests, scripts
make test           # pytest + coverage, 100% statements and branches
make agent          # compile + jar the javaagent with the game's own JDK
make agent-selftest # patch the real jar, verified by the JVM's bytecode verifier
```

`make check` includes `agent-selftest`, so the Java half is gated too: a patcher
regression, or an obfuscated class name that moved in a game update, fails at
the gate rather than inside a live engine.

## Playing

```bash
make play    # one headless match: agent starts it, the planner plays it
make sweep   # a batch of matches in parallel, one scorecard per match
make watch   # a real window on a champion match (Impossible by default)
make host    # bot-hosted LAN game a human can join (sparring; shelved)
make income  # the economy probe behind the measured credits/s figures
```

A match is shaped by a **doctrine** file (`doctrines/*.doctrine`) — goals,
wave mass, and one flag per behaviour, every field required so an arm differs
from its control by exactly what the experiment says. A **sweep**
(`sweeps/*.txt`, `label|seed|doctrine|samples` per line) freezes a snapshot of
the code under `runs/sweeps/<name>/.tree` so the working tree stays editable
mid-batch, files one scorecard per match, and resumes whatever is missing when
re-run. `scripts/analyze_sweep.py` prints a batch as one table;
`scripts/ledger.py` flattens every match ever played into TSV.

Matches are reproducible: the engine's generators are seeded, the world is
held for the planner in lockstep, and the simulation's frame delta is pinned
(`PLAY_PINDELTA=3`, `SWEEP_PINDELTA=3`), making a solo run bit-identical per
seed — traces carry a world digest that proves it. The pins are skipped for
`make watch` and `make host`, where the simulation must track the wall clock.

## Game probes

The pinned game copy lives at `.game/` (untracked). Runs launch from there, never from the Steam install — every run rewrites `preferences.ini`, and Steam would otherwise update the build out from under the recorded class mappings. Sweeps give each worker its own clone (`.game-w1` …).

```bash
make boot-probe     # headless engine boot, archives the engine log
make unit-dump      # -printunits stat catalogue
make sandbox-probe  # live skirmish with the agent attached; runs until killed
make discover-probe # reflective engine-state snapshots at timed offsets
make wire-capture   # archive the NDJSON state stream a live agent emits
make type-flags     # placement flags + build-tree edges, one registry pass
make decompile      # CFR over the pinned jar into runs/decompiled
rw-boot-log <log>   # structured summary of an engine log; exit 1 if it crashed
```

The engine dies on its first in-game frame without the agent attached, so every
in-game run goes through it — see `wiki/pages/agent-render-callback-noop.md`.

## Layout

```
agent/            Java javaagent, built by `make agent` (no third-party deps)
  src/rwbot/agent/  Premain, the class-file patcher, transformers, SelfTest
  manifest.mf       declares Premain-Class; checked against source by tests
src/rw_bot/       Python planner
  wire/             the NDJSON contract: samples in, orders out
  control/          the socket channel to a running agent
  mechanics/        catalogue, combat profiles, build tree — the game's facts
  policy/           the decisions: plan, economy, combat, dispatch, doctrine
  harness/          sweep runner, frozen trees, boot-log decoding
scripts/          entry points and operational scripts, linted and covered like src
doctrines/        gameplay styles, one file per arm
sweeps/           batch definitions, one file per experiment
tests/            unit and integration tests against real archived engine logs
wiki/             the project's knowledge base and its cited source archive
runs/             match artifacts: scorecards, traces, logs, decompiled source
.game/            pinned game copy (untracked, 451 MB)
```

Exceptions live beside the code that raises them, deriving from the shared base
in `rw_bot/__init__.py`; a central `errors.py` is banned by the guard.

## Standards

Strict typing throughout: no `Any`, no casts, no `type: ignore`, no `.pyi`, no `noqa`. Every TypedDict crossing an untyped boundary carries encode/decode with `require_*` validation. Dependency injection through `_test_hooks` modules bound to real implementations at import time — no mocks, no monkey-patching, no conditional test branches. Failures propagate with specific error codes rather than being softened. Every behavioural change is screened against seeded matches before it stays; refutations are recorded where the code was.
