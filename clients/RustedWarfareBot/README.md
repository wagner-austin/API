# RustedWarfareBot

Headless Rusted Warfare client. A Java agent inside the game's JVM dispatches orders and serialises simulation state; this Python package plans and evaluates.

The game is a plain Java program and boots fully headless via `-nodisplay`, so there is no virtual framebuffer, no screen scraping, and no input synthesis.

## Wiki

**Read `wiki/index.md` at the start of every session.** It is the single source of truth for engine internals, the headless harness, game mechanics, bot architecture, and multiplayer constraints. Navigate index → hub → page.

Claims about engine internals are pinned to a game build (`game_version` frontmatter) because the jar is obfuscated and class names change silently between releases.

## Build

```bash
make check          # lint + test + agent-selftest (the gate)
make lint           # guard + ruff + mypy over src, tests, scripts
make test           # pytest + coverage, 100% statements and branches
make agent          # compile + jar the javaagent with the game's own JDK 13
make agent-selftest # patch the real jar, verified by the JVM's bytecode verifier
```

`make check` includes `agent-selftest`, so the Java half is gated too: a patcher
regression, or an obfuscated class name that moved in a game update, fails at
the gate rather than inside a live engine.

## Game probes

The pinned game copy lives at `.game/` (untracked). Runs launch from there, never from the Steam install — every run rewrites `preferences.ini`, and Steam would otherwise update the build out from under the recorded class mappings.

```bash
make boot-probe    # headless engine boot, archives the engine log
make unit-dump     # -printunits stat catalogue
make sandbox-probe # live skirmish with the agent attached; runs until killed
rw-boot-log <log>  # structured summary of an engine log; exit 1 if it crashed
```

The engine dies on its first in-game frame without the agent attached, so every
in-game run goes through it — see `wiki/pages/agent-render-callback-noop.md`.

## Layout

```
agent/            Java javaagent, built by `make agent` (no third-party deps)
  src/rwbot/agent/  Premain, the class-file patcher, the target list, SelfTest
  manifest.mf       declares Premain-Class; checked against source by tests
src/rw_bot/       Python planner
  validation.py     require_* field validators
  harness/          launch configuration, agent resolution, boot-log decoding
scripts/          repo-local operational scripts, linted and covered like src
tests/            unit and integration tests against real archived engine logs
wiki/             the project's knowledge base and its cited source archive
.game/            pinned game copy (untracked, 451 MB)
```

Exceptions live beside the code that raises them, deriving from the shared base
in `rw_bot/__init__.py`; a central `errors.py` is banned by the guard.

## Standards

Strict typing throughout: no `Any`, no casts, no `type: ignore`, no `.pyi`, no `noqa`. Every TypedDict crossing an untyped boundary carries encode/decode with `require_*` validation. Dependency injection through `_test_hooks` modules bound to real implementations at import time — no mocks, no monkey-patching, no conditional test branches. Failures propagate with specific error codes rather than being softened.
