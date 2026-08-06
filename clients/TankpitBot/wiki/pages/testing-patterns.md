---
title: Testing Patterns
tags: [codebase, testing, di]
related:
  - "[[services]]"
  - "[[coding-standards]]"
source_paths:
  - "tests"
  - "scripts/guard.py"
source_git_blobs:
  "tests": "a2ceaaaa48c0f1ef8b2915c74841faa22a7b5e5b"
  "scripts/guard.py": "bbb7d497792759b7b6cc0df7489e2007a4c2ee6c"
fact_checked: "2026-08-06"
confidence: high
hubs: [codebase]
---

# Testing Patterns

5,631 tests, 100% coverage, zero mocks, zero monkey-patching.[^1]

## `_test_hooks` DI pattern

Production code declares protocol interfaces in `tankpit_bot/_test_hooks/`. 8 submodules organized by domain:[^2]

| Submodule | Protocols | What it replaces |
|-----------|-----------|-----------------|
| `bot.py` | `BotProtocol`, `BufferedMessageSourceProtocol` | Bot command dispatch surface |
| `browser.py` | `BrowserProtocol`, `BrowserContextProtocol`, etc. | Playwright browser objects |
| `cdp.py` | `PageProtocol`, `CDPSessionProtocol`, `KeyboardProtocol` | Playwright Page + CDP |
| `env.py` | env-var resolution | `os.environ` access |
| `fs.py` | filesystem operations | file I/O |
| `playwright_loader.py` | `SyncPlaywrightFactoryProtocol` | `sync_playwright()` |
| `terrain.py` | `TerrainMapProtocol` + loader hook | Terrain GIF loading |
| `runtime.py` | argv, static-byte discovery, replay dispatch | CLI + runtime hooks |

## How to write a test

1. **Import the protocol** from `_test_hooks` (e.g. `BotProtocol`)
2. **Create a class that implements the protocol** — must match the real API signature exactly. No `unittest.mock`, no `MagicMock`, no `patch`.
3. **Inject it** via the production code's constructor or function parameter
4. **Assert on concrete values** — no `assert len(x) >= 0` or `assert x is not None` (the guard rejects weak assertions)[^3]

Example pattern (this shape recurs throughout `tests/` — enforced by the guard, [[coding-standards]]):
```python
class FakeBot:
    """Protocol-matching test implementation."""
    def __init__(self) -> None:
        self.commands_sent: list[BotCommand] = []
    
    def shoot_at(self, x: int, y: int, target_id: int) -> None:
        self.commands_sent.append(make_shoot_command(x, y, target_id))
```

## MonkeyPatchBanRule

The `_hooks_guard.py` module enforces: **never set module attributes directly in tests**. The guard scans for `setattr` / direct attribute assignment on imported modules. Instead, use the save-and-restore pattern:[^4]

```python
# Save original
original = tankpit_bot._test_hooks.env.get_env
# Swap
tankpit_bot._test_hooks.env.get_env = fake_get_env
try:
    # test code
finally:
    # Restore
    tankpit_bot._test_hooks.env.get_env = original
```

## Replay regression tests

`tests/replay/` re-runs captured sessions through the bot's decision logic using `replay/engine.py`. These verify that AI decisions haven't regressed against known-good captures. Assertions are on specific decision outputs (command type, target coordinates, state transitions), not just "it didn't crash."[^5]

## Coverage

`fail_under = 100` in `pyproject.toml`. Branch coverage enabled.
`concurrency = ["greenlet", "thread"]` — `greenlet` keeps the tracer
across Playwright sync-API context switches; `thread` was added for the
bot service, whose `StatusBus` / `ModeBridge` primitives cross the
aiohttp thread and the tick-loop thread by design ([[bot-service-architecture]]).
Four live-only probe paths are `omit`-ed: `action_lab/combat_probe.py`,
`action_lab/enemy_tracking.py`, and their two `scripts/` wrappers.[^1]

[^1]: pyproject.toml [tool.coverage.report] — fail_under=100, branch=true, concurrency=greenlet
[^2]: tankpit_bot/_test_hooks/__init__.py — 8 submodules, all protocol-based
[^3]: `WeakAssertionRule` (`name = "test-quality"`) in the api monorepo at `libs/monorepo_guards/src/monorepo_guards/test_quality_rules.py:418,421`, reached from this project through `scripts/guard.py`. Its module docstring at `:1-14` enumerates what it rejects: `weak-assertion-is-not-none`, `-isinstance`, `-hasattr`, `-len-zero` (`assert len(x) > 0` "checks existence not content"), `-in-output`, `-key-in-dict`, and `mock-without-assert-called-with`. The stated rationale is the one this page relies on — "Coverage shows lines executed, not correctness proven."
[^4]: _hooks_guard.py MonkeyPatchBanRule — enforces save-and-restore, 0 violations
[^5]: `tests/replay/test_real_session_regressions.py` — 11 regression tests replaying captured sessions through the bot's decision logic, loaded via `tests/replay/fixture_loader.py` from `tests/replay/fixtures/`; the engine itself is covered by `tests/replay/test_engine.py`. Counted 2026-08-06.
