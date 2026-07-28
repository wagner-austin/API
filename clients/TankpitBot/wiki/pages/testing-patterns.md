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
  "tests": "3e7ad8a90c094b9594575cb6fb71ee1456c2759f"
  "scripts/guard.py": "508e6328c73be452a042fd00162168a921d7b1b9"
fact_checked: "2026-06-16"
confidence: high
hubs: [codebase]
---

# Testing Patterns

3,923 tests, 100% coverage, zero mocks, zero monkey-patching.[^1]

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

`fail_under = 100` in `pyproject.toml`. Branch coverage enabled. `concurrency = ["greenlet"]` to handle Playwright's sync API context switches.[^1]

[^1]: pyproject.toml [tool.coverage.report] — fail_under=100, branch=true, concurrency=greenlet
[^2]: tankpit_bot/_test_hooks/__init__.py — 8 submodules, all protocol-based
[^3]: test quality guard — rejects weak assertions like `>= 0` or `is not None`
[^4]: _hooks_guard.py MonkeyPatchBanRule — enforces save-and-restore, 0 violations
[^5]: tests/replay/ — regression tests against captured sessions
