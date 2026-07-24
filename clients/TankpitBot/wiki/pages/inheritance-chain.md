---
title: Inheritance Chain and Composition
tags: [architecture, di, composition]
related:
  - "[[coding-standards]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "b347a7ecabff0e55f5aa17840d35a4ccae1eca10"
fact_checked: "2026-06-16"
confidence: high
hubs: [architecture]
---

# Inheritance Chain and Composition

## Linear chain

```
Bot → DispatchMixin → CompletionsMixin → SessionBase
```

| Module | Lines | Concern |
|--------|-------|---------|
| `browser/session_base.py` | 196 | CDPService + CommandService composition, property delegations |
| `bot/completions.py` | 340 | HFSM state data, `_transition`, `_maybe_complete_*` methods |
| `bot/bot_dispatch.py` | 397 | Command dispatch, equipment, map operations |
| `bot/base.py` | 465 | Init, state access, game log, account stats, run loop |

## Shared composition

`SessionBase` provides CDPService + CommandService DI via constructor kwargs. Three consumers share it with zero duplicate code:[^1]
- `Bot(DispatchMixin)` — the game-playing bot
- `ProbeBase(SessionBase)` — action lab probes (6 probe types via `create_probe()` factory)
- `BrowserSession(SessionBase)` — sniffer-specific scrapers only (134 lines)

## Factory DI

Bot accepts `cdp_service` and `command_service` as constructor kwargs. `create_probe()` injects both into all 6 probes. No service locator, no global state.[^1]

## Standalone lifecycle

`navigate_and_login()`, `wait_for_game_ready()`, `gather_intel()`, `cleanup_browser()` — standalone hookable functions in `browser/lifecycle.py`. Both sniffer and bot use them.[^1]

## Barrel removal

`__init__.py` files in `bot/`, `sniffer/`, `capture/` slimmed to docstrings (539 → 34 lines total). All callers import from specific submodules. This broke circular imports between CDPService → bot → browser.[^2]

[^1]: architecture phases A-F + bot decomposition, 52 commits on combat-rework branch (2026-06-14 through 2026-06-16)
[^2]: barrel removal fixed 1081 mypy errors from circular import chain; all test imports rewritten to specific submodules
