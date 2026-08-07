---
title: Inheritance Chain and Composition
tags: [architecture, di, composition]
related:
  - "[[coding-standards]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "fee6f258a2eca770e7edb72e4a4911af56ea8cd1"
fact_checked: "2026-08-07"
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
| `browser/session_base.py` | 208 | CDPService + CommandService composition, property delegations |
| `bot/completions.py` | 361 | HFSM state data, `_transition`, `_maybe_complete_*` methods |
| `bot/bot_dispatch.py` | 466 | Command dispatch, equipment, map operations |
| `bot/base.py` | 598 | Init, state access, game log, account stats, run loop |

Line counts re-measured 2026-08-07. `bot/base.py` came back under the
600-line ceiling by DELETING a duplicate rather than moving lines: the
account-stats capture became a free function taking `cdp` and `page`
(it needed no bot state), and `_init_game_log_scraper` / `_poll_game_log`
— implemented once here and once on `BrowserSession` — collapsed into
`browser/game_log.py`. The ceiling is now machine-checked
([[coding-standards]]), so this table cannot silently drift again.

## Shared composition

`SessionBase` provides CDPService + CommandService DI via constructor kwargs. Three consumers share it with zero duplicate code:[^1]
- `Bot(DispatchMixin)` — the game-playing bot
- `ProbeBase(SessionBase)` — action lab probes (14 probe entry points via the `create_probe()` factory)
- `BrowserSession(SessionBase)` — sniffer-specific scrapers only (130 lines)

## Factory DI

Bot accepts `cdp_service` and `command_service` as constructor kwargs. `create_probe()` injects both into all 14 probes. No service locator, no global state.[^1]

## Standalone lifecycle

`navigate_and_login()`, `wait_for_game_ready()`, `gather_intel()`, `cleanup_browser()` — standalone hookable functions in `browser/lifecycle.py`. Both sniffer and bot use them.[^1]

## Barrel removal

`__init__.py` files in `bot/`, `sniffer/`, `capture/` slimmed to docstrings (539 → 34 lines total). All callers import from specific submodules. This broke circular imports between CDPService → bot → browser.[^2]

[^1]: The architecture phases A-F landed on the `combat-rework` branch (`remotes/origin/combat-rework`, fully merged — `git log main..origin/combat-rework` is empty). Recounted 2026-08-05: **56** commits dated 2026-06-14 through 2026-06-16, of which **41** touch `clients/TankpitBot`. **Corrected 2026-08-05:** this footnote said "52 commits", which matches neither count on either scoping. The DI claims it backs are verified independently against the current tree: `SessionBase.__init__` takes `cdp_service` / `command_service` keyword-only, at `src/tankpit_bot/browser/session_base.py:38-45`; `create_probe` is called by 14 `action_lab` modules (`grep -c "= create_probe(" src/tankpit_bot/action_lab/*.py`); the three standalone lifecycle functions live in `src/tankpit_bot/browser/lifecycle.py`. Same figure appears on [[module-map]] and is corrected there identically.
[^2]: Barrel line counts re-measured 2026-08-05: `bot/__init__.py` 12 + `sniffer/__init__.py` 14 + `capture/__init__.py` 11 = **37 lines total** (re-counted 2026-08-07; it was 34 when written -- `sniffer/__init__.py` has since gained three). The "539 → 34" starting point and the "1081 mypy errors" are historical measurements of a pre-removal tree that no longer exists in any checkout reachable from HEAD; they are not re-derivable and are carried as narrative, not as verified figures. `BrowserSession` is at `src/tankpit_bot/browser/session.py`, 129 lines — matching the count above.
