---
title: Inheritance Chain and Composition
tags: [architecture, di, composition]
related:
  - "[[coding-standards]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "0c8b596c7022a130d131f51e210c6945bbb59cd8"
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
- `BrowserSession(SessionBase)` — sniffer-specific scrapers only (127 lines)

## Factory DI

Bot accepts `cdp_service` and `command_service` as constructor kwargs. `create_probe()` injects both into all 14 probes. No service locator, no global state.[^1]

## Standalone lifecycle

`navigate_and_login()`, `wait_for_game_ready()`, `gather_intel()`, `cleanup_browser()` — standalone hookable functions in `browser/lifecycle.py`. Both sniffer and bot use them.[^1]

## Barrel removal

`__init__.py` files in `bot/`, `sniffer/`, `capture/` slimmed to docstrings (539 → 35 lines total). All callers import from specific submodules. This broke circular imports between CDPService → bot → browser.[^2]

[^1]: The architecture phases A-F landed on the `combat-rework` branch (`remotes/origin/combat-rework`, fully merged — `git log main..origin/combat-rework` is empty). Recounted 2026-08-05: **56** commits dated 2026-06-14 through 2026-06-16, of which **41** touch `clients/TankpitBot`. **Corrected 2026-08-05:** this footnote said "52 commits", which matches neither count on either scoping. The DI claims it backs are verified independently against the current tree: `SessionBase.__init__` takes `cdp_service` / `command_service` keyword-only, at `src/tankpit_bot/browser/session_base.py:38-42` — joined since by `world: WorldService | None = None`, so the composition section's "CDPService + CommandService DI" is now a three-service list (see [[services]] [^4]); `create_probe` is called by 14 `action_lab` modules (`grep -c "= create_probe(" src/tankpit_bot/action_lab/*.py`); the three standalone lifecycle functions live in `src/tankpit_bot/browser/lifecycle.py`. Same figure appears on [[module-map]] and is corrected there identically.
[^2]: Barrel line counts re-measured **2026-08-12**: `bot/__init__.py` 12 + `sniffer/__init__.py` 12 + `capture/__init__.py` 11 = **35 lines total**. The figure has moved twice — 34 when written, 37 on 2026-08-07 after `sniffer/__init__.py` gained three lines, and 35 now that it has shed two. The "539 → 35" starting point and the "1081 mypy errors" are historical measurements of a pre-removal tree that no longer exists in any checkout reachable from HEAD; they are not re-derivable and are carried as narrative, not as verified figures. `BrowserSession` is at `src/tankpit_bot/browser/session.py:38`, and the module is **127** lines (129 when last measured) — matching the count above.
