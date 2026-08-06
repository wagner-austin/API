---
title: Inheritance Chain and Composition
tags: [architecture, di, composition]
related:
  - "[[coding-standards]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "238116afef165cc82b0d7213e11804b4764cf060"
fact_checked: "2026-08-05"
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
| `bot/completions.py` | 361 | HFSM state data, `_transition`, `_maybe_complete_*` methods |
| `bot/bot_dispatch.py` | 466 | Command dispatch, equipment, map operations |
| `bot/base.py` | 645 | Init, state access, game log, account stats, run loop |

Line counts re-measured 2026-08-05 (`wc -l`). **Corrected:** `bot_dispatch.py`
was listed at 440 and `base.py` at 621; both grew. `session_base.py` (196) and
`completions.py` (361) are unchanged. Note `bot/base.py` at 645 is now **over
the 600-line ceiling** in [[coding-standards]] — it is not in the 2026-07-31
backlog list, so it crossed the bar after that sweep and is due a split when
next touched.

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
[^2]: Barrel line counts re-measured 2026-08-05: `bot/__init__.py` 12 + `sniffer/__init__.py` 11 + `capture/__init__.py` 11 = **34 lines total**, matching the figure above. The "539 → 34" starting point and the "1081 mypy errors" are historical measurements of a pre-removal tree that no longer exists in any checkout reachable from HEAD; they are not re-derivable and are carried as narrative, not as verified figures. `BrowserSession` is at `src/tankpit_bot/browser/session.py`, 130 lines — matching the count above.
