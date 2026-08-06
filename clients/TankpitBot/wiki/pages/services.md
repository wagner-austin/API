---
title: Services — DI Architecture
tags: [codebase, di, services]
related:
  - "[[module-map]]"
  - "[[inheritance-chain]]"
  - "[[testing-patterns]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "238116afef165cc82b0d7213e11804b4764cf060"
fact_checked: "2026-08-05"
confidence: high
hubs: [codebase]
---

# Services — DI Architecture

Three services, injected two different ways. `CDPService` and `CommandService` are constructor kwargs on `SessionBase` (`cdp_service=` / `command_service=`, each defaulting to a self-constructed instance when omitted).[^1] `WorldService` is **not** a `SessionBase` kwarg — it is a module-level singleton in `sniffer/world_state.py` reached through `get_world_service()`, with `reset_world_state()` rebinding it for test isolation.[^4]

## CDPService (`browser/cdp_service.py`)

Owns CDP (Chrome DevTools Protocol) event handling and message capture.[^1]

- Accumulates captured frames on the public `messages` / `ws_urls` / `magic`
  attributes (`__init__`); `_record_frame` appends both directions
- Extracts the session magic in `_extract_magic_and_notify` and fires the
  `_on_magic_captured` callback; `SessionBase._on_magic_captured` is what
  turns that into the XOR key table via `init_trackers_with_magic`
- Sets up console listeners and CDP handlers
- `SessionBase` re-exposes the three attributes as `_messages`, `_ws_urls`,
  and `_magic` properties (getter + setter) so consumers never reach into
  the service directly[^2]

Created by `CDPService()` (no args). Injected as `cdp_service=` kwarg on SessionBase subclasses.[^1]

## CommandService (`bot/command_service.py`)

Owns XOR encoding and command dispatch to the game server.[^1]

- `send_bytes(data)` — XOR-encodes and sends via the WebSocket
- Holds reference to the `send_ws_bytes` callback (the actual WebSocket write)
- All bot commands (`move_to`, `shoot_at`, `teleport_to`, etc.) delegate to this

Created by `CommandService(send_ws_bytes=callback)`. The callback comes from the browser session's WebSocket handle.[^1]

## WorldService (`sniffer/world_service.py`)

Owns the mutable world state — tanks, containers, viewport, inventory.[^1]

- Injectable singleton: one instance shared across all consumers in a session
- 60+ files migrated to use it instead of direct state mutation
- Provides the `WorldStateDict` that AI decision functions read

## Factory wiring

**Bot**: `Bot.__init__` accepts `cdp_service` and `command_service` kwargs, passed through the inheritance chain to `SessionBase`.[^2]

**Probes**: `create_probe(probe_class, url)` in `action_lab/probe_factory.py` creates both services and injects them:[^3]
```
cdp_service = CDPService()
commands = CommandService(send_ws_bytes=send_websocket_bytes)
probe = probe_class(target_url=url, cdp_service=cdp_service, command_service=commands)
```
All 14 probe entry points go through this factory: combat, density, enemy-teleport, enemy-tracking, fuel, key, larder, mine-landing, movement, queue, radar-watch, respawn-watch, teleport, viewport.[^3] Note the arithmetic mismatch this invites: 14 `action_lab` modules call the factory, but only **13** carry a registered `[tool.poetry.scripts]` name — `queue_probe` has a `main()` at `scripts/queue_probe.py:39` and is not registered, so it runs via `python -m scripts.queue_probe` rather than a `tankpit-*` console script.

**BrowserSession**: same pattern — SessionBase constructor accepts the kwargs.[^2]

## Lifecycle functions

Standalone hookable functions in `browser/lifecycle.py`, used by both bot and sniffer:[^2]
- `navigate_and_login()` — opens URL, handles login flow
- `wait_for_game_ready()` — waits for game client to initialize
- `gather_intel()` — extracts static XOR key, terrain, field info
- `cleanup_browser()` — closes browser and cleans up

[^1]: Architecture phases A-C, landed 2026-06-14 on the merged `combat-rework` branch (commit accounting on [[module-map]] [^1]). Verified against the current tree 2026-08-05, since the whole-tree pin has drifted 84 files since this page was written: the three service modules are `src/tankpit_bot/browser/cdp_service.py`, `src/tankpit_bot/bot/command_service.py`, and `src/tankpit_bot/sniffer/world_service.py`, all present; the two injected as constructor kwargs appear in the `SessionBase.__init__` signature at `src/tankpit_bot/browser/session_base.py:38-45`, which is keyword-only for `headless`, `prefer_account`, `cdp_service`, `command_service`.
[^2]: Architecture phase D + bot decomposition, 2026-06-16 (same commit accounting as [^1]). The composition it produced is verified in the [[inheritance-chain]] line table, re-measured 2026-08-05. The four standalone lifecycle functions named above are in `src/tankpit_bot/browser/lifecycle.py`.
[^3]: `src/tankpit_bot/action_lab/probe_factory.py:18` — `create_probe_services()`; `create_probe()` is defined in the same module. Call sites counted 2026-08-05: 14 modules under `src/tankpit_bot/action_lab/` contain `= create_probe(`, matching the list above exactly (`queue_probe.py:542` is the one whose factory import is function-local).
[^4]: sniffer/world_state.py:18-33 — `_service = WorldService()` at module scope; `get_world_service()` returns it; `reset_world_state()` rebinds the global for tests. Verified 2026-07-31 against `browser/session_base.py:38-45`, whose `__init__` keyword-only parameters are `headless`, `prefer_account`, `cdp_service`, `command_service` — no `world_service`.
