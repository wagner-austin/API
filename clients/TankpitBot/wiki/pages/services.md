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
  "src/tankpit_bot": "fe0b742e68683be646eb604743ac226ded9af783"
fact_checked: "2026-08-12"
confidence: high
hubs: [codebase]
---

# Services — DI Architecture

Three services, all injected the same way. `CDPService`, `CommandService` and `WorldService` are constructor kwargs on `SessionBase` (`cdp_service=` / `command_service=` / `world=`), each defaulting to a self-constructed instance when omitted.[^1][^4]

**This page previously said `WorldService` was the exception** — a module-level singleton in `sniffer/world_state.py` reached through `get_world_service()`, rebound by `reset_world_state()` for test isolation. That is no longer true in any part: `world_state.py` **has been deleted**, and `get_world_service` and `reset_world_state` no longer exist anywhere in `src/tankpit_bot`. Each session now owns its world as `self.world`, which is what makes two sessions in one process keep two independent worlds — the thing the singleton made impossible. See [[session-state-deglobalisation]] step 8, which did the work.[^4]

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

[^1]: Architecture phases A-C, landed 2026-06-14 on the merged `combat-rework` branch (commit accounting on [[module-map]] [^1]). Verified against the current tree 2026-08-05, since the whole-tree pin has drifted 84 files since this page was written: the three service modules are `src/tankpit_bot/browser/cdp_service.py`, `src/tankpit_bot/bot/command_service.py`, and `src/tankpit_bot/sniffer/world_service.py`, all present; all three now appear in the `SessionBase.__init__` signature at `src/tankpit_bot/browser/session_base.py:38-42`, keyword-only for `headless`, `prefer_account`, `cdp_service`, `command_service` and — added since this footnote was written — `world: WorldService | None = None`, whose docstring calls it "The third member of the same injection list as the two services above". Re-read 2026-08-12.
[^2]: Architecture phase D + bot decomposition, 2026-06-16 (same commit accounting as [^1]). The composition it produced is verified in the [[inheritance-chain]] line table, re-measured 2026-08-05. The four standalone lifecycle functions named above are in `src/tankpit_bot/browser/lifecycle.py`.
[^3]: `src/tankpit_bot/action_lab/probe_factory.py:18` — `create_probe_services()`; `create_probe()` is defined in the same module. Call sites re-counted 2026-08-07: 14 modules under `src/tankpit_bot/action_lab/` contain `= create_probe(`, matching the list above exactly (`queue_probe.py:155` is the one whose factory import is function-local).
[^4]: **Re-verified 2026-08-12, and the earlier reading is now false.** `src/tankpit_bot/browser/session_base.py:74` — `self.world: WorldService = world if world is not None else WorldService()`, carrying the comment "This session's world state -- ITS OWN, not a process global. Two sessions in one process now keep two independent worlds, which the module singleton made impossible ([[session-state-deglobalisation]] step 8)." `src/tankpit_bot/sniffer/world_state.py` no longer exists, and a grep across `src/tankpit_bot` for `get_world_service` / `reset_world_state` / `_service = WorldService()` returns nothing. `WorldService` itself lives at `sniffer/world_service.py`, with `world_service_beliefs.py` / `world_service_movement.py` / `world_service_radar.py` alongside it. What this footnote previously recorded — `world_state.py:18-33`, the module-scope `_service`, and a `SessionBase.__init__` whose keyword-only parameters were `headless`, `prefer_account`, `cdp_service`, `command_service` with no world parameter — was accurate on 2026-07-31 and was undone by the de-globalisation.
