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
  "src/tankpit_bot": "eade4f75f4a72733d539a9faeb6991857c41ed3e"
fact_checked: "2026-07-31"
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
All 14 probe entry points go through this factory: combat, density, enemy-teleport, enemy-tracking, fuel, key, larder, mine-landing, movement, queue, radar-watch, respawn-watch, teleport, viewport.[^3]

**BrowserSession**: same pattern — SessionBase constructor accepts the kwargs.[^2]

## Lifecycle functions

Standalone hookable functions in `browser/lifecycle.py`, used by both bot and sniffer:[^2]
- `navigate_and_login()` — opens URL, handles login flow
- `wait_for_game_ready()` — waits for game client to initialize
- `gather_intel()` — extracts static XOR key, terrain, field info
- `cleanup_browser()` — closes browser and cleans up

[^1]: architecture phases A-C (WorldService, CommandService, CDPService) — 2026-06-14
[^2]: architecture phase D + bot decomposition — SessionBase composition, 2026-06-16
[^3]: action_lab/probe_factory.py — create_probe() and create_probe_services()
[^4]: sniffer/world_state.py:18-33 — `_service = WorldService()` at module scope; `get_world_service()` returns it; `reset_world_state()` rebinds the global for tests. Verified 2026-07-31 against `browser/session_base.py:38-45`, whose `__init__` keyword-only parameters are `headless`, `prefer_account`, `cdp_service`, `command_service` — no `world_service`.
