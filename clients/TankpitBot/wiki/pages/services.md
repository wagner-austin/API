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
  "src/tankpit_bot": "e6caa14c02ab54236704da0d429fae01cbec8aa6"
fact_checked: "2026-06-16"
confidence: high
hubs: [codebase]
---

# Services — DI Architecture

Three injectable services, composed via constructor kwargs on `SessionBase`. No service locator, no global state.[^1]

## CDPService (`browser/cdp_service.py`)

Owns CDP (Chrome DevTools Protocol) event handling and message capture.[^1]

- Buffers received WebSocket messages (`_cdp_message_buffer`)
- Builds XOR key table from magic bytes (`_on_magic_captured`)
- Sets up console listeners and CDP handlers
- Property delegations: `_messages`, `_ws_urls`, `_magic`

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
All 6 probe types go through this factory.[^3]

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
