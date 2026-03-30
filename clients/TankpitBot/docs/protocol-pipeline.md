# Protocol Pipeline

## Purpose

This document describes the actual runtime path for Tankpit traffic in this
repo:

- capture
- decode
- dispatch
- world-state mutation
- bot planning
- command encoding and send

If this document disagrees with the code, the code wins.

## Runtime Flow

### 1. WebSocket Capture

Live browser traffic is captured through Playwright + CDP in:

- `src/tankpit_bot/browser/session.py`
- `src/tankpit_bot/sniffer/core.py`
- `src/tankpit_bot/bot/base.py`

Important point:

- bot runtime and sniffer runtime both depend on the same lower-level WebSocket
  capture path

### 2. Bot Message Buffer

In live bot mode, received CDP payloads are buffered in:

- `Bot._on_message_captured()` in `src/tankpit_bot/bot/base.py`

That method appends received payloads to:

- `bot._cdp_message_buffer`

No decode happens there beyond the shared message-capture hooks.

### 3. Tick Sync Phase

Each game tick, the bot drains buffered payloads in:

- `src/tankpit_bot/bot/world_sync.py`

Entry point:

- `drain_messages(bot)`

This feeds each buffered payload into:

- `process_received_message()` in `src/tankpit_bot/sniffer/decoders.py`

### 4. Frame Splitting and Binary/Text Decode

`process_received_message()` in `src/tankpit_bot/sniffer/decoders.py`:

1. base64-decodes the frame
2. splits one WebSocket frame into one or more logical messages using the
   2-byte little-endian length prefix
3. routes each message through `_process_single_message()`

For text messages:

- decode as UTF-8
- log only

For binary messages:

- XOR-decode the body
- call `protocol.decode_message(...)`
- format for logs
- dispatch to world-state update logic

## Binary Decode Layers

### Protocol Decoder Layer

Primary binary decode entry points:

- `src/tankpit_bot/protocol/decoders/__init__.py`
- `src/tankpit_bot/container/decoders/__init__.py`

These decoders produce TypedDict-shaped protocol messages.

Examples:

- `movement_response`
- `tank_status`
- `radar_ack`
- `radar_response`
- `world_state`
- `deactivation`

### Container Decode Layer

Some traffic uses container-style wrapping and is decoded in:

- `src/tankpit_bot/container/decoders/`

This is especially important for:

- 0x2E subtypes
- large `world_state`-like container messages

## World-State Dispatch

After decode, messages are fed into:

- `dispatch_world_state_update()` in
  `src/tankpit_bot/sniffer/world_state.py`

This is the central mutation router for:

- fuel/resource updates
- tank updates
- position updates
- movement responses
- viewport entities
- radar results
- combat hits
- deactivation
- world-state blobs

### Important World-State Sources

1. `world_state`
   - large container/blob style map snapshots
   - parsed by `_parse_world_state_blob(...)`
   - primary global tank-position source

2. `radar_response`
   - local fuel/equipment/mine discovery

3. `movement` / `movement_response` / `position_update`
   - self and tank positional updates

4. `enemy_detection`
   - nearest-enemy style absolute detection

5. `viewport_update`
   - local entity update source

## Bot Planning Flow

After sync, the live bot executes:

1. `Bot._update_state_from_world()` in `src/tankpit_bot/bot/base.py`
2. `_tick_once()` in `src/tankpit_bot/bot/tick_loop.py`
3. `decide(...)` in `src/tankpit_bot/bot/ai_strategy.py`
4. `execute(...)` in `src/tankpit_bot/bot/executor.py`

### Separation of Concerns

- `base.py`
  - execution state convergence
  - completion of in-flight actions

- `tick_loop.py`
  - tick orchestration
  - sync / readiness / stall handling

- `ai_strategy.py`
  - choose one action for the tick

- `executor.py`
  - turn chosen command into bot method calls

## Command Encoding and Send

Outgoing commands are built in:

- `src/tankpit_bot/protocol/commands.py`
- `src/tankpit_bot/bot/commands.py`

Sent through:

- `_send_bytes()` in `src/tankpit_bot/bot/base.py`

That path:

1. frames the command
2. XOR-encodes bytes after the `!` prefix when required
3. sends through the captured CDP WebSocket session

## Map Open, Radar, and Teleport

### Map Open

- command ID is `CMD_MAP_OPEN` in `src/tankpit_bot/protocol/commands.py`
- bot sends it through `open_map()` in `src/tankpit_bot/bot/base.py`
- the game does not expose a reliable authoritative map-open flag

Current behavior:

- map-open is tracked as an in-flight action for sequencing only
- it is cleared after a fresh world sync arrives

### Radar

- command ID is `CMD_RADAR`
- sent via `use_radar()`
- completion is driven by radar ack/result handling in
  `src/tankpit_bot/sniffer/world_state.py`

### Teleport

- command ID is `CMD_MAP_TELEPORT`
- bot currently opens the map immediately before teleport in
  `src/tankpit_bot/bot/executor.py`

## Offline Decode Path

Offline captured sessions are decoded with:

- `scripts/decode.py`

That script:

1. loads a saved capture session
2. restores the XOR table from captured magic
3. decodes each recorded frame
4. prints the decoded output

This is useful for protocol inspection, but it is not yet a full bot replay
system.

## Current Limits

1. There is no dedicated known protocol command for “full world state now”.
2. `map_open` is used as the current practical trigger for global enemy refresh.
3. The live bot path and offline decode path share decoder logic, but there is
   not yet a first-class replay harness for planner regression testing.

## Files to Read First

If you are debugging the protocol/control path, start here:

1. `src/tankpit_bot/bot/tick_loop.py`
2. `src/tankpit_bot/bot/world_sync.py`
3. `src/tankpit_bot/sniffer/decoders.py`
4. `src/tankpit_bot/sniffer/world_state.py`
5. `src/tankpit_bot/bot/ai_strategy.py`
6. `src/tankpit_bot/bot/executor.py`
7. `src/tankpit_bot/protocol/commands.py`
