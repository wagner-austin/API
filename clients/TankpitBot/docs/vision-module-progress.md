# Vision Module Progress

## Goal

Make the bot more robust by adding multi-perspective tracking capabilities similar to what `sniffer/` and `tools/test_homing_exploit.py` have. The bot was "weak, brittle, and blind" - relying entirely on a single world state source with no fallback.

## Completed

### 1. Created `bot/vision.py` Module

New module with:

- **TypedDicts** for strict typing:
  - `TankRegistryEntryDict` - tank_id -> name/team mapping
  - `PositionEntryDict` - tank_id -> x/y position
  - `ContainerEntryDict` - x/y -> volume
  - `VisionStateDict` - complete vision snapshot

- **Factory Functions**:
  - `make_tank_registry_entry()`
  - `make_position_entry()`
  - `make_container_entry()`
  - `make_empty_vision_state()`

- **Encode/Decode** (with require_* validation):
  - `encode_*` / `decode_*` for each TypedDict
  - Full JSON roundtrip support

- **Immutable Mutations**:
  - `add_tank_to_registry()`
  - `update_tank_position()`
  - `update_container()`
  - `remove_container()`
  - `update_self_fuel_vision()`
  - `add_fuel_delta()`
  - `set_self_tank_id()`
  - `pickup_container_vision()`

- **Merge Functions** (multi-perspective):
  - `get_merged_fuel_containers()` - combines vision cache + world state
  - `get_merged_fuel()` - world state with vision fallback

- **Rendering**:
  - `render_vision_ascii()` - ASCII viewport from world state
  - `render_vision_debug()` - debug info for vision caches

### 2. Updated `bot/base.py`

Integrated vision module:

- Added `_vision_state: VisionStateDict` attribute
- Added methods:
  - `get_vision_state()` - access fallback caches
  - `get_all_fuel_containers()` - merged container view
  - `get_all_fuel()` - merged fuel with fallback
  - `render_ascii()` - ASCII viewport rendering
  - `render_debug()` - debug output
  - `get_nearest_all_fuel_container()` - nearest using merged sources
- Reset vision state in `run()` method

### 3. Updated `bot/__init__.py`

Exported all new vision types and functions (28 new exports).

### 4. Created `tests/test_vision.py`

Comprehensive test coverage for:
- Factory functions
- Encode functions
- Decode functions (including error cases)
- Mutation functions (immutability verified)
- Merge functions
- Render functions
- Roundtrip encode/decode tests

### 5. Tests for Vision Module

Created comprehensive tests in `tests/test_vision.py`:
- Factory function tests
- Encode/decode tests with roundtrip verification
- Mutation function tests (immutability verified)
- Merge function tests (multi-perspective)
- Render function tests
- Edge cases (missing fields, None dicts, non-dict values)

Added vision method tests to `tests/test_bot.py`:
- `test_get_vision_state()`
- `test_get_all_fuel_containers_empty()`
- `test_get_all_fuel()`
- `test_render_ascii()`
- `test_render_debug()`
- `test_get_nearest_all_fuel_container_*()` (3 cases)

**Result: 100% test coverage (1834 tests pass)**

### 6. Modularized `tests/test_container_decoder.py`

Split the 2168-line monolithic test file into 9 focused modules under `tests/container/`:

| Module | Description | Classes |
|--------|-------------|---------|
| `test_data.py` | Shared test constants from real captures | - |
| `test_helpers.py` | Validation helper tests | 4 classes |
| `test_structure.py` | Structure detection tests | 21 classes |
| `test_tank_decoders.py` | Tank registry/status/update tests | 8 classes |
| `test_combat_decoders.py` | Combat hit, deactivation tests | 3 classes |
| `test_movement_decoders.py` | Movement/position decoder tests | 5 classes |
| `test_radar_decoders.py` | Radar/pickup/teleport tests | 4 classes |
| `test_world_decoders.py` | World/chunk/tip/player list tests | 5 classes |
| `test_dispatcher.py` | Type identification and dispatch tests | 2 classes |
| `test_misc.py` | Error/DecodeLevel/unknown tests | 5 classes |

### 7. Modularized `tests/fakes.py`

Split the 1840-line monolithic fakes file into 4 focused modules under `tests/fakes/`:

| Module | Description | Classes |
|--------|-------------|---------|
| `__init__.py` | Package entry point with exports | - |
| `base.py` | Core fakes for general testing | 12 classes + 7 functions |
| `probe.py` | Probe-specific fakes | 8 classes + 9 functions |
| `bot.py` | Bot-specific fakes | 8 classes + 1 function |

### 8. Modularized `tests/test_protocol.py`

Split the 1788-line monolithic test file into 11 focused modules under `tests/protocol/`:

| Module | Description | Classes |
|--------|-------------|---------|
| `__init__.py` | Package init with module descriptions | - |
| `test_enums.py` | Enum tests (Rank, Team, Equipment, TerrainType) | 4 classes |
| `test_helpers.py` | Helper function tests (x16, x24, require_*) | 6 classes |
| `test_text.py` | Text message decoder tests | 3 classes |
| `test_combat.py` | Combat decoder tests (shoot, hit, deactivation, mines) | 5 classes |
| `test_resources.py` | Resource decoder tests (fuel, inventory, equipment) | 5 classes |
| `test_radar.py` | Radar decoder tests and validators | 9 classes |
| `test_movement.py` | Movement decoder tests | 2 classes |
| `test_tank.py` | Tank decoder tests (info, entry, exit, status) | 8 classes |
| `test_world.py` | World decoder tests (viewport, terrain, sync, container) | 8 classes |
| `test_dispatcher.py` | Main dispatcher and message type detection tests | 5 classes |

### 9. Modularized `tests/test_world_state.py`

Split the 1586-line monolithic test file into 9 focused modules under `tests/world_state/`:

| Module | Description | Classes |
|--------|-------------|---------|
| `__init__.py` | Package init with module descriptions | - |
| `helpers.py` | Shared test helper function | - |
| `test_constants.py` | Module constant tests | 1 class |
| `test_factories.py` | Factory function tests | 6 classes |
| `test_coord_key.py` | Coordinate key function tests | 2 classes |
| `test_encoders.py` | Encode function tests | 7 classes |
| `test_decoders.py` | Decode function tests | 8 classes |
| `test_mutations.py` | State mutation function tests | 13 classes |
| `test_rendering.py` | ASCII rendering tests | 2 classes |

### 10. Modularized `tests/test_bot.py`

Split the 1509-line monolithic test file into 9 focused modules under `tests/bot/`:

| Module | Description | Classes |
|--------|-------------|---------|
| `__init__.py` | Package init with module descriptions | - |
| `test_main.py` | main() function tests and error classes | 5 standalone tests |
| `test_commands.py` | Command types and encoding tests | 2 classes |
| `test_state_machine.py` | State machine tests | 4 classes |
| `test_class.py` | Bot class initialization and basic methods | 4 classes |
| `test_world_state.py` | Tests involving world state integration | 1 class |
| `test_cdp.py` | Tests with CDP session | 3 classes |
| `test_run.py` | Run method and game loop tests | 4 classes |
| `test_vision.py` | Vision method tests | 1 class |

### 11. Modularized `tests/test_browser.py`

Split the 1219-line monolithic test file into 7 focused modules under `tests/browser/`:

| Module | Description | Tests |
|--------|-------------|-------|
| `__init__.py` | Package init with module descriptions | - |
| `test_helpers.py` | Time helper function tests | 3 tests |
| `test_errors.py` | Error class tests | 3 tests |
| `test_session_basic.py` | BrowserSession init, properties, WebSocket, cleanup | 14 tests |
| `test_session_scrapers.py` | Game log, inventory, combat tracker tests | 12 tests |
| `test_session_fuel.py` | Fuel prober tests | 4 tests |
| `test_static_key.py` | Static key helpers, load/save, capture, derive | 17 tests |

### 12. Modularized `tests/test_types.py`

Split the 1071-line monolithic test file into 8 focused modules under `tests/types/`:

| Module | Description | Tests |
|--------|-------------|-------|
| `__init__.py` | Package init with module descriptions | - |
| `test_captured_message.py` | CapturedMessage encode/decode tests | 5 tests |
| `test_capture_session.py` | CaptureSession encode/decode with game_log/tank_names | 11 tests |
| `test_config.py` | SnifferConfig and BotConfig tests | 6 tests |
| `test_websocket.py` | WebSocketInfo and CDP WebSocket types | 10 tests |
| `test_input.py` | KeyInput, MouseInput, ProbeInput tests | 12 tests |
| `test_probe.py` | ProbeResult and ProbeSession tests | 10 tests |
| `test_session_summary.py` | CombatEvent, MessageStats, SessionSummary tests | 5 tests |

## Remaining Work

### File Modularization (>800 lines)

Files that still need to be split:

| File | Lines | Status |
|------|-------|--------|
| `tools/test_homing_exploit.py` | 1418 | Skipped (tool script, not test file) |
| `tests/test_browser.py` | 1219 | Done |
| `tests/test_types.py` | 1071 | Done |
| `tests/test_probe.py` | 1030 | Pending |
| `tests/sniffer/test_core.py` | 973 | Pending |
| `tests/test_login.py` | 890 | Pending |
| `tests/sniffer/test_formatters.py` | 845 | Pending |
| `tests/sniffer/test_world_state.py` | 840 | Pending |
| `tests/test_game_state.py` | 805 | Pending |

**Total: 5,383 lines across 6 test files remaining** (tool script excluded)

## Architecture Notes

The vision system follows the same patterns as existing code:
- Immutable state updates (new dict returned, original unchanged)
- TypedDicts with encode/decode pairs
- require_* validation in decoders
- Factory functions for creating instances
- Module-level state with reset function
- No Any, casts, type: ignore, or stubs
