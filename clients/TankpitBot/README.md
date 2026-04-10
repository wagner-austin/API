# TankpitBot

Automated bot client for Tankpit.com browser game. Uses Playwright and Chrome DevTools Protocol (CDP) to capture and reverse-engineer the game's WebSocket protocol, with XOR codec for message encoding/decoding. Includes a modular AI behavior system for autonomous tank control.

## Features

- **Autonomous AI Bot**: Durable HFSM owner routing with HUNT, RECOVER_FUEL, and RECOVER_EQUIPMENT modes, pathfinding, threat analysis, and executor-side command validation
- **Protocol Discovery**: Captures WebSocket traffic via Playwright CDP integration
- **WebSocket Injection**: Sends commands via captured WebSocket (synthetic JS events don't work)
- **XOR Codec**: Full encode/decode with static + session magic keys
- **Container Decoder**: Length-based message identification for 13+ container subtypes
- **Intel Gathering**: Console listener, WebSocket URLs, JS debug info, script URLs
- **Shared Architecture**: BrowserSession base class for sniffer, probe, and bot
- **Type Safety**: mypy strict mode, zero `Any` types, immutable TypedDict models
- **100% Test Coverage**: statements and branches, no mocks
- **Monorepo Integration**: Guard rules, platform_core utilities

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+

### Installation

```bash
cd clients/TankpitBot
poetry install --with dev
```

This installs dependencies and Playwright's Chromium browser.

### Run the Bot

```bash
make bot
```

The bot joins a game, captures the WebSocket, and runs the AI behavior loop autonomously.
Each run is saved to:

- `runs/bot/latest.log`
- `runs/bot/latest.events.jsonl`
- `runs/bot/latest.capture_session.json`

plus timestamped archive copies for all three.

### Capture the Protocol

```bash
make sniff
```

This will:
1. Launch a Chromium browser
2. Navigate to tankpit.com
3. Capture all WebSocket messages
4. Save the session to `runs/sniff/latest.capture_session.json` by default
5. Save companion files `runs/sniff/latest.raw_capture.json` and
   `runs/sniff/latest.session_summary.json`

If `TANKPIT_OUTPUT` is set, that requested output is still written, and the same
run is mirrored into the canonical `runs/sniff/latest.*` paths.

### Probe Input Commands

```bash
make probe
```

This will:
1. Join a game using account credentials
2. Capture the game's WebSocket via prototype hook
3. Send known commands via WebSocket injection:
   - `f` - Map open (XOR command via WebSocket)
   - `f` - Map close (JavaScript keypress toggle)
   - `s` - Radar (XOR command)
   - `d` - Mine (XOR command)
   - `q` - Quit (plain command)
4. Record WebSocket responses from the server
5. Save results to `probe_session.json`

**Note**: Synthetic JavaScript KeyboardEvents don't work because browsers set `isTrusted: false` on programmatically created events. The probe uses WebSocket injection instead.

### Teleport Probes

```bash
make teleport-probe          # Run safe + aggressive strategies
make teleport-probe-safe     # sync_before_teleport only
make teleport-probe-aggressive  # immediate_after_map_open only
make enemy-teleport-probe    # Enemy-directed combat approach
make fuel-probe              # Fuel container pickup sequences
```

These live diagnostic probes isolate transport-level server acceptance from
planner logic. Results are saved to `teleport_probe.json`,
`enemy_teleport_probe.json`, and `fuel_probe.json` with companion capture
session files.

### Replay Bot

```bash
poetry run python -m scripts.replay_bot [session.json] [--json]
```

Loads a captured WebSocket session and replays it offline through the protocol
decoders and AI planner tick-by-tick, outputting structured decision traces
without a live browser.

### Decode Captured Session

```bash
make decode
```

Loads a capture session JSON, extracts the magic key, builds the XOR table, and decodes all command messages.

---

## AI Behavior System

The bot uses a tick-based planning pipeline built on pure functions, immutable
TypedDicts, and a durable HFSM owner model. The important layers are:

- **World sync**: CDP WebSocket frames are drained each tick and decoded into
  world state. Every tracked entity (tank, container, mine) carries a `source`
  field (`viewport`, `radar`, or `world_state`) for freshness validation.
- **Planning**: `bot/ai_strategy.py` selects exactly one durable mode owner per
  tick (`RECOVER_FUEL`, `RECOVER_EQUIPMENT`, or `HUNT`) and delegates to the
  corresponding owner module.
- **Execution**: `bot/executor.py` validates commands against live world state
  before dispatching. AI state is only persisted after successful dispatch.
- **Action lifecycle**: `bot/states.py` and `bot/tick_loop.py` track in-flight
  actions and completion/timeouts.

The control architecture is documented in
[`docs/bot-control-model.md`](docs/bot-control-model.md) and the HFSM
migration plan is in
[`docs/bot-hfsm-refactor-plan.md`](docs/bot-hfsm-refactor-plan.md).

### Durable Mode Owners

| Mode | Entry | Exit | Description |
|------|-------|------|-------------|
| `RECOVER_FUEL` | fuel <= low threshold | fuel >= full threshold (1100) | Locked target pursuit, known-fuel registry, radar sense, sector hop, edge walk |
| `RECOVER_EQUIPMENT` | any combat reserve <= break | all reserves >= resume | Locked target pursuit, known-equipment registry, radar sense, sector hop |
| `HUNT` | no recovery needed | recovery takes priority | ACQUIRE, REFRESH, CLOSE, ENGAGE, CONFIRM_KILL substates |

### Executor Validation

Before dispatching any command, the executor validates against current world
state:

- **Shoots**: target must be tracked, at the expected coordinates, and
  viewport-confirmed
- **Pickups**: container must exist and match the expected kind (fuel/equipment)
- **Moves/Teleports**: destination must not be a known mine
- **Combat teleports**: locked combat target must still be tracked with a valid
  source
- **Resource teleports**: locked resource target must still exist with a locally
  trustworthy source

### Current Planner Notes

- Combat uses map-open as the known fallback for global enemy refresh.
- Radar is used for local resource search, not global enemy positions.
- Extra radar scans the full 18x18 viewport envelope; built-in radar scans a
  7x7 area centered on the tank. Viewport-scanned state is only set by extra
  radar, not built-in radar.
- World viewport state tracks the real visible 16x16 viewport. Direct move and
  pickup commands are allowed on visible edge tiles; only tiles beyond the
  visible viewport are treated as non-actionable.
- Movement uses viewport-bounded A* pathfinding. Paths that would leave the
  current visible viewport fall back to teleport instead of waypoint walking.
- The bot only trusts current-viewport fuel/equipment targets after extra radar
  has confirmed that viewport. Sparse `0x5A` cache entries remain unconfirmed
  until radar refreshes the current screen.
- Repeated radar in the same already-confirmed viewport is intentionally
  skipped.
- Teleport affordability uses the exact in-game fuel formula
  `floor(6 * sqrt(dx^2 + dy^2))` rather than a flat estimated cost.
- `0x5A` is a sparse tile patch, not a full visible-tank snapshot. It is
  authoritative for viewport origin and tile cache updates, but absence from a
  single `0x5A` patch does not imply tank absence.
- Fresh enemy-mine reveal is proven through tunneled `0x2E -> 0x4F` radar
  results, and local mine placement through tunneled `0x2E -> 0x4B` placement
  updates.
- The bot tracks in-flight actions explicitly (`move`, `collect`, `teleport`,
  `scan`, `shoot`, `map_open`) and waits for completion or timeout before
  replanning.
- Teleport completion validates the actual landed position against the requested
  landing target and blacklists mismatched teleport landings immediately.
- Equipment recovery uses break/resume thresholds rather than a single
  hard-coded “critical” level.

### Documentation Status

- Protocol/decode docs are in `docs/protocol-discovery.md`,
  `docs/protocol-reference.md`, `docs/protocol-decoding-status.md`, and
  `docs/protocol-pipeline.md`.
- The current bot control model is documented in `docs/bot-control-model.md`.
- The HFSM/control-architecture migration plan is in
  `docs/bot-hfsm-refactor-plan.md`.
- Run output locations are documented in `docs/run-artifacts.md`.
- Bot terminal/event channels are documented in `docs/bot-logging.md`.
- The old README descriptions of `DEFEND`, `PATROL`, `DEPOSIT_FUEL`, and
  evaluator-based AI layers were stale and should not be used as the current
  architecture reference.

---

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_URL` | `https://tankpit.com/play` | Target URL for sniffer |
| `TANKPIT_OUTPUT` | `runs/sniff/latest.capture_session.json` | Capture output path |
| `TANKPIT_HEADLESS` | `true` | Run browser headlessly |
| `TANKPIT_DURATION_MS` | `60000` | Capture duration in ms |
| `TANKPIT_LIVE_DECODE` | `true` | Show decoded messages in real-time |
| `TANKPIT_PREFER_ACCOUNT` | `false` | Skip guest login, use account directly |
| `TANKPIT_USERNAME` | (none) | Account username for login |
| `TANKPIT_PASSWORD` | (none) | Account password for login |
| `PYTHONUTF8` | `1` | Windows console UTF-8 support |

**Note**: Guest accounts are rate-limited per IP. If you hit the limit, the sniffer will automatically attempt to log in using `TANKPIT_USERNAME` and `TANKPIT_PASSWORD` if configured.

---

## Protocol Status

**94% signature coverage, 92% fully decoded**

- 31+ message types documented in `protocol/types.py`
- Length-based identification for session-independent decoding
- Container decoder (`container/`) for 0x2E subtypes
- See `docs/protocol-decoding-status.md` for detailed message formats

### Fully Decoded Messages

| Sig | Name | Description |
|-----|------|-------------|
| 0x21 | tank_info | Tank metadata (team, id, decoration, score, name) |
| 0x2B | promotion | Promotion event (text format) |
| 0x2E | container | Container wrapping 13+ subtypes |
| 0x3D | movement_response | Movement confirmation with position |
| 0x3E | tank_status | Full tank status (22 bytes) |
| 0x41 | deactivation | Kill/death event |
| 0x47 | movement | Movement command response |
| 0x49 | item_pickup | Equipment pickup notification |
| 0x4F | radar_result | Radar scan results |
| 0x53 | shooting | Shot fired event |
| 0x5A | viewport_update | Delta-compressed map update |

### Container Subtypes (0x2E)

| Length | Type | Description |
|--------|------|-------------|
| 2-3 | tank_status_sync | Heartbeat/sync |
| 4 | tunneled terrain_update / legacy short subtype | Real captures include tunneled `0x4A` structure updates here |
| 5 | deactivation_kill | You killed another tank |
| 6 | tank_leave | Player exits game |
| 7 | deactivation_death | You were killed |
| 9 | tank_status_short | Enemy HP/rank update |
| 10 | tank_update_compact | Compact tank update |
| 11 | combat_hit | Combat hit event |
| 13 | position_update | Position/status update |
| 14-15 | tank_update_extended/full | Extended tank updates |
| 16-20 | tank_registry | Tank registry entry |

---

## Development

### Commands

```bash
make install          # Install dependencies + Playwright
make lint             # Run guards + ruff + mypy
make test             # Run pytest with coverage
make check            # Run lint + test
make sniff            # Run WebSocket sniffer
make probe            # Run input probe
make teleport-probe   # Run teleport timing probes (safe + aggressive)
make enemy-teleport-probe  # Run enemy-directed teleport probe
make fuel-probe       # Run fuel pickup probe
make bot              # Run bot client
make decode           # Decode captured session
make discover         # Run command discovery
```

Bot terminal logging is documented in [docs/bot-logging.md](docs/bot-logging.md).

### Quality Gates

All code must pass:

1. **Guard Scripts**: No `Any`, no `cast`, no `type: ignore`, no mocks, no weak assertions
2. **Ruff**: Linting and formatting
3. **Mypy**: Strict type checking (src, tests, scripts)
4. **Pytest**: 100% statement and branch coverage (src, scripts)

### Running Tests

```bash
# Run all tests (parallel via xdist)
make test

# Run specific test file
poetry run pytest tests/bot/ai/test_tactics.py -v

# Run with coverage report
poetry run pytest --cov=src --cov=scripts --cov-branch --cov-report=html
```

---

## Project Structure

```
TankpitBot/
├── src/tankpit_bot/
│   ├── __init__.py           # Package exports
│   ├── _pillow.py            # Typed Pillow image adapter
│   ├── _test_hooks.py        # Dependency injection hooks
│   ├── combat.py             # Combat event tracking
│   ├── decoder.py            # Session decoder for captured data
│   ├── game_state.py         # Game state management
│   ├── inventory.py          # Inventory tracking
│   ├── parser.py             # Lobby message parser (room list, etc.)
│   ├── probe.py              # Input injection and command discovery
│   ├── state_decoder.py      # Game state message decoder
│   ├── terrain.py            # Terrain/map decoding
│   │
│   ├── action_lab/           # Live protocol-level action probes
│   │   ├── teleport.py       # Teleport timing probe
│   │   ├── enemy_teleport.py # Enemy-directed teleport probe
│   │   ├── fuel_probe.py     # Fuel pickup probe
│   │   ├── capture.py        # Session capture helpers
│   │   ├── session.py        # Probe session management
│   │   └── types.py          # Probe type definitions
│   │
│   ├── replay/               # Offline bot decision replay
│   │   ├── engine.py         # Session replay engine
│   │   └── types.py          # Replay trace types
│   │
│   ├── bot/                  # Bot client package
│   │   ├── __init__.py       # Re-exports Bot and all AI types
│   │   ├── ai_strategy.py    # Durable owner selection orchestrator
│   │   ├── base.py           # Bot class (state machine, CDP, commands)
│   │   ├── commands.py       # Command encoding helpers
│   │   ├── executor.py       # Command validation and dispatch
│   │   ├── states.py         # Execution state machine + in-flight actions
│   │   ├── tick_loop.py      # Sync -> decide -> execute orchestrator
│   │   ├── tick_loop_types.py# Tick decision types
│   │   ├── types.py          # Command TypedDicts
│   │   ├── vision.py         # Vision/world-state fallback helpers
│   │   ├── world_sync.py     # CDP buffer drain into decoder pipeline
│   │   └── ai/                      # AI decision modules
│   │       ├── __init__.py          # Re-exports
│   │       ├── types.py             # AI config/state/behavior types
│   │       ├── modes.py             # HFSM mode/substate literals + validation
│   │       ├── mode_controller.py   # Entry/exit rules, substate derivation
│   │       ├── context.py           # DecideCtx and shared helpers
│   │       ├── hunt_mode.py         # Durable HUNT owner
│   │       ├── recover_fuel_mode.py # Durable RECOVER_FUEL owner
│   │       ├── recover_equipment_mode.py # Durable RECOVER_EQUIPMENT owner
│   │       ├── combat_strategy.py   # Combat route primitives
│   │       ├── combat_landing.py    # Shared combat landing helpers
│   │       ├── movement.py          # Walk/teleport/exploration planning
│   │       ├── equipment.py         # Fuel/equipment target selection
│   │       ├── reachability.py      # Viewport-bounded reachability
│   │       ├── resource_search.py   # Shared resource search hop logic
│   │       ├── teleport_cost.py     # Exact teleport fuel cost formula
│   │       ├── threats.py           # Enemy analysis from world state
│   │       ├── pathfinding.py       # Terrain-aware path helpers
│   │       └── tactics.py           # Equipment/radar helper logic
│   │
│   ├── browser/              # Browser automation package
│   │   ├── __init__.py
│   │   ├── dom_scraper.py    # DOM scraping for game log
│   │   ├── fuel_probe.py     # Fuel bar probing
│   │   ├── key_discovery.py  # Key binding discovery
│   │   ├── login.py          # Guest/account login logic
│   │   ├── session.py        # BrowserSession base class
│   │   └── types.py          # Browser-specific types
│   │
│   ├── capture/              # WebSocket capture package
│   │   ├── __init__.py
│   │   ├── signature.py      # Message signature extraction
│   │   ├── stats.py          # Capture statistics
│   │   ├── summary.py        # Session summary generation
│   │   ├── xor.py            # XOR utilities for capture
│   │   └── trackers/         # Message trackers
│   │       ├── combat.py     # Combat event tracker
│   │       ├── container.py  # Container message tracker
│   │       ├── equipment.py  # Equipment tracker
│   │       ├── fuel.py       # Fuel deposit tracker
│   │       ├── items.py      # Item pickup tracker
│   │       ├── mine.py       # Mine event tracker
│   │       ├── position.py   # Position tracker
│   │       ├── radar.py      # Radar result tracker
│   │       └── tank.py       # Tank info tracker
│   │
│   ├── container/            # Container decoder package
│   │   ├── __init__.py
│   │   ├── helpers.py        # Container decoding helpers
│   │   ├── identification.py # Length-based identification
│   │   ├── mapper.py         # Container type mapping
│   │   ├── types.py          # Container TypedDicts
│   │   └── decoders/         # Subtype decoders
│   │       ├── combat.py     # Combat container decoder
│   │       ├── misc.py       # Misc container decoders
│   │       ├── position.py   # Position container decoder
│   │       ├── radar.py      # Radar container decoder
│   │       └── tank.py       # Tank container decoder
│   │
│   ├── protocol/             # Protocol encoding package
│   │   ├── __init__.py
│   │   ├── codec.py          # XOR encode/decode with static + session keys
│   │   ├── commands.py       # Command type definitions
│   │   ├── constants.py      # Protocol constants
│   │   ├── framing.py        # 2-byte length framing encode/decode
│   │   ├── helpers.py        # Protocol helpers
│   │   ├── types.py          # Protocol TypedDicts
│   │   └── decoders/         # Message decoders
│   │       ├── combat.py     # Combat message decoder
│   │       ├── misc.py       # Misc message decoders
│   │       ├── movement.py   # Movement message decoder
│   │       ├── radar.py      # Radar message decoder
│   │       ├── resources.py  # Resource message decoder
│   │       ├── tank.py       # Tank message decoder
│   │       ├── text.py       # Text message decoder
│   │       └── world.py      # World/viewport decoder
│   │
│   ├── sniffer/              # WebSocket sniffer package
│   │   ├── __init__.py
│   │   ├── constants.py      # Sniffer constants
│   │   ├── core.py           # Core sniffer logic
│   │   ├── decoders.py       # Sniffer message decoders
│   │   ├── formatters.py     # Output formatters
│   │   ├── player_tracking.py# Player tracking
│   │   ├── trackers.py       # Tracker coordination
│   │   ├── viewport.py       # Viewport handling
│   │   ├── world_state.py           # Core state, accessors, reset
│   │   ├── world_state_combat.py    # Combat hit and kill tracking
│   │   ├── world_state_containers.py# Container and fuel updates
│   │   ├── world_state_dispatch.py  # Protocol message routing
│   │   ├── world_state_inventory.py # Inventory sync/gain/toggle
│   │   ├── world_state_radar.py     # Radar scan and cache promotion
│   │   ├── world_state_tanks.py     # Tank state mutations
│   │   ├── world_state_tiles.py     # Viewport and tile patches
│   │   └── xor.py                   # XOR utilities for sniffer
│   │
│   ├── state/                       # Game state package
│   │   ├── __init__.py
│   │   ├── mutations.py             # Immutable state mutations
│   │   ├── renderer.py              # ASCII state rendering
│   │   ├── types.py                 # State TypedDicts (WorldState, SelfState, etc.)
│   │   └── viewport_geometry.py     # Viewport dimension constants and helpers
│   │
│   └── types/                # Shared types package
│       ├── __init__.py       # Re-exports all types
│       ├── cdp.py            # CDP WebSocket frame types
│       ├── config.py         # SnifferConfig, BotConfig
│       ├── literals.py       # Literal types + validation helpers
│       ├── message.py        # CapturedMessage, WebSocketInfo
│       ├── probe.py          # Probe input/result types
│       └── session.py        # CaptureSession, SessionSummary
│
├── tests/
│   ├── conftest.py           # Test fixtures (FakeEnv, FakeFileSystem)
│   ├── fakes/                # Fake Playwright classes
│   │   ├── base.py           # Core fakes (FakeCDPSession, FakePage)
│   │   ├── bot.py            # Bot-specific fakes
│   │   └── probe.py          # Probe-specific fakes
│   ├── bot/                  # Bot tests
│   │   ├── ai/               # AI behavior tests
│   │   │   ├── test_types.py
│   │   │   ├── test_evaluators.py
│   │   │   ├── test_equipment.py
│   │   │   ├── test_threats.py
│   │   │   ├── test_pathfinding.py
│   │   │   ├── test_tactics.py
│   │   │   ├── test_actions.py
│   │   │   └── test_loop.py
│   │   ├── test_cdp.py       # CDP session + equipment + AI integration
│   │   ├── test_class.py     # Bot initialization
│   │   ├── test_commands.py  # Command encoding
│   │   ├── test_main.py      # Entry point
│   │   ├── test_run.py       # Game loop
│   │   ├── test_state_machine.py
│   │   ├── test_vision.py    # Vision module
│   │   └── test_world_state.py
│   ├── browser/              # Browser session tests
│   ├── capture/              # Capture tracker tests
│   ├── container/            # Container decoder tests
│   ├── game_state/           # Game state tests
│   ├── login/                # Login flow tests
│   ├── probe/                # Probe tests
│   ├── protocol/             # Protocol decoder tests
│   ├── sniffer/              # Sniffer tests
│   │   └── trackers/         # Sniffer tracker tests
│   ├── types/                # Type tests
│   └── world_state/          # World state tests
│
├── scripts/
│   ├── guard.py              # Monorepo guard orchestrator
│   ├── decode.py             # Session decode script
│   ├── teleport_probe.py     # Live teleport probe entry point
│   ├── enemy_teleport_probe.py # Enemy-directed teleport probe
│   ├── fuel_probe.py         # Fuel pickup probe entry point
│   ├── replay_bot.py         # Offline session replay
│   └── _test_hooks.py        # Guard test hooks
│
├── docs/
│   ├── protocol.md           # Protocol documentation
│   ├── protocol_reference.md # Protocol reference
│   ├── decoding_status.md    # Message decoding status + formats
│   └── vision-module-progress.md # Vision module progress
│
├── pyproject.toml            # Poetry + tool config
└── Makefile                  # Development commands
```

---

## Architecture

### Modular Package Design

The codebase is organized into focused packages:

| Package | Purpose |
|---------|---------|
| `bot/` | Bot client, state machine, executor validation, command dispatch |
| `bot/ai/` | Durable HFSM owners (HUNT, RECOVER_FUEL, RECOVER_EQUIPMENT), mode controller, combat, movement, pathfinding |
| `action_lab/` | Live protocol-level action probes (teleport, enemy teleport, fuel) |
| `replay/` | Offline session replay engine and decision trace types |
| `browser/` | Playwright automation, CDP setup, login flows |
| `protocol/` | XOR codec, framing, command encoding |
| `capture/` | Message capture, trackers, statistics |
| `container/` | 0x2E container subtype decoding |
| `sniffer/` | Live WebSocket analysis, entity source tracking, radar geometry |
| `state/` | Game state management, entity source fields, rendering |
| `types/` | Shared TypedDict models and validation |

### AI Decision Architecture

```
┌──────────────────────────────────────────────┐
│  decide() — Durable owner selection          │
│  Select one owner per tick, delegate routing │
└──────────────┬───────────────────────────────┘
               │
     ┌─────────┼──────────────────┐
     ▼         ▼                  ▼
┌──────────┐ ┌──────────────┐ ┌──────────────────┐
│RECOVER_  │ │ RECOVER_     │ │ HUNT             │
│FUEL      │ │ EQUIPMENT    │ │ decide_hunt_mode │
│fuel<=500 │ │ reserve<=brk │ │ default owner    │
└────┬─────┘ └──────┬───────┘ └────────┬─────────┘
     │              │                  │
     ▼              ▼                  ▼
┌──────────────────────────────────────────────┐
│  Owner route: locked target → visible target │
│  → known registry → radar sense → hop/edge   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  executor._is_dispatchable() — World-state   │
│  validation before command dispatch          │
└──────────────────────────────────────────────┘
```

### Shared BrowserSession Base Class

Bot, sniffer, and probe inherit from `browser.BrowserSession` which provides:

- **CDP Setup**: WebSocket event handlers for frame capture
- **WebSocket Prototype Hook**: Captures game's WebSocket instance via `Page.addScriptToEvaluateOnNewDocument`
- **Intel Gathering**:
  - Console listener (filters for WS/Hook/WebSocket keywords)
  - WebSocket URL logging
  - JavaScript WebSocket debug check
  - Script URL logging
- **Magic Key Capture**: Reads `tankpit.magic` for XOR encoding
- **Login Integration**: Guest or account authentication

### Sniffer Flow

```
┌─────────────────┐
│  Playwright     │
│  sync_api       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BrowserSession │  Console listener
│  CDP Handlers   │  Intel gathering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CDP Session    │  Network.enable
│  Event Handlers │  webSocketCreated
└────────┬────────┘  webSocketFrameSent
         │           webSocketFrameReceived
         ▼
┌─────────────────┐
│  CaptureSession │
│  JSON output    │
└─────────────────┘
```

### Probe Flow

```
┌─────────────────┐
│  BrowserSession │  WebSocket prototype hook
│  CDP Handlers   │  Console + intel gathering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  XOR Encoding   │  Static key + session magic
│  Command Build  │  encode_frame(XOR'd bytes)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  WebSocket      │  window.__capturedWS.send()
│  Injection      │  (or fallback to tankpit.ws)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Toggle Keys    │  First press: WS open
│  State Machine  │  Second press: JS close
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ProbeSession   │
│  JSON output    │
└─────────────────┘
```

### Test Hooks Pattern (`_test_hooks.py`)

Dependency injection via hooks for testability without mocks:

```python
# In _test_hooks.py — production code sets hooks to real implementations
def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)

def path_exists(path: Path) -> bool:
    return path.exists()

def read_text(path: Path) -> str:
    return path.read_text()

# Tests replace hooks with fakes — no conditional logic in production code
```

### Type Models (`types/` package)

```python
# types/message.py - Captured WebSocket message
class CapturedMessage(TypedDict):
    timestamp_ms: int
    direction: MessageDirection  # Literal["sent", "received"]
    payload: str
    ws_url: str

# types/session.py - Complete capture session
class CaptureSession(TypedDict):
    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int | None
    base_url: str
    messages: list[CapturedMessage]
    magic: str | None  # XOR key from tankpit.magic
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `playwright` | Browser automation |
| `websockets` | Direct WebSocket client |
| `httpx` | HTTP client |
| `rich` | Console output formatting |
| `python-dotenv` | .env file support |
| `platform-core` | JSON utilities, logging |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`, no `type: ignore`
- **Coverage**: 100% statements and branches (2069 tests)
- **Guard Rules**: Enforced via `scripts/guard.py` (typing, patterns, test-quality, mock-ban)
- **Test Hooks**: Dependency injection for testing, no mocks
- **Immutable State**: All TypedDicts with encode/decode and require_* validation
- **Google-style Docstrings**: Args, Returns, Raises sections

---

## License

Apache-2.0
