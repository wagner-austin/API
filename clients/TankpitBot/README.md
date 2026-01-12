# TankpitBot

Automated bot client for Tankpit.com browser game. Uses Playwright and Chrome DevTools Protocol (CDP) to capture and reverse-engineer the game's WebSocket protocol, with XOR codec for message encoding/decoding.

## Features

- **Protocol Discovery**: Captures WebSocket traffic via Playwright CDP integration
- **WebSocket Injection**: Sends commands via captured WebSocket (synthetic JS events don't work)
- **XOR Codec**: Full encode/decode with static + session magic keys
- **Container Decoder**: Length-based message identification for 13+ container subtypes
- **Intel Gathering**: Console listener, WebSocket URLs, JS debug info, script URLs
- **Shared Architecture**: BrowserSession base class for sniffer and probe
- **Type Safety**: mypy strict mode, zero `Any` types, TypedDict models
- **100% Test Coverage**: Statements and branches
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

### Capture the Protocol

```bash
make sniff
```

This will:
1. Launch a Chromium browser
2. Navigate to tankpit.com
3. Capture all WebSocket messages
4. Save the session to `capture_session.json`

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

### Decode Captured Session

```bash
make decode
```

Loads a capture session JSON, extracts the magic key, builds the XOR table, and decodes all command messages.

### Run the Bot

```bash
make bot
```

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
| `TANKPIT_OUTPUT` | `capture_session.json` | Output file path |
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
- See `docs/decoding_status.md` for detailed message formats

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
| 4 | player_list_short | Active players query |
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
make install  # Install dependencies + Playwright
make lint     # Run guards + ruff + mypy
make test     # Run pytest with coverage
make check    # Run lint + test
make sniff    # Run WebSocket sniffer
make probe    # Run input probe
make decode   # Decode captured session
make bot      # Run bot client
```

### Quality Gates

All code must pass:

1. **Guard Scripts**: No `Any`, no `cast`, no `type: ignore`
2. **Ruff**: Linting and formatting
3. **Mypy**: Strict type checking
4. **Pytest**: 100% statement and branch coverage

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_types.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
TankpitBot/
├── src/tankpit_bot/
│   ├── __init__.py           # Package exports
│   ├── _test_hooks.py        # Dependency injection hooks
│   ├── bot.py                # Bot client entry point
│   ├── combat.py             # Combat event tracking
│   ├── decoder.py            # Session decoder for captured data
│   ├── game_state.py         # Game state management
│   ├── inventory.py          # Inventory tracking
│   ├── parser.py             # Lobby message parser (room list, etc.)
│   ├── probe.py              # Input injection and command discovery
│   ├── state_decoder.py      # Game state message decoder
│   ├── terrain.py            # Terrain/map decoding
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
│   │   ├── player_tracking.py # Player tracking
│   │   ├── trackers.py       # Tracker coordination
│   │   ├── viewport.py       # Viewport handling
│   │   ├── world_state.py    # World state management
│   │   └── xor.py            # XOR utilities for sniffer
│   │
│   ├── state/                # Game state package
│   │   ├── __init__.py
│   │   ├── mutations.py      # State mutations
│   │   ├── renderer.py       # State rendering
│   │   └── types.py          # State TypedDicts
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
│   ├── conftest.py           # Test fixtures
│   ├── fakes.py              # Fake Playwright classes
│   ├── capture/              # Capture tracker tests
│   ├── sniffer/              # Sniffer tests
│   └── ...                   # Module tests
│
├── scripts/
│   ├── guard.py              # Monorepo guard orchestrator
│   └── _test_hooks.py        # Guard test hooks
│
├── docs/
│   ├── protocol.md           # Protocol documentation
│   └── decoding_status.md    # Message decoding status + formats
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
| `browser/` | Playwright automation, CDP setup, login flows |
| `protocol/` | XOR codec, framing, command encoding |
| `capture/` | Message capture, trackers, statistics |
| `container/` | 0x2E container subtype decoding |
| `sniffer/` | Live WebSocket analysis and formatting |
| `state/` | Game state management and rendering |
| `types/` | Shared TypedDict models and validation |

### Shared BrowserSession Base Class

Both sniffer and probe inherit from `browser.BrowserSession` which provides:

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
# In _test_hooks.py
def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)

def path_exists(path: Path) -> bool:
    return path.exists()

def read_text(path: Path) -> str:
    return path.read_text()

# Tests can patch these functions directly
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

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Test Hooks**: Dependency injection for testing

---

## License

Apache-2.0
