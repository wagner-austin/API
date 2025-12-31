# TankpitBot

Automated bot client for Tankpit.com browser game. Uses Playwright and Chrome DevTools Protocol (CDP) to capture and reverse-engineer the game's WebSocket protocol.

## Current Status

**Phase 1: Protocol Discovery** (complete)
- Sniffer captures WebSocket traffic via Playwright CDP
- Auto guest login with rate-limit detection
- Account login fallback when guest limit reached
- Outputs structured JSON for analysis

**Phase 1.5: Automated Protocol Probe** (complete)
- Programmatic input injection via CDP
- Tests keyboard and mouse inputs
- Correlates inputs with server messages
- Discovers which inputs generate protocol commands

**Phase 2: Protocol Analysis** (complete)
- WebSocket endpoint: `wss://dorothy.tankpit.com/ws/`
- Message format: 2-byte header + pipe-delimited fields
- Documented: AUTH, ROOM_LIST, SELECT, JOIN_CONFIRM messages
- Both guest and authenticated flows captured
- See `docs/protocol.md` for details

**Phase 3: Bot Implementation** (next)
- WebSocket client speaking the discovered protocol
- Game logic and AI strategy

## Features

- **Protocol Discovery**: Captures WebSocket traffic via Playwright CDP integration
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
make install
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
1. Join a game as guest
2. Inject keyboard inputs (WASD, arrows, numbers)
3. Inject mouse clicks at various positions
4. Record which inputs generate server messages
5. Save results to `probe_session.json`

### Run the Bot

```bash
make bot
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_URL` | `https://tankpit.com/play` | Target URL for sniffer |
| `TANKPIT_OUTPUT` | `capture_session.json` | Output file path |
| `TANKPIT_HEADLESS` | `true` | Run browser headlessly |
| `TANKPIT_DURATION_MS` | `60000` | Capture duration in ms |
| `TANKPIT_USERNAME` | (none) | Account username for login |
| `TANKPIT_PASSWORD` | (none) | Account password for login |
| `PYTHONUTF8` | `1` | Windows console UTF-8 support |

**Note**: Guest accounts are rate-limited per IP. If you hit the limit, the sniffer will automatically attempt to log in using `TANKPIT_USERNAME` and `TANKPIT_PASSWORD` if configured.

## Development

### Commands

```bash
make install  # Install dependencies + Playwright
make lint     # Run guards + ruff + mypy
make test     # Run pytest with coverage
make check    # Run lint + test
make sniff    # Run WebSocket sniffer
make probe    # Run input probe
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

## Project Structure

```
TankpitBot/
├── src/tankpit_bot/
│   ├── __init__.py       # Package exports
│   ├── _test_hooks.py    # Dependency injection hooks
│   ├── types.py          # TypedDict models
│   ├── login.py          # Shared guest/account login logic
│   ├── sniffer.py        # WebSocket capture via Playwright
│   ├── probe.py          # Input injection and command discovery
│   └── bot.py            # Bot client entry point
├── tests/
│   ├── conftest.py           # Test fixtures (FakeEnv, FakeFileSystem)
│   ├── fakes.py              # Fake Playwright classes for testing
│   ├── test_types.py         # Type encode/decode tests
│   ├── test_login.py         # Login flow tests
│   ├── test_sniffer.py       # Sniffer tests with fake Playwright
│   ├── test_probe.py         # Probe tests with fake CDP
│   ├── test_bot.py           # Bot entry point tests
│   ├── test_test_hooks.py    # Hook function tests
│   └── test_guard_checks.py  # Guard script tests
├── scripts/
│   ├── guard.py          # Monorepo guard orchestrator
│   └── _test_hooks.py    # Guard test hooks
├── docs/
│   └── protocol.md       # Protocol documentation
├── pyproject.toml        # Poetry + tool config
└── Makefile              # Development commands
```

## Architecture

### Sniffer Flow

```
┌─────────────────┐
│  Playwright     │
│  sync_api       │
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

### Type Models

```python
# Captured WebSocket message
class CapturedMessage(TypedDict):
    timestamp_ms: int
    direction: MessageDirection  # Literal["sent", "received"]
    payload: str
    ws_url: str

# Complete capture session
class CaptureSession(TypedDict):
    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int | None
    base_url: str
    messages: list[CapturedMessage]

# CDP WebSocket frame event
class CDPWebSocketFrameEvent(TypedDict):
    requestId: str
    timestamp: float
    response: CDPWebSocketFrame
```

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

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Test Hooks**: Dependency injection for testing

## License

Apache-2.0
