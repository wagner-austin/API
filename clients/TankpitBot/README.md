# TankpitBot

**System identification against an undocumented network protocol.** Tankpit.com
publishes no wire spec and no mechanics documentation, so this package infers both
and then proves the inferred model is right.

Playwright + Chrome DevTools Protocol capture the live WebSocket; a fully
reverse-engineered XOR wire protocol decodes it; controlled probes measure the
game's laws; and every derived claim is bound to a wiki page that re-derives from
the runs archive on every `make check`. A durable HFSM plays the game — which is
the end-to-end proof that the model holds.

The instrument is the larger half of the project. `action_lab/` runs isolated live
experiments, `facts/` carries provenance and confidence on every observed entity,
`validate/` prices the model against the capture archive, `contracts/` fails at the
offending state transition rather than N observations later, and `sim/` is a server
twin that replays the whole model with no browser and no live server.

Also ships a machine-checked physics layer bound to the wiki and a phone-driven
HTTP + SSE service.

## Wiki first

**`wiki/index.md` is the source of truth** for game mechanics, wire protocol, combat
strategy, and architecture decisions — 6 hubs, 67 content pages. This README is an
orientation layer; when it and the wiki disagree, the wiki wins.

| Hub | Covers |
|-----|--------|
| [Game Mechanics](wiki/hubs/game-mechanics.md) | viewport, teleport, radar, fuel, ferries, map, equipment, walk mechanics |
| [Protocol](wiki/hubs/protocol.md) | wire format, decode coverage, MAP_DATA, viewport shift, push gating |
| [Combat](wiki/hubs/combat.md) | shot range, enemy behavior, weapon selection, gameplay loop, economy |
| [JS Client](wiki/hubs/js-client.md) | reverse-engineered `tpclient.js` — V table, commands, XOR, terrain |
| [Architecture](wiki/hubs/architecture.md) | inheritance chain, DI, freshness model, behavior contract, service |
| [Codebase](wiki/hubs/codebase.md) | [module map](wiki/pages/module-map.md), services, testing patterns, make targets |

## Features

- **Autonomous HFSM bot** — two durable mode owners (`HUNT`, `COLLECT`) with
  rank-derived readiness thresholds, committed collect intents, ferry-aware
  pathfinding, and mine-as-terrain walkability
- **Complete wire coverage** — every message type in the client's V table has exactly
  one decoder reachable from `protocol.decode_message`; encoders round-trip
  byte-identically against the capture archive (`make roundtrip`)
- **Server twin** (`sim/`) — the production bot plays full sessions against a
  simulated server on real terrain, with no browser and no live server
- **Executable physics** (`physics/`) — one symbol per machine-checked wiki claim,
  re-derived from the runs archive on every `make check` and `make audit`
- **Phone-driven service** — long-running aiohttp + SSE server with a live MJPEG
  view, mode pinning, and a self-contained watch page
- **Live diagnostics** — in-page HUD, click-to-flag channel with lead-up snapshots,
  issue reports, ledger audits, and cross-session stats
- **Type safety** — mypy strict, zero `Any`/`cast`/`type: ignore`, immutable TypedDicts
- **100% test coverage** — statements and branches, no mocks, `_test_hooks` DI only

---

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+

### Installation

```bash
cd clients/TankpitBot
poetry install --with dev
```

`make install` does the same plus `playwright install chromium`. Copy
`accounts.json.example` to `accounts.json` and `.env.example` to `.env` before any
live run.

### Run the bot

```bash
make bot     # live HFSM bot, runs until stopped
make run     # 5-min timed session (TANKPIT_BOT_SESSION_SECONDS) + scorecard
make analyze # issue report + forage economy + run audit + cross-session stats
```

Every run writes canonical artifacts plus a timestamped archive copy:

```text
runs/bot/latest.log                     # operator-readable timeline
runs/bot/latest.events.jsonl            # AI / SYNC / STATE / WIRE / WORLD records
runs/bot/latest.capture_session.json    # raw wire capture
```

See [`docs/run-artifacts.md`](docs/run-artifacts.md) for the full layout.

### Run without a live server

```bash
make sim-run           # production bot vs the simulator on real terrain
make sim-run-practice  # production bot vs the certified practice-bot roster
```

Free soak testing — no browser, no server, no account. The sim's laws are priced
against the real capture archive by `make shadow`.

### Capture the protocol

```bash
make sniff
```

Writes `runs/sniff/latest.{log,events.jsonl,capture_session.json,raw_capture.json,session_summary.json}`
plus archives. Setting `TANKPIT_OUTPUT` (or `OUTPUT=`) writes the requested path
*and* mirrors into the canonical paths.

### Live probes

Isolated protocol-level experiments against the real server. Each one answers a
single question without the planner in the way — see
[Adding a Probe](wiki/pages/adding-a-probe.md).

```bash
make movement-probe    # walk to 3 targets — cheapest smoke test
make teleport-probe    # safe + aggressive teleport strategies
make fuel-probe        # 3 fuel pickups via 9 attempts
make equipment-probe   # 3 equipment pickups via 9 attempts
make combat-probe      # 3 engagements, 20 shots each
make radar-watch       # free-radar spawn watch
make larder-probe      # own-tile pickup vs adjacent control
make mine-landing-probe# teleport onto enemy mines, read the bill
```

`make help` lists all of them with one-line descriptions.

### Offline tools

```bash
make decode            # replay a capture through the real decoders
make discover          # extract commands from the JS client
make analyze-viewport  # viewport bounds analysis
make analyze-timing    # command-response latencies

poetry run python -m scripts.replay_bot [session.json] [--json]
```

`replay_bot` re-runs a captured session through the decoders and planner tick by
tick, emitting decision traces with no browser involved.

---

## Bot service

`make service` starts the long-running SPA-driven server on `0.0.0.0:27100`
(nginx proxies `/api/tankbot/*` to it in production). Nine routes:

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness probe |
| `POST /start` / `POST /stop` | session lifecycle (`202`, `409` if already running) |
| `POST /mode` | pin a durable mode (`HUNT`, `COLLECT`, `UNSET`) from the phone |
| `GET /status` | SSE stream of `SessionStatusDict` frames, one per tick |
| `POST /shutdown` | stop the session and the service |
| `GET /watch` | self-contained phone watch page |
| `GET /video` / `GET /frame` | MJPEG live view / one-shot JPEG snapshot |

Three threadsafe primitives cross the aiohttp-loop ↔ tick-loop boundary:
`ModeBridge` (latest-wins mode slot), `StatusBus` (latest-wins fan-out), and
`SessionRunner` (one active session, stop via the same sentinel `Bot.run` polls).
Details in [Bot Service Architecture](wiki/pages/bot-service-architecture.md).

---

## AI behavior system

Tick-based planning over pure functions and immutable TypedDicts, with a durable
HFSM owner model. Each tick: sync world state → select exactly one owner → plan →
execute. AI state persists only after a successful dispatch.

### Durable mode owners

| Mode | Owns the tick when | Substates |
|------|--------------------|-----------|
| `HUNT` | fully stocked and no COLLECT trigger pending | ACQUIRE, REFRESH, CLOSE, SCAN_ON_LANDING, ENGAGE, CONFIRM_KILL |
| `COLLECT` | fuel low, weapon/radar break, or short of full between kills | SENSE, SEARCH, APPROACH, PICKUP, DONE |
| `UNSET` | the SPA has pinned idle | — (holds position, stays armed) |

Thresholds are **rank-derived, not fixed**: `hunt_fuel_floor` is
`fuel_capacity(rank)` (1000 at recruit through 1800 at general), and HUNT entry
requires duals and homings at `inventory_capacity(rank)` with extra radars within 5
of cap. Entry-at-break / exit-at-full gives hysteresis, so the bot rebuilds a full
stock rather than leaving the moment it scrapes together one radar. Nothing bypasses
the readiness gate — the bot never hunts below full stock.

`manual_mode` from the service surface overrides auto-arbitration when set.

### Planner facts worth knowing

- **Walkability has one owner.** `compose_decision_terrain` (`bot/ai/ferry.py`)
  layers the static minimap, wire ferry tiles, and hostile mines into a single
  `is_passable` that every consumer reads. There is no separate executor mine veto.
  See [Terrain Composition](wiki/pages/terrain-composition.md).
- **Collect plans survive the tick boundary.** `bot/ai/intent.py` owns typed
  collect plans with explicit validity and reasoned release (`plan_released` events).
- **Teleports need an open map.** The open and the teleport never share a tick — a
  same-tick pair races the server's open processing and is silently dropped.
- **Teleport cost is exact**: `floor(6 * sqrt(dx² + dy²))`, computed with integer
  sqrt (`physics/costs.py`). The server charges to the *actual* landing tile, so
  drift off the requested target changes the bill.
- **Viewport is 16×16 visible**; the radar patch envelope is 18×18 (one tile of
  margin per side). Direct moves and pickups are allowed on visible edge tiles.
- **`0x5A` is a sparse tile patch**, not a full snapshot — authoritative for
  viewport origin and tile-cache updates, but absence from one patch does not imply
  tank absence.
- **Rank-windowed human targeting**: `TANKPIT_BOT_HUMAN_MIN_RANK`/`_MAX_RANK`
  default to `(1, 8)` — recruits are protected. Practice bots are farmed at any rank.

### Supporting layers

`facts/` carries provenance and confidence for every observed entity. `ledger/`
keeps live fuel, ammo, and per-enemy damage books with divergence verdicts.
`physics/` holds the measured game laws, each symbol bound to a wiki claim.
`diagnostics/` turns the event stream into issue reports, alignment checks, and
scorecards.

Read the [Bot Behavior Contract](wiki/pages/bot-behavior-contract.md) before
proposing any behavior fix — it is a MUST / MUST NOT / verified-by table.

---

## Protocol

Every wire byte has exactly one decoder, reachable from
`protocol.decode_message(msg_type, body)`. The `0x2E` container envelope is handled
by a subtype-first dispatcher that routes tunneled subtypes to their protocol
decoders and leaves only the container-only subtypes (`0x43` ContainerPickup,
`0x45` MineDetonation, `0x4B` MinePlacement, and 1-byte TeleportLanded) to the
container path. No dual paths, no length-based blob fallbacks.

Encoders mirror the decoders: `make roundtrip` asserts `encode(decode(x)) == x` for
every archived binary message.

The per-message-type status table — every V-table type with its JS handler name,
field list, and known gaps — lives in
[Wire Decode Coverage Map](wiki/pages/decode-coverage.md). Protocol prose docs are
in [`docs/`](docs/): `protocol-reference.md`, `protocol-decoding-status.md`,
`protocol-discovery.md`, `protocol-pipeline.md`.

---

## Configuration

Copy `.env.example` to `.env` and customize.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_URL` | `https://tankpit.com/` | Target URL |
| `TANKPIT_ROOM` | `Practice` | Room to join |
| `TANKPIT_HEADLESS` | `false` | Run browser headlessly |
| `TANKPIT_PREFER_ACCOUNT` | `false` | Skip guest login, use account directly |
| `TANKPIT_ACCOUNT` | (none) | Account name or index from `accounts.json` |
| `TANKPIT_USERNAME` / `TANKPIT_PASSWORD` | (none) | Override `accounts.json` |
| `PYTHONUTF8` | `1` | Windows console UTF-8 support |

### Bot

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_BOT_SESSION_SECONDS` | `0` (unbounded) | Session length; `make run` sets `300` |
| `TANKPIT_BOT_SESSION_KILLS` | `0` (no bound) | Stop after N kills |
| `TANKPIT_BOT_HUMAN_MIN_RANK` | `1` | Lowest human rank the bot may engage |
| `TANKPIT_BOT_HUMAN_MAX_RANK` | `8` | Highest human rank the bot may engage |
| `TANKPIT_BOT_PRIORITY_TARGET` | (none) | Account name to hunt preferentially |
| `TANKPIT_BOT_WEAPON_RESUME_SLACK` | `0` | Relax the weapons bar to `cap - slack` |
| `TANKPIT_SHOT_SCREENSHOTS` | (off) | Capture screenshots around shots |

### Service

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS` | `1800` | Idle window before the service exits |
| `TANKPIT_BOT_VIDEO_FPS` | `12.0` | Live-view capture rate |
| `TANKPIT_BOT_VIDEO_QUALITY` | `0.8` | Live-view JPEG quality |

### Sniffer

| Variable | Default | Description |
|----------|---------|-------------|
| `TANKPIT_OUTPUT` | (canonical paths) | Capture output path; also mirrored to `runs/sniff/latest.*` |
| `TANKPIT_DURATION_MS` | `0` | Capture duration in ms (0 = indefinite) |
| `TANKPIT_LIVE_DECODE` | `true` | Show decoded messages in real time |

Each probe reads its own `TANKPIT_<PROBE>_*` variables (output path, timeouts,
attempt counts) — see the probe's module docstring or
[Make Targets](wiki/pages/make-targets.md).

**Note**: guest accounts are rate-limited per IP. On hitting the limit the session
falls back to account credentials from `accounts.json`.

---

## Development

### Gate

```bash
make check   # lint | test — the gate
make lint    # guard + undecoded-field check + ruff + mypy
make test    # pytest with branch coverage
```

`make lint` runs `scripts/guard.py`, which enforces:

1. **Typing rules** — no `Any`, `cast`, `type: ignore`, `TYPE_CHECKING`, `.pyi`, `noqa`
2. **Mock ban** — no mocks, no monkey-patching; `_test_hooks` DI only
3. **Contract rules** — `contracts/` enforcement decorators are present
4. **Physics claims** — every `physics/` symbol re-derives from its bound wiki claim
5. **Undecoded-field check** — no silently dropped wire fields

then ruff (check + format) and mypy strict over `src`, `tests`, `scripts`.

`make test` requires **100% statement and branch coverage** (`fail_under = 100`).

### Archive validators

```bash
make audit      # re-derive every wiki physics claim from the runs archive
make shadow     # price the sim's laws against the real archive
make roundtrip  # encode(decode(x)) == x for every archived binary message
```

### Running tests

```bash
make test                                            # all, parallel via xdist
poetry run pytest tests/bot/ai/test_tactics.py -v     # one file
poetry run pytest --cov=src --cov=scripts --cov-branch --cov-report=html
```

Tests mirror the source layout under `tests/` (359 test files across 24 packages).
Fakes live in `tests/fakes/`; probe tests replay real captured sessions through the
production Playwright stack rather than faking the wire. See
[Testing Patterns](wiki/pages/testing-patterns.md).

---

## Project structure

All source lives under `src/tankpit_bot/`; tests mirror it under `tests/`.
Per-file detail is maintained in [Module Map](wiki/pages/module-map.md) — this table
is the package-level orientation.

| Package | Purpose | Key files |
|---------|---------|-----------|
| `bot/` | The game-playing bot — tick loop, dispatch, executor | `base.py`, `tick_loop.py`, `executor.py`, `ai_strategy.py` |
| `bot/ai/` | All decision logic — mode owners and planners | `mode_controller.py`, `hunt_mode.py`, `collect_mode.py`, `ferry.py`, `intent.py`, `movement.py`, `pathfinding.py` |
| `browser/` | Playwright automation — CDP, login, room join, HUD, live view | `session_base.py`, `lifecycle.py`, `login.py`, `room_join.py`, `overlay_hud.py` |
| `protocol/` | Wire protocol — framing, XOR codec, decoders, encoders | `codec.py`, `framing.py`, `commands.py`, `decoders/`, `encoders/` |
| `container/` | `0x2E` container-only subtypes | `identification.py`, `decoders/`, `encoders.py` |
| `state/` | World state types and mutations | `types/`, `mutations.py`, `viewport_geometry.py` |
| `sniffer/` | Passive WebSocket sniffer and the live world-state machine | `core.py`, `world_service.py`, `world_state_*.py` |
| `capture/` | Post-hoc capture analysis | `stats.py`, `viewport_analysis.py`, `trackers/` |
| `action_lab/` | Live probes — isolated experiments against the real server | probe base, factory, per-probe modules |
| `sim/` | The server twin — laws, world, transport, practice room | `server.py`, `viewport_window.py`, `combat_emissions.py`, `bot_policy.py` |
| `physics/` | Measured game laws, one symbol per machine-checked wiki claim | `costs.py`, `capacity.py`, `damage.py`, `combat.py`, `map.py` |
| `ledger/` | Live bookkeeping — fuel, ammo, per-enemy damage, outcomes | `fuel_book.py`, `ammo_book.py`, `damage_book.py`, `outcome/` |
| `facts/` | Provenance and confidence for observed entities | `fact.py`, `provenance.py`, `tank_facts.py` |
| `validate/` | Archive-priced law validators | `audit.py`, `shadow*.py`, `roundtrip.py`, `wire_timeline.py` |
| `diagnostics/` | Issue reports, alignment checks, scorecards, CLIs | `issue_report.py`, `run_audit.py`, `session_stats.py` |
| `service/` | Phone-driven aiohttp + SSE service | `http_server.py`, `session_runner.py`, `mode_bridge.py`, `status_bus.py` |
| `replay/` | Offline replay of captures through bot decision logic | `engine.py` |
| `contracts/` | Enforcement decorators backing the contract guard rule | `enforcement.py` |
| `types/` | Shared TypedDict models and validation | `cdp.py`, `config.py`, `literals.py`, `session.py` |
| `_test_hooks/` | Protocol interfaces for DI, 8 submodules by domain | `bot.py`, `browser.py`, `cdp.py`, `env.py`, `fs.py` |

Top-level support modules: `decoder.py` / `state_decoder.py` (wire blob decoders),
`parser.py` / `parser_messages.py` (CDP message parsing), `terrain.py` (minimap GIF
loader), `combat.py` / `combat_tracker.py`, `inventory.py`, `runtime_logging.py`,
`runtime_artifacts.py`, `_hooks_guard.py`.

`scripts/` holds standalone entry points reached two ways: the live probes and the
smoke test are registered in `pyproject.toml` under `[tool.poetry.scripts]` as
`tankpit-*` commands, while the guard rule modules (`guard.py`, `contract_rules.py`,
`wiki_rules.py`, `physics_claims.py`) and the offline analysis utilities
(`analyze_*.py`, `decode.py`, `download_fields.py`, `trace_vtable.py`,
`replay_bot.py`, `queue_probe.py`) are invoked through `make` targets instead.

### Dependency flow

```text
bot/ai/ ──→ bot/ ──→ browser/ ──→ protocol/
  │           │         │
  └───────────┴─────────┴──→ state/

action_lab/ ─→ browser/ ──→ protocol/ ──→ state/
sniffer/    ─→ browser/ ──→ protocol/ ──→ state/
capture/    ─→ state/                    (no browser)
replay/     ─→ bot/ai/ + state/          (no browser)
sim/        ─→ bot/ai/ + protocol/       (no browser, no server)
```

Bot, `ProbeBase`, and `BrowserSession` all inherit from `SessionBase`
(`browser/session_base.py`) — see [Inheritance Chain](wiki/pages/inheritance-chain.md)
and [Services](wiki/pages/services.md) for the DI wiring.

### Test hooks pattern

Dependency injection without mocks. Production code sets hooks to real
implementations; tests save, replace with a fake, and restore. Monkey-patching is
banned by `MonkeyPatchBanRule` in the guard.

```python
# _test_hooks/env.py — production default
get_env: Callable[[str], str | None] = lambda key: os.environ.get(key)

# tests swap the attribute, restore in teardown — no conditional logic in prod code
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `playwright` | Browser automation |
| `websockets` | Direct WebSocket client |
| `aiohttp` | Bot service HTTP + SSE server |
| `httpx` | HTTP client |
| `rich` | Console output formatting |
| `python-dotenv` | `.env` support |
| `pillow` | Image processing (minimap terrain, live view) |
| `platform-core` | JSON utilities, logging |
| `monorepo-guards` | Guard rule enforcement |

### Development

`pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-xdist`, `mypy`, `ruff`.

---

## Quality standards

- **Type safety** — mypy strict; no `Any`, `cast`, `type: ignore`, `TYPE_CHECKING`
- **Coverage** — 100% statements and branches, enforced in CI config
- **No mocks, no monkey-patching** — `_test_hooks` DI exclusively
- **No back-compat shims, wrappers, fallbacks, or legacy code**
- **Immutable state** — TypedDicts with encode/decode and `require_*` validation
- **Google-style docstrings** — Args, Returns, Raises
- **Files under 400 lines** where possible

Full list: [Coding Standards](wiki/pages/coding-standards.md).

---

## License

Apache-2.0
</content>
