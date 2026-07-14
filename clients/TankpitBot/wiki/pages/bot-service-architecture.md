# Bot Service Architecture

The bot service (`tankpit-bot-service`) is the long-running Python process that lets the phone SPA drive live tankpit sessions over HTTP. It hosts an aiohttp server on `127.0.0.1:27100`; the fiesta docker container's nginx proxies `/api/tankbot/*` to it. Landed 2026-07-12 as Phase A of the SPA-driven bot design.

## The three shared primitives

The service runs a single aiohttp event loop on the main thread. Every session runs on a background executor thread (Playwright's sync API must own its own thread). Three primitives cross that boundary:

- **`ModeBridge`** (`service/mode_bridge.py`) — a threadsafe latest-wins slot the aiohttp handler writes into when `POST /mode` arrives. The tick loop drains it at the top of every tick and stamps the value onto `ai_state.manual_mode`.
- **`StatusBus`** (`service/status_bus.py`) — a threadsafe fan-out. The tick loop calls `publish(SessionStatusDict)` after every tick; SSE subscribers on the aiohttp thread wake up and forward the frame. Every subscriber uses latest-wins semantics so a slow SPA never blocks the tick loop.
- **`SessionRunner`** (`service/session_runner.py`) — coordinator for one active game session at a time. Its `start()` blocks the caller for the session's lifetime; `request_stop()` writes the same stop-file sentinel `Bot.run` already polls, so the tick loop needs no new signalling code.

The primitives are constructed once at service boot in `_async_main` and shared by reference with the `Bot` (via the bridge/bus fields introduced in Phase A6) and the aiohttp handlers (via closures inside `make_app`).

## HTTP surface (`service/http_server.py`)

Five routes, all under nginx's `/api/tankbot/*` prefix in production:

| Route | Handler | Response |
|---|---|---|
| `GET  /health` | Cheap liveness probe | `200 ok` |
| `POST /start`  | Offloads `runner.start()` to an executor thread | `202 starting` / `409 session already running` |
| `POST /stop`   | Calls `runner.request_stop()` | `202 stopping` (idempotent) |
| `POST /mode`   | Decodes `ModeCommandDict`, calls `mode_bridge.submit(...)` | `204` on success |
| `GET  /status` | Subscribes to `StatusBus`, streams `SessionStatusDict` frames as SSE `data: <json>` lines | `200 text/event-stream` |

The SSE handler runs `subscriber.next_frame(timeout=15.0)` inside `loop.run_in_executor` so the event loop stays responsive; on timeout it writes a `: heartbeat` SSE comment to keep intermediaries (nginx, cloudflared) from idling the TCP connection out.

## Shared bot-launch config (`bot/config.py`)

Two settings — the tankpit target URL and the guest-vs-account login preference — need to be read the same way from every code path that launches a `Bot`. Both live in `bot/config.py`:

- `DEFAULT_TARGET_URL = "https://tankpit.com/"`
- `resolve_target_url() -> str` — honours `TANKPIT_URL`; empty string treated as unset.
- `resolve_prefer_account() -> bool` — reads `TANKPIT_PREFER_ACCOUNT`; case-insensitive match against `("true", "1", "yes")`.

`bot/entry.py` (`tankpit-bot` — one-shot CLI) and `service/service_main.py` (`tankpit-bot-service` — long-running service) both consume these resolvers. That keeps them in lockstep — a divergence in env-var handling used to be a silent risk when the two code paths carried their own copies. `entry.py` also routes its `.env` loading through `service_hooks.load_dotenv` so tests can stub it the same way.

## Wire types (`service/types.py`, `service/types_codecs.py`)

Every dict that crosses either the HTTP boundary or the cross-thread boundary is a TypedDict with paired `encode_*` / `decode_*` functions and `require_*` validation:

- `ModeCommandDict{ manual_mode: WireMode }` — `POST /mode` payload.
- `LiveStatsDict{ kills, hits, misses, radars_used, teleports }` — SPA stats panel counters.
- `SessionStatusDict{ running, manual_mode, active_mode, active_mode_state, session_started_ms, tick_timestamp_ms, stats }` — SSE frame.

`WireMode = Literal["UNSET", "HUNT", "COLLECT", "AUTO"]` — the SPA vocabulary. `wire_mode_to_manual` translates it to the `AIMode | None` the tick loop's `manual_mode` field accepts. `"AUTO"` maps to `None` (restore auto-arbitration).

## Session lifecycle

1. Service boots via `tankpit-bot-service`. `service_main.main` reads `.env`, calls `service_hooks.serve()` which runs `asyncio.run(_async_main())`.
2. `_async_main` constructs the shared `ModeBridge` / `StatusBus`, builds a `SessionRunner`, publishes one initial `idle_session_status(now)` frame, wires up the aiohttp app, and enters `run_service_forever`.
3. SPA hits `POST /start`. Handler pre-checks `runner.is_running()`; if idle, offloads `runner.start()` to an executor thread.
4. `runner.start()` scrubs any stale stop file, constructs a `Bot` via `service_hooks.build_bot_factory` (which threads the shared bridge/bus into `Bot.__init__`), and calls `bot.run(session_seconds=0, stop_file_path=STOP_FILE)`. That blocks until the tick loop exits.
5. During play: SPA's `POST /mode` submits to the bridge; the tick loop drains it each tick and stamps `ai_state.manual_mode`. SPA's SSE `/status` receives every published frame.
6. SPA hits `POST /stop`. Handler calls `runner.request_stop()`, which writes the stop file. The tick loop's next iteration observes the file and exits gracefully.
7. `bot.run` returns. `runner.start()`'s `finally` clears the state to `idle` and publishes one final `idle_session_status(now)` so the SPA sees "session ended". The service is ready for the next `POST /start`.

## Dependency injection via `service/_test_hooks.py`

Every non-pure operation the service main touches goes through a module-level symbol in `service/_test_hooks.py` assigned to a real implementation at boot:

- `build_site: SiteFactoryProtocol` — production wires the aiohttp `AppRunner` + `TCPSite` pair inside `_AiohttpSite`; tests inject a fake site that never opens a socket.
- `load_dotenv: LoadDotenvProtocol` — production reads the real `.env`; tests replace with a no-op.
- `serve: ServeProtocol` — production drives `asyncio.run(_async_main())`; tests replace to exercise `main`'s `KeyboardInterrupt` branch without a real event loop.
- `build_bot_factory: BotFactoryBuilderProtocol` — production returns a factory that constructs a real `Bot`; tests inject a factory that returns a fake bot.

The pattern is unconditional — the service code always calls the hook directly, never a real function guarded by `if TESTING`. Rationale: keeps the runtime path identical between production and tests.

## Non-service dependencies A8 needed

- **Aiohttp** (`^3.10`) added as a runtime dep.
- **`asyncio_mode = "auto"`** in `pyproject.toml`'s `[tool.pytest.ini_options]` — every `async def test_*` runs as if decorated with `@pytest.mark.asyncio`. Without it, the marker decorator leaks `Any` through mypy's strict rules; with it, async tests type-check clean without a decorator surface.
- **`concurrency = ["greenlet", "thread"]`** in coverage config (added in A5) — needed so cross-thread arcs in `StatusBus` / `ModeBridge` get traced properly.

## Why hooks live in `service/_test_hooks.py`, not top-level `_test_hooks/service.py`

The service package pulls `service/types.py`, which transitively imports `tankpit_bot.bot.ai.modes`, whose package init imports `TerrainMapProtocol` from the top-level `_test_hooks` tree. Locating the service hooks inside the service tree (instead of adding a `_test_hooks/service.py` submodule at top level) keeps the import graph acyclic during `_test_hooks` initialisation. The Karpathy-style scoped pattern — `Services/: _test_hooks.py` — is the direct mitigation.

## Phase B — SPA bot-controls panel (fiesta side)

Landed 2026-07-12. The tankpit profile now mounts a `<section class="bot-panel">` widget above the video, offering Start / Stop / mode buttons (Hunt / Gather / Auto / Halt) and a live stats readout (kills, hits, misses, radars, teleports). The panel is opt-in per profile — every other fiesta profile skips the widget entirely (no DOM cost, no `/api/tankbot/*` traffic).

Files land under `MCPs/fiesta/src/tankbot/`:

- **`types.ts`** — mirror of `service/types.py`. `WireMode` / `AIMode` / `AIModeState` literal unions, `ModeCommand` / `LiveStats` / `SessionStatus` interfaces, and strict `decodeSessionStatus` / `decodeLiveStats` validators (no `any`, no soft fallbacks — an unknown mode literal throws at the SSE seam instead of silently rendering blank).
- **`TankbotHttpClient.ts`** — same constructor-DI shape as `WebrtcHttpClient`. `postStart` / `postStop` / `postMode` throw on non-2xx; `subscribeStatus(onStatus, onError)` returns a dispose function that closes the underlying `EventSource`.
- **`BotController.ts`** — reactive state layer. Owns a single immutable `BotUIState`, publishes changes to observers. `runIntent` uses `.then/.catch` instead of a lexical `try {` block (matches the `no-try-catch-in-core` guard convention used elsewhere in fiesta). Non-Error rejections rethrow — soft coercion to a state message would violate "no fallbacks".
- **`BotControlsView.ts`** — DOM widget. Subscribes to the controller and re-renders on every state change. The Start / Stop pair swaps visibility on `running`; the mode buttons highlight the current `manualMode`; the pending intent greys just the button whose HTTP call is in flight; the SSE-error banner reveals a Reconnect button that calls `controller.reconnect()`.
- **`_test_hooks.ts`** — the same `FetchFn` + `EventSourceFactory` protocols the WebRTC client uses. Production wires them via `productionFetch` + `productionEventSourceFactory` in `production-hooks.ts`; tests pass hand-written fakes.

Wiring (`boot/bot-controls.ts` — excluded from coverage like every other `boot/**` file):

1. `main.ts` calls `wireBotControls(autoLaunchProfile)`.
2. `wireBotControls` no-ops on any profile whose id ≠ `"tankpit"` and throws on drift (tankpit profile active but `#bot-panel-host` missing from the document — a silent no-op would leave the operator without controls).
3. On the tankpit profile: builds a real `TankbotHttpClient`, wraps it in `BotController`, mounts a `BotControlsView` under `#bot-panel-host`, and calls `controller.connect()` to open the SSE stream.

The panel positions itself absolutely at top-center over the video (`.bot-panel-host` in `style.css` — `pointer-events: none` on the host so it doesn't intercept game taps outside the panel bounds). The `?v=` cache-buster on `style.css` bumps to `64` so phones caching the previous stylesheet fetch the new rules.

## Phase C — nginx route + docker rebuild + startup shortcut

Landed 2026-07-12. The last plumbing step that stitches Phases A + B into a working end-to-end flow.

**nginx (`MCPs/fiesta/nginx.conf`)** — new `location /api/tankbot/` block placed before the broader `/api/` block. Uses the same Tailscale-IP literal `proxy_pass` as `/api/webrtc/` (`host.docker.internal` is unreachable under WSL2 mirrored networking — see the nginx.conf history comments). SSE knobs (`proxy_buffering off`, `proxy_read_timeout 24h`) mirror the ICE-stream settings from the `/api/webrtc/` block so the `/status` frame stream flows without intermediary buffering.

**Bot service (`service/service_main.py`)** — `_DEFAULT_HOST` is `"0.0.0.0"`, not `"127.0.0.1"`. The fiesta docker container's nginx reaches the host through the Tailscale IPv4, not loopback, so the aiohttp site has to bind on the Tailscale interface (or, simpler, on every interface). Trust boundary is the machine's LAN + the operator's Tailnet — the same boundary Vibeshine already accepts on 47990.

**Launcher (`make service` in `tankpitbot/`)** — a Makefile target that respawns `poetry run tankpit-bot-service` on crash with a 5-second cooldown. Lives next to `make bot` / `make sniff` / `make run` so the mental model stays "there's one Makefile for everything tankpit-adjacent." The operator opens a terminal, runs `make service`, and leaves the window open. Ctrl+C exits the respawn loop cleanly.

Chose `make service` over a `shell:startup` `.cmd` after weighing both:

| Trade-off | `make service` (chosen) | `shell:startup` .cmd |
|---|---|---|
| Setup friction | zero, works out of repo | copy the .cmd into `shell:startup` once |
| Runs when | operator types `make service` | every login, silently |
| Debuggability | foreground terminal, tail-friendly | hidden background window |
| Discoverability | sits next to `make bot` / `make sniff` | new pattern to remember |
| Respawn on crash | yes (PowerShell `while ($true)` loop) | yes (`.cmd` :loop label) |

The always-on argument (make headed Chromium ready) is a nothing-burger: the service is just an aiohttp listener until the phone POSTs `/start`, so having it running or not costs nothing while idle. The Makefile route wins on discoverability and debuggability.

**Deployment:** `make up-fiesta` from `MCPs/` (which runs `docker compose up -d --no-deps --build fiesta`) is the only step needed to ship an nginx.conf change. The bot service side is `make service` in the tankpitbot repo — no install step at all.

**Idempotency (probe-before-bind)** — a second launch is a no-op, not a crash-loop. The `main()` entry-point calls `service_hooks.probe_existing_instance()` before `serve()`. The default implementation sends an HTTP `GET /health` on `127.0.0.1:27100`; a `200 ok` response is the marker we own end-to-end (any other body means a foreign server on the port, not us) — so we exit 0 with an "already responding" log line. The Makefile's respawn loop treats exit 0 as "graceful, break" and only retries on nonzero, with a 3-consecutive-crash cap. Net effect: double-tap of the phone SERVER button spawns a new terminal, probes the existing service, prints "already responding" and stays open (the user closes it manually) — no port fight, no lockup.

**Phone-driven `SERVER` button** — `profiles/tankpit.json` gained a `menu-button` labeled `SERVER` beside `SNIFF`. Its `runCommand` (`cmd /c start cmd /k "cd ... && make service"`) spawns a new persistent cmd window on the PC running `make service`. Combined with the idempotency check, tapping the button is now safe under any state: service down → new instance boots; service up → new instance exits immediately with the "already running" log.

## What Phase C does NOT do

- No always-on auto-start. The operator runs `make service` (or taps the phone SERVER button) when they want the bot available. If they never do, `/api/tankbot/*` times out from the phone — the failure mode is loud, not subtle.
- No "Stop Server" button. To stop the server itself (not the game session), the operator Ctrl+Cs the `make service` terminal or closes the window. Killing the SERVER from the phone would require a taskkill-by-title hack that is fragile; the trade-off is deliberate.
- No Windows Firewall automation. The first launch of the service will prompt for port 27100; the operator accepts for private networks.

See also: [Coding Standards](coding-standards.md) (the strictness rules Phases A / B / C were written under), [Inheritance Chain](inheritance-chain.md) (how Bot slots into the runner).
