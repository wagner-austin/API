---
title: "Bot Service Architecture"
tags: [architecture, service, spa]
related:
  - "[[coding-standards]]"
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot/service"
  - "src/tankpit_bot/bot/config.py"
  - "src/tankpit_bot/stream"
source_git_blobs:
  "src/tankpit_bot/service": "8fb55988f5c6b233fd3b9ab76b05e3c25ad6872f"
  "src/tankpit_bot/bot/config.py": "0748a03a52204e4b1ddf3aa95612e8a15e19c2e5"
  "src/tankpit_bot/stream": "b24edbdd53f315d07737060e1b11fa8dfe003bbd"
fact_checked: "2026-09-05"
confidence: medium
hubs: [architecture]
---

# Bot Service Architecture

> **STATUS 2026-09-05 — VIDEO CHANGED CLASS. The in-page canvas
> caster, the frame/chunk bus, and the MJPEG relay described below
> were all DELETED.** A streamed bot now runs Chromium HEADED on its
> own Xvfb display inside the container and an ffmpeg owned by the
> bot's run records that display into HLS segments
> (`stream/capture.py`); the files land in
> `runs/bot/<instance>/hls/` and are served as plain files — by the
> child at `/video/{file}` and by the fleet manager at
> `/demo/video/{slot}/{file}` straight off the shared filesystem
> (`service/video_files.py`), no relay and no per-child port dial.
> Capture rides the compositor: nothing about video touches the
> page, the tick loop, or the CDP connection any more, which is the
> failure class the whole 2026-09-04 slideshow session was spent
> inside. Enabled per child by `TANKPIT_STREAM_VIDEO` (fleet compose
> sets it, with `TANKPIT_HEADLESS: "false"`); the display number is
> the child's own service port. Sections below describing the
> caster, `frame_bus`/`chunk_bus`, `/cast`, or an endless `/video`
> response are HISTORY of the replaced design.

> **STATUS 2026-09-03 — READ THIS FIRST. The SPA that this page is
> written around no longer exists, and neither does the console script
> in the sentence below.** Tankpit was decoupled from fiesta in MCPs
> `02cfd967`: no tankpit profile, no bot overlay, no `/api/tankbot/`
> nginx proxy, no `botCommand` / `botVideoUrl` /
> `botServerLaunchCommand`. The `tankpit-bot-service` console script and
> the `make service` target were deleted in `10f97042`.
>
> WHAT IS TRUE NOW. The same `service_main.main()` still runs, and the
> HTTP surface below is still accurate as a route table — what changed
> is who reaches it. Every FLEET CHILD runs this service (spawned by the
> fleet manager, one per instance, on a port from
> `FLEET_CHILD_PORT_BASE`), and the manager relays their video. A PUBLIC
> DEMO reaches exactly three routes (`/demo/fleet`, `/demo/spawn`,
> `/demo/video/{slot}`) through `tankpit-public`, an nginx filter that
> forwards `/demo/` and 404s everything else so the operator surface on
> the same port stays unpublished (MCPs `54925b6d`).[^7]
>
> Sections below marked with a date are kept as history. Where a
> paragraph describes the SPA in the present tense, read it as "was true
> until 2026-09-03".[^7]

The bot service was the long-running Python process that let the phone
SPA drive live tankpit sessions over HTTP. It hosts an aiohttp server on
`127.0.0.1:27100`; the fiesta docker container's nginx proxied
`/api/tankbot/*` to it until 2026-09-03. Landed 2026-07-12 as Phase A of
the SPA-driven bot design.[^1]

## The three shared primitives

The service runs a single aiohttp event loop on the main thread. Every session runs on a background executor thread (Playwright's sync API must own its own thread). Three primitives cross that boundary:[^1]

> **Where these live (2026-08-07).** The three buses moved from
> `service/` to a new `bus/` package, along with the
> `SessionStatusDict` contract. They are cross-thread primitives, not
> HTTP: `frame_bus.py` imports nothing from `tankpit_bot` at all. While
> they sat under `service/`, the tick loop had to import the HTTP
> package to run, and `service` imported `bot` back for the mode
> vocabulary — a cycle whose real shape was three misfiled files
> ([[package-layering]]).

- **`ModeBridge`** (`bus/mode_bridge.py`) — a threadsafe latest-wins slot the aiohttp handler writes into when `POST /mode` arrives. The tick loop drains it at the top of every tick and stamps the value onto `ai_state.manual_mode`.
- **`StatusBus`** (`bus/status_bus.py`) — a threadsafe fan-out. The tick loop calls `publish(SessionStatusDict)` after every tick; SSE subscribers on the aiohttp thread wake up and forward the frame. Every subscriber uses latest-wins semantics so a slow SPA never blocks the tick loop.
- **`SessionRunner`** (`service/session_runner.py`) — coordinator for one active game session at a time. Its `start()` blocks the caller for the session's lifetime; `request_stop()` writes the same stop-file sentinel `Bot.run` already polls, so the tick loop needs no new signalling code.

The primitives are constructed once at service boot in `_async_main` and shared by reference with the `Bot` (via the bridge/bus fields introduced in Phase A6) and the aiohttp handlers (via closures inside `make_app`).[^1]

## HTTP surface (`service/http_server.py`)

Nine routes, all under nginx's `/api/tankbot/*` prefix in production:[^1]

| Route | Handler | Response |
|---|---|---|
| `GET  /health` | Cheap liveness probe | `200 ok` |
| `POST /start`  | Offloads `runner.start()` to an executor thread | `202 starting` / `409 session already running` |
| `POST /stop`   | Calls `runner.request_stop()` | `202 stopping` (idempotent) |
| `POST /mode`   | Decodes `ModeCommandDict`, calls `mode_bridge.submit(...)` | `204` on success |
| `GET  /status` | Subscribes to `StatusBus`, streams `SessionStatusDict` frames as SSE `data: <json>` lines | `200 text/event-stream` |
| `POST /shutdown` | Requests session stop, fires the service shutdown signal | `202 shutting down` |
| `GET  /watch`  | Self-contained phone watch page (`service/watch_page.py`), playing HLS natively or via the vendored hls.js | `200 text/html` |
| `GET  /watch/hls.js` | The wheel's own hls.js build (`tankpit_bot/data/hls.min.js`, pinned 1.5.20) | `200 application/javascript` |
| `GET  /video/{file}` | One HLS file off this session's capture directory (`stream/hls.py::read_hls_file`) | `200`; `503` + `Retry-After` while the encoder warms; `404` for a rotated-out segment, a name outside the grammar, or a session with no stream |

(2026-09-05: `GET /video` as an endless MJPEG response, `GET /frame`,
`POST /cast` and `GET /frames` all left with the canvas-scrape
pipeline.)

The SSE handler runs `subscriber.next_frame(timeout=15.0)` inside `loop.run_in_executor` so the event loop stays responsive; on timeout it writes a `: heartbeat` SSE comment to keep intermediaries (nginx, cloudflared) from idling the TCP connection out. (The MJPEG drain that once mirrored this shape is gone — HLS viewers are discrete file GETs with nothing to keep alive.)[^1]

## Watch surface — tankpit cut loose from fiesta (2026-07-28)

> **HISTORY AS OF 2026-09-05.** Everything below in this section
> describes the canvas-scrape era: the in-page caster
> (`browser/live_view.py`, deleted), the frame bus
> (`bus/frame_bus.py`, deleted), the per-tick demand sync
> (`_sync_live_view_demand`, deleted) and the MJPEG `<img>` watch
> page (rewritten for HLS). The measurements are kept because they
> are the evidence for WHY the class change happened: even fixed and
> tuned, capture that rides the page and delivery that rides one
> endless HTTP response measured 9.65 fps with 12 stalls covering
> 26.7 s of a 45.6 s public-stream window. The replacement records
> the display itself — see the 2026-09-05 status block at the top.

The phone no longer needs the Sunshine/Vibeshine stack to SEE the bot.
The fiesta path streamed a virtual monitor (kernel IDD driver, patched
Sunshine fork) and injected phone input via `SendInput` — which warps
the one real Windows cursor onto the invisible isolated display, where
non-adjacency strands it (the user's "steals my mouse" complaint; see
the fiesta wiki's 2026-07-01 desktop-takeover incident page). The
replacement is wire-only, inside this repo:[^3]

- **`browser/live_view.py`** — the page-push caster (2026-07-29,
  REPLACING the one-day CDP `Page.startScreencast` relay: Chrome
  sends the next screencast frame only after the previous ACK, and
  acks rode the same Playwright thread the tick loop owns, so every
  heavy tick operation stalled the stream for seconds — the user's
  "laggy... seems to freeze" report; measured 0.6 fps idle / 2.8 fps
  bursty in play). An injected interval INSIDE the game page
  composites the client's six stacked canvases
  ([[rendering-pipeline]]: Background/Tanks/Action/Map/Overlay at
  384x256 + the 384x48 Menu strip, DPI-scaled) to one JPEG per frame
  and hands the data URL to the bot over a CDP BINDING
  (`Runtime.addBinding` → `window.__botCastDeliver`; the handler
  decodes and publishes to the frame bus, loud on drift). **~~Law
  (2026-07-29): a game-page fetch to loopback CANNOT deliver the
  frames~~ — WITHDRAWN 2026-09-03, it is false as stated.** Chrome's
  Local Network Access gate does park page → `127.0.0.1` fetches
  behind a permission that cannot be granted at runtime, and they
  hang forever resolving nothing and rejecting nothing (in-page
  telemetry: `ticks=36 posts=0` in 3 s; Playwright 1.57 has no
  `local-network-access` permission). But the gate is a Chromium
  FEATURE and the bot owns its own launch args. Measured, five POSTs
  from a real `https://tankpit.com` page to a loopback listener in
  the fleet container: **0/5 received with default flags, 5/5 with
  `--disable-features=LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessSendPreflights,PrivateNetworkAccessRespectPreflightResults`**.
  The real constraint is "not grantable per page at runtime", not
  "impossible". Nobody had tried turning the feature off, and the law
  was later cited as the reason an entire transport could not be
  reconsidered -- which is what a law stated without its escape hatch
  costs.[^8] The binding channel has no
  gate and no backpressure: the page captures at its configured
  fps — env `TANKPIT_BOT_VIDEO_FPS` (default 30 since 2026-09-03, was 12) /
  `TANKPIT_BOT_VIDEO_QUALITY` (default 0.8) — regardless of what
  the bot thread is doing; frames queued during a tick stall
  burst-deliver and collapse into the latest-wins bus. Measured:
  117 MJPEG parts / 10 s (11.7 fps) with the tank IDLE, ~800 KB/s,
  and the picture is PURE GAME pixels — no page chrome, which also
  delivers the user's "game fullscreen" wish without touching the
  client's fullscreen button.
- **`bus/frame_bus.py`** — `FrameBus`, byte-for-byte the
  `StatusBus` pattern (latest-wins, cache-on-publish, explicit
  unsubscribe) plus a `latest()` accessor for `/frame`. Its
  `subscriber_count()` doubles as the DEMAND signal.
- **`bot/tick_body.py::_sync_live_view_demand`** — runs each tick
  inside the `_tick_once` `TargetClosedError` guard: subscribers > 0
  re-`ensure`s the idempotent caster EVERY tick (the repetition
  self-heals across page navigations, which wipe injected JS); zero
  subscribers stops it. Unwatched sessions (and every `make run` —
  inert default bus) never run the caster.
- **`service/watch_page.py`** — one HTML string served at `/watch`:
  MJPEG `<img>`, SSE stats strip, START/STOP + mode buttons over the
  existing HTTP routes. All URLs are RELATIVE so the page works both
  direct (`:27100/watch`) and behind nginx's `/api/tankbot/` prefix
  strip (`https://tankpit.austinwagner.org/api/tankbot/watch`) — the
  existing proxy block already forwards every subpath unbuffered.
  The 2026-07-28 exact-root redirect to this page lived one day:
  since 2026-07-29 the RETRO SPA is the tankpit front door again
  (next paragraph), and `/api/tankbot/watch` remains as the
  dependency-free fallback view.
- Idle-exit gate (`exit_when_idle`) counts `/video` viewers alongside
  SSE subscribers, so watching keeps the service alive.[^3]

**The SPA is the tankpit UI (2026-07-29, MCPs `95f27215`).** The
fiesta SPA's tankpit profile went stream-less: `"stream": null` +
`"botVideoUrl": "/api/tankbot/video"`. The SPA's bot-overlay binding
drives a `#bot-video` image (same screen box as the WebRTC video)
from live `BotUIState` — MJPEG src attached while a session runs,
detached otherwise. No Vibeshine session is ever created, so no
input data channel exists; every input surface was removed from the
profile (joystick, L/R click bubbles, Q, ALT+F4, `server:start`) and
what remains is watch-and-control over this service's HTTP routes:
START/STOP BOT, the four mode buttons, STOP SERVER, and the stats
strip. `botServerLaunchCommand` stays in the profile for the
games-hub hash-swap cold chain (a Vibeshine session from the games
host CAN still fire `make service` via run_command).[^3]

**THE ENTIRE SPA CONTROL PLANE WAS DELETED 2026-09-03, AND SO WAS
`make service`.** Everything in the paragraph above is history. Tankpit
was decoupled from fiesta in MCPs commit `02cfd967`, which removed
`src/tankbot/` (the HTTP client, controller and overlay view-model),
`boot/bot-overlay.ts`, `profiles/tankpit.json`, the nginx
`location /api/tankbot/` proxy, and every profile field that fed them:
`botCommand`, `botVideoUrl`, `botServerLaunchCommand`, `BOT_COMMANDS`.
There are no bot buttons, no mode buttons, no stats strip and no
`run_command` cold chain, because there is no tankpit profile. The
`make service` target and the `tankpit-bot-service` console script went
with them in `10f97042`.[^7]

**What replaced it.** The service is now reached two ways only. The
FLEET runs it: every fleet child executes `service_main.main()` through
the child bootstrap, which is why the service's own defaults are now
fleet defaults. And a PUBLIC DEMO reaches three narrow routes through
`tankpit-public`, an nginx filter that forwards `/demo/` and 404s
everything else, so the operator surface on the same port is not
published (MCPs `54925b6d`).[^7]

**The idle pin is gone (`ff1ac1be`).** A service session used to submit
`"UNSET"` to the mode bridge before running the bot, pinning the AI to
idle so an operator could release it from the SPA overlay. With the
overlay deleted and the fleet running this entry point for every child,
that pin had no UI left to lift it: bots entered the game and logged
`reason=manual_hold` forever. Sessions now start in auto-arbitration
like `make run`; `POST /mode` still pins deliberately.[^7]

**Always-on service (2026-07-29).** With the SPA's video served by
this process, the phone expects the URL to answer at any hour:
`TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS` (resolver
`service/config.py::resolve_idle_exit_seconds`) overrides the 1800 s
idle window, and `0` disables the self-exit — `exit_when_idle`
returns immediately. **The shell:startup launcher was removed 2026-07-31 —
the service no longer starts at logon and `tankpit.austinwagner.org` only
answers while `make service` is running.** Default behavior (no env) is
unchanged.[^3]

The restore recipe below is DEAD as written (2026-09-03): it invokes
`make service`, a target that no longer exists. Always-on today is
`make up`, which runs the fleet manager from the newest release
snapshot as a container with `restart` policy and a 10-minute drain on
stop. The historical launcher, for the record only[^4]:

```bat
@echo off
cd /d C:\Users\Test\PROJECTS\API\clients\TankpitBot
set TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS=0
start "TankpitBotService" /min cmd /c "make service"
```

It ran `make service` minimized at logon with the idle self-exit
disabled; that target's own respawn loop covered crashes[^4].

No input path exists anywhere in this surface — the buttons are HTTP
POSTs to the bot service; nothing can touch the host mouse, in the
SPA and plain watch page alike.[^3]

**Service sessions now get run artifacts.** The first live watch test
exposed a gap as old as the service itself: only `bot/entry.py`
(`make run`) called `configure_bot_runtime_logging`, so every
phone-driven session ran with UNCONFIGURED logging — INFO lines
dropped, no archive log/events file, no `_index.tsv` scorecard row
(the 2026-07-28 22:31 service session played a full 10-kill
`session_complete` run and left nothing on disk but
`latest.summary.txt`). `SessionRunner.start` now configures the
per-session artifact bundle before constructing the bot, logging
`Session artifacts: <archive path>` as its first line.[^3]

## Shared bot-launch config (`bot/config.py`)

Two settings — the tankpit target URL and the guest-vs-account login preference — need to be read the same way from every code path that launches a `Bot`. Both live in `bot/config.py`:[^1]

- `DEFAULT_TARGET_URL = "https://tankpit.com/"`
- `resolve_target_url() -> str` — honours `TANKPIT_URL`; empty string treated as unset.
- `resolve_prefer_account() -> bool` — reads `TANKPIT_PREFER_ACCOUNT`; case-insensitive match against `("true", "1", "yes")`.

`bot/entry.py` (`tankpit-bot` — one-shot CLI) and `service/service_main.py` (`tankpit-bot-service` — long-running service) both consume these resolvers. That keeps them in lockstep — a divergence in env-var handling used to be a silent risk when the two code paths carried their own copies. `entry.py` also routes its `.env` loading through `service_hooks.load_dotenv` so tests can stub it the same way.[^1]

## Wire types (`service/types.py`, `service/types_codecs.py`)

Every dict that crosses either the HTTP boundary or the cross-thread boundary is a TypedDict with paired `encode_*` / `decode_*` functions and `require_*` validation:[^1]

- `ModeCommandDict{ manual_mode: WireMode }` — `POST /mode` payload.
- `LiveStatsDict{ kills, hits, misses, radars_used, teleports }` — SPA stats panel counters.
- `SessionStatusDict{ running, manual_mode, active_mode, active_mode_state, session_started_ms, tick_timestamp_ms, stats }` — SSE frame.

`WireMode = Literal["UNSET", "HUNT", "COLLECT", "AUTO"]` — the SPA vocabulary. `wire_mode_to_manual` translates it to the `AIMode | None` the tick loop's `manual_mode` field accepts. `"AUTO"` maps to `None` (restore auto-arbitration).[^1]

## Session lifecycle

1. Service boots via `tankpit-bot-service`. `service_main.main` reads `.env`, calls `service_hooks.serve()` which runs `asyncio.run(_async_main())`.
2. `_async_main` constructs the shared `ModeBridge` / `StatusBus`, builds a `SessionRunner`, publishes one initial `idle_session_status(now)` frame, wires up the aiohttp app, and enters `run_service_forever`.
3. SPA hits `POST /start`. Handler pre-checks `runner.is_running()`; if idle, offloads `runner.start()` to an executor thread.
4. `runner.start()` scrubs any stale stop file, constructs a `Bot` via `service_hooks.build_bot_factory` (which threads the shared bridge/bus into `Bot.__init__`), and calls `bot.run(session_seconds=0, stop_file_path=STOP_FILE)`. That blocks until the tick loop exits.
5. During play: SPA's `POST /mode` submits to the bridge; the tick loop drains it each tick and stamps `ai_state.manual_mode`. SPA's SSE `/status` receives every published frame.
6. SPA hits `POST /stop`. Handler calls `runner.request_stop()`, which writes the stop file. The tick loop's next iteration observes the file and exits gracefully.
7. `bot.run` returns. `runner.start()`'s `finally` clears the state to `idle` and publishes one final `idle_session_status(now)` so the SPA sees "session ended". The service is ready for the next `POST /start`.[^1]

## Dependency injection via `service/_test_hooks.py`

Every non-pure operation the service main touches goes through a module-level symbol in `service/_test_hooks.py` assigned to a real implementation at boot:[^1]

- `build_site: SiteFactoryProtocol` — production wires the aiohttp `AppRunner` + `TCPSite` pair inside `_AiohttpSite`; tests inject a fake site that never opens a socket.
- `load_dotenv: LoadDotenvProtocol` — production reads the real `.env`; tests replace with a no-op.
- `serve: ServeProtocol` — production drives `asyncio.run(_async_main())`; tests replace to exercise `main`'s `KeyboardInterrupt` branch without a real event loop.
- `build_bot_factory: BotFactoryBuilderProtocol` — production returns a factory that constructs a real `Bot`; tests inject a factory that returns a fake bot.

The pattern is unconditional — the service code always calls the hook directly, never a real function guarded by `if TESTING`. Rationale: keeps the runtime path identical between production and tests.[^1]

## Non-service dependencies A8 needed

- **Aiohttp** (`^3.10`) added as a runtime dep.
- **`asyncio_mode = "auto"`** in `pyproject.toml`'s `[tool.pytest.ini_options]` — every `async def test_*` runs as if decorated with `@pytest.mark.asyncio`. Without it, the marker decorator leaks `Any` through mypy's strict rules; with it, async tests type-check clean without a decorator surface.
- **`concurrency = ["greenlet", "thread"]`** in coverage config (added in A5) — needed so cross-thread arcs in `StatusBus` / `ModeBridge` get traced properly.[^1]

## Why hooks live in `service/_test_hooks.py`, not top-level `_test_hooks/service.py`

The original reason (2026-07-12): the service package pulled `service/types.py`, which transitively imported `tankpit_bot.bot.ai.modes`, whose package init imported `TerrainMapProtocol` from the top-level `_test_hooks` tree. Locating the service hooks inside the service tree (instead of adding a `_test_hooks/service.py` submodule at top level) kept the import graph acyclic during `_test_hooks` initialisation. The Karpathy-style scoped pattern — `Services/: _test_hooks.py` — is the direct mitigation.[^1]

**That premise is gone (2026-08-07).** `bot/ai/modes.py` no longer exists: it was a pure leaf (only `platform_core` and `typing`) whose position inside `bot/ai` was the single thing forcing `service` to import `bot`, and it is now `types/modes.py`. `service/types.py` today imports exactly one thing from this codebase — `WireMode` from `bus/session_status.py` — and `bot` imports `service` zero times ([[package-layering]]). The scoped hooks file is kept because it is the right shape, not because the cycle still exists; the remaining `service` -> `bot` edges are `service_main.py` importing `bot/config.py` and two `_test_hooks` references, all of which run in one direction only.

## Phase B — SPA bot-controls panel (fiesta side)

**Staleness note (2026-07-23):** the file inventory below is the
2026-07-12 as-built. It has since been reworked — commit `88fc8ae5`
in the MCPs repo replaced `BotControlsView.ts` with
`overlay-viewmodel.ts` and `boot/bot-controls.ts` with
`boot/bot-overlay.ts`; `types.ts`, `TankbotHttpClient.ts`,
`BotController.ts`, and `_test_hooks.ts` survive. The section is
kept as landing history.[^2]

Landed 2026-07-12. The tankpit profile now mounts a `<section class="bot-panel">` widget above the video, offering Start / Stop / mode buttons (Hunt / Gather / Auto / Halt) and a live stats readout (kills, hits, misses, radars, teleports). The panel is opt-in per profile — every other fiesta profile skips the widget entirely (no DOM cost, no `/api/tankbot/*` traffic).[^2]

Files land under `MCPs/fiesta/src/tankbot/`:[^2]

- **`types.ts`** — mirror of `service/types.py`. `WireMode` / `AIMode` / `AIModeState` literal unions, `ModeCommand` / `LiveStats` / `SessionStatus` interfaces, and strict `decodeSessionStatus` / `decodeLiveStats` validators (no `any`, no soft fallbacks — an unknown mode literal throws at the SSE seam instead of silently rendering blank).
- **`TankbotHttpClient.ts`** — same constructor-DI shape as `WebrtcHttpClient`. `postStart` / `postStop` / `postMode` throw on non-2xx; `subscribeStatus(onStatus, onError)` returns a dispose function that closes the underlying `EventSource`.
- **`BotController.ts`** — reactive state layer. Owns a single immutable `BotUIState`, publishes changes to observers. `runIntent` uses `.then/.catch` instead of a lexical `try {` block (matches the `no-try-catch-in-core` guard convention used elsewhere in fiesta). Non-Error rejections rethrow — soft coercion to a state message would violate "no fallbacks".
- **`BotControlsView.ts`** — DOM widget. Subscribes to the controller and re-renders on every state change. The Start / Stop pair swaps visibility on `running`; the mode buttons highlight the current `manualMode`; the pending intent greys just the button whose HTTP call is in flight; the SSE-error banner reveals a Reconnect button that calls `controller.reconnect()`.
- **`_test_hooks.ts`** — the same `FetchFn` + `EventSourceFactory` protocols the WebRTC client uses. Production wires them via `productionFetch` + `productionEventSourceFactory` in `production-hooks.ts`; tests pass hand-written fakes.[^2]

Wiring (`boot/bot-controls.ts` — excluded from coverage like every other `boot/**` file):[^2]

1. `main.ts` calls `wireBotControls(autoLaunchProfile)`.
2. `wireBotControls` no-ops on any profile whose id ≠ `"tankpit"` and throws on drift (tankpit profile active but `#bot-panel-host` missing from the document — a silent no-op would leave the operator without controls).
3. On the tankpit profile: builds a real `TankbotHttpClient`, wraps it in `BotController`, mounts a `BotControlsView` under `#bot-panel-host`, and calls `controller.connect()` to open the SSE stream.

The panel positions itself absolutely at top-center over the video (`.bot-panel-host` in `style.css` — `pointer-events: none` on the host so it doesn't intercept game taps outside the panel bounds). The `?v=` cache-buster on `style.css` bumps to `64` so phones caching the previous stylesheet fetch the new rules.[^2]

## Phase C — nginx route + docker rebuild + startup shortcut

Landed 2026-07-12. The last plumbing step that stitches Phases A + B into a working end-to-end flow.[^2]

**nginx (`MCPs/fiesta/nginx.conf`)** — new `location /api/tankbot/` block placed before the broader `/api/` block. Uses the same Tailscale-IP literal `proxy_pass` as `/api/webrtc/` (`host.docker.internal` is unreachable under WSL2 mirrored networking — see the nginx.conf history comments). SSE knobs (`proxy_buffering off`, `proxy_read_timeout 24h`) mirror the ICE-stream settings from the `/api/webrtc/` block so the `/status` frame stream flows without intermediary buffering.[^2]

**Bot service (`service/service_main.py`)** — `_DEFAULT_HOST` is `"0.0.0.0"`, not `"127.0.0.1"`. The fiesta docker container's nginx reaches the host through the Tailscale IPv4, not loopback, so the aiohttp site has to bind on the Tailscale interface (or, simpler, on every interface). Trust boundary is the machine's LAN + the operator's Tailnet — the same boundary Vibeshine already accepts on 47990.[^2]

**Launcher (`make service` in `tankpitbot/`)** — a Makefile target that respawns `poetry run tankpit-bot-service` on crash with a 5-second cooldown. Lives next to `make bot` / `make sniff` / `make run` so the mental model stays "there's one Makefile for everything tankpit-adjacent." The operator opens a terminal, runs `make service`, and leaves the window open. Ctrl+C exits the respawn loop cleanly.[^2]

> **SUPERSEDED 2026-09-03.** `make service` and the
> `tankpit-bot-service` console script were deleted in `10f97042`: the
> target's only consumer was the fiesta SPA, and the service's other job
> — serving video — is now done by fleet children, which run
> `service_main.main()` through the child bootstrap in
> `service/_test_hooks/processes.py`. The two decisions below are
> preserved as the reasoning of the time. The launcher question is now
> answered by `make up`, which runs the fleet manager from the newest
> release snapshot as a container; crash recovery is the container's
> restart policy rather than a Makefile respawn loop, and `make down`
> is a drain rather than a kill.[^7]

Chose `make service` over a `shell:startup` `.cmd` after weighing both:[^2]

| Trade-off | `make service` (chosen) | `shell:startup` .cmd |
|---|---|---|
| Setup friction | zero, works out of repo | copy the .cmd into `shell:startup` once |
| Runs when | operator types `make service` | every login, silently |
| Debuggability | foreground terminal, tail-friendly | hidden background window |
| Discoverability | sits next to `make bot` / `make sniff` | new pattern to remember |
| Respawn on crash | yes (PowerShell `while ($true)` loop) | yes (`.cmd` :loop label) |

The always-on argument (make headed Chromium ready) is a nothing-burger: the service is just an aiohttp listener until the phone POSTs `/start`, so having it running or not costs nothing while idle. The Makefile route wins on discoverability and debuggability.[^2]

**Deployment:** `make up-fiesta` from `MCPs/` (which runs `docker compose up -d --no-deps --build fiesta`) is the only step needed to ship an nginx.conf change. The bot service side is `make service` in the tankpitbot repo — no install step at all.[^2]

**Idempotency (probe-before-bind)** — a second launch is a no-op, not a crash-loop. The `main()` entry-point calls `service_hooks.probe_existing_instance()` before `serve()`. The default implementation sends an HTTP `GET /health` on `127.0.0.1:27100`; a `200 ok` response is the marker we own end-to-end (any other body means a foreign server on the port, not us) — so we exit 0 with an "already responding" log line. The Makefile's respawn loop treats exit 0 as "graceful, break" and only retries on nonzero, with a 3-consecutive-crash cap. Net effect: double-tap of the phone SERVER button spawns a new terminal, probes the existing service, prints "already responding" and stays open (the user closes it manually) — no port fight, no lockup.[^2]

**Phone-driven `SERVER` button** — `profiles/tankpit.json` gained a `menu-button` labeled `SERVER` beside `SNIFF`. Its `runCommand` (`cmd /c start cmd /k "cd ... && make service"`) spawns a new persistent cmd window on the PC running `make service`. Combined with the idempotency check, tapping the button is now safe under any state: service down → new instance boots; service up → new instance exits immediately with the "already running" log.[^2]

## What Phase C does NOT do

- No always-on auto-start. The operator runs `make service` (or taps the phone SERVER button) when they want the bot available. If they never do, `/api/tankbot/*` times out from the phone — the failure mode is loud, not subtle.
- No "Stop Server" button. To stop the server itself (not the game session), the operator Ctrl+Cs the `make service` terminal or closes the window. Killing the SERVER from the phone would require a taskkill-by-title hack that is fragile; the trade-off is deliberate.
- No Windows Firewall automation. The first launch of the service will prompt for port 27100; the operator accepts for private networks.[^2]

## The fleet manager — the AI-operated control plane (2026-08-06)

`tankpit-fleet` (`service/fleet.py`, run by `make up` in its container or `make dev` on the hot tree since the 2026-09-02 consolidation, default port **27300**, `TANKPIT_FLEET_PORT` overrides) is a separate, coexisting surface: where `make service` serves the phone SPA one session at a time, the fleet manager serves the *operating AI* and spawns any number of bots. User ruling: "the goal is the ai can spin up and maintain and see the bots, not the spa method."[^5]

- **One user-owned process, N bot child processes.** In-process multi-bot is impossible (the world service is a module singleton), and harness background tasks die at ~46 minutes (the 41-kill session's death), so the manager runs in a terminal the operator owns and each bot is a `subprocess.Popen` child running the ordinary `tankpit_bot.bot.entry` main. Per-bot isolation rides entirely on the instance-namespace lift: the child's bootstrap applies `TANKPIT_BOT_INSTANCE`, `TANKPIT_BOT_SESSION_KILLS` / `_SECONDS`, `TANKPIT_ROLE` (always set explicitly from the resolved role — a `TANKPIT_ROLE` lingering in the manager's own environment must never silently re-role the fleet; added 2026-08-20 with the [[fleet-coordination]] roles), and optional `TANKPIT_ACCOUNT` to its own environment from argv (the manager never reads `os.environ` — the child inherits the parent env whole and layers overrides on its own side of the process boundary).
- **HTTP surface**: `GET /bots` (every instance: pid, alive, returncode, bounds, role, room), `GET /accounts` (accounts.json usernames, first is default), `GET /rooms` (the lobby's room selectors, first is the default — see the room note below), `GET /troops` (the four tank colors in wire team-id order), `POST /bots` `{"instance", "account?", "kills?", "seconds?", "role?", "room?", "troop?"}` → 201 (400 malformed, 409 duplicate-live/invalid name/unknown role; `role` empty means fighter — a gatherer is an explicit operator choice), `GET /bots/{instance}/stats` (latest-run digest summary: kills, deaths, rank countdown, duration, clean/crash — computed from the instance's events artifact, works mid-run and on crashes; `{"available": false}` before first events), `POST /bots/{instance}/stop` → writes the instance's `runs/bot/<instance>/STOP` sentinel for a graceful boundary exit with full teardown, `POST /bots/{instance}/restart` → respawns a FINISHED instance with ALL the parameters it had — account, bounds, role AND room (409 while alive — stop first; the room was silently dropped here from 2026-08-26 to 2026-08-28, relocating a restarted World bot to Practice — see the 2026-08-28 entry in `wiki/log.md`), `DELETE /bots/{instance}` → registry removal, 409 while alive (the fleet never silently kills — stop first).
- **Color is a spawn choice (2026-08-28).** `troop` picks WHICH TANK plays: an account holds four tanks per world, one per color, each with its own rank, inventory, fuel and points (awards alone are shared), and a 5-minute cooldown throttles re-entering a world on a different color ([[game-rules]]). The form sends the color NAME; `_child_environment` converts it to the wire's team id through `TROOP_COLOR_NAMES`, whose INDEX is that id — one home for a mapping `validate/shadow_bot_laws.py` used to restate. Empty means the account's own default tank, which is why that option is static markup rather than a served one.
- **Rooms are picked, not typed (2026-08-28).** Every selector on the spawn form is a dropdown: accounts from `GET /accounts`, roles from the fleet-role vocabulary, colors from `GET /troops`, and rooms from `GET /rooms`, whose list is `tankpit_bot.types.rooms.LOBBY_ROOMS` — `Practice` and `World`. The lobby only ever lists those two, and the world's DISPLAY name carries the current map ("World (Desert)"), so the offered selector is the durable PREFIX rather than a name that rotates: `browser/room_join.py::_resolve_room_entry` matches a selector exactly, or as a prefix followed by a space or `(`. The list is a suggestion surface, not a closed set — spawn still accepts an exact room name over the API. Before this the Room field was a free-text box and an operator had to spell "Practice" from memory.[^6]
- **Control page** (`GET /`, `service/fleet_page.py`, added 2026-08-06): one self-contained HTML file — live bot table (state, bounds, kills/deaths/rank/duration from the stats endpoint, 3 s poll), spawn form, per-row stop/restart/remove buttons. A skin over the same JSON API the operating AI uses, never a second control path. Zero fiesta involvement: no nginx, no SSE, no external assets, no Sunshine/Vibeshine anything — browse to `http://127.0.0.1:27300/` on the desktop.
- **The manager's terminal is clean (2026-08-28).** Each child's stdout/stderr is redirected to its own `runs/bot/<instance>/console.log`; until this landed, `Popen` was called with no `stdout`/`stderr` so every bot inherited the manager's console and the `make fleet` window carried N interleaved tick streams, viewport dumps and all, with no instance prefix — duplicating `latest.log` and contradicting the lifecycle-only rule below. A FILE, not `DEVNULL`: the interpreter prints an uncaught traceback to stderr after the bot's file logging is gone, which is why the bad-password run's `latest.log` ends at "Login errors" while its `GameNotJoinedError` traceback lived only on the console. Opened in append mode so a restart cannot erase the previous run's fatal.
- **Telemetry stays on disk.** The manager owns lifecycle only; the AI observes bots exactly as before — `runs/bot/<instance>/latest.log` and `tankpit-run-digest` on the instance's events.
- Seams: `service_hooks.spawn_bot_process` (tests inject a process double) and `service_hooks.run_web_app` (tests exercise `main` without a socket).

See also: [[coding-standards]] (the strictness rules Phases A / B / C were written under), [[inheritance-chain]] (how Bot slots into the runner).

[^1]: code truth on disk, frontmatter-pinned: `src/tankpit_bot/service/` (`session_runner.py`, `http_server.py`, `service_main.py`, `probe.py`, `types.py`/`types_codecs.py`, `_test_hooks.py`) and `src/tankpit_bot/bot/config.py` — file inventory re-verified 2026-07-23; `make service` target at `Makefile:209`; landed via the 2026-07-12 Phase A commits in git history.
[^2]: cross-repo truth in `~/PROJECTS/MCPs`: `fiesta/src/tankbot/` and `fiesta/nginx.conf`; Phase B/C landing commit `6c78deff` ("fiesta: bot-controls SPA panel + /api/tankbot proxy"), later reworked by `88fc8ae5` ("bot-controls view replaced by overlay viewmodel") — see the staleness note; file inventory re-verified against that repo 2026-07-23.
[^3]: code truth on disk, frontmatter-pinned: `src/tankpit_bot/browser/live_view.py`, `src/tankpit_bot/bus/frame_bus.py`, `service/watch_page.py`, `service/http_server.py` (`_add_watch_routes` at `:236`, `_latest_frame_snapshot` at `:378`, `_drain_frame_bus_to_response` at `:407`), `bot/tick_body.py` (`_sync_live_view_demand` at `:421`), `service/session_runner.py` (per-session `configure_bot_runtime_logging`). **Repinned 2026-08-07:** this footnote named `browser/screencast.py` and `bot/tick_loop.py::_sync_screencast_demand`, neither of which exists. The CDP screencast relay was replaced 2026-07-29 by in-page capture over a `Runtime.addBinding` channel (`live_view.py:1-34` records why: the relay shared the tick thread, so Chrome's ACK-gated frame pacing stalled the stream through every map open and teleport, and a loopback HTTP POST is unusable because Chrome's Local Network Access gate hangs the fetch forever). The demand wiring survived the swap unchanged in shape — subscribers on the frame bus call `ensure` every tick, zero subscribers call `stop` — and moved to `tick_body.py` with the tick-loop split. Live proof 2026-07-28: run `runs/bot/bot-20260728-230140.*` (first line `Session artifacts:`, `Screencast started (viewer connected)` / `stopped (no viewers)` bracketing a 3 s `/frame` subscription, `_index.tsv` row) versus the artifactless 22:31 service session (10/10 kills, only `latest.summary.txt` on disk); MJPEG rate measurements 6 vs 28 parts per 10 s (idle vs AUTO). Mouse-stealing diagnosis from the fiesta wiki (`~/PROJECTS/fiesta/wiki`): `arch-virtual-display-headless.md` (SendInput `abs_mouse` path + unfixed offset bug, task #16; isolated virtual display parked non-adjacent) and `hist-2026-07-01-desktop-takeover-incident.md`; nginx prefix-strip proxy `MCPs/fiesta/nginx.conf` `location /api/tankbot/` (forwards all subpaths, `proxy_buffering off`) **-- DELETED 2026-09-03 in MCPs `02cfd967`; the public path is now `tankpit-public` (MCPs `54925b6d`), an nginx filter that forwards only `/demo/` and 404s the operator surface**. SPA-port + always-on truth (2026-07-29): MCPs commit `95f27215` (`fiesta/src/profiles/types.ts` `botVideoUrl`, `fiesta/src/tankbot/overlay-viewmodel.ts::computeBotVideoView`, `fiesta/src/boot/bot-overlay.ts` video glue, `fiesta/profiles/tankpit.json` stream-less rewrite; 782 SPA tests, 100% coverage) **-- every one of those paths was DELETED 2026-09-03 in MCPs `02cfd967`; cited here as the historical record of the SPA-served era, not as live code** and this repo commit `59201238` (`service/config.py::resolve_idle_exit_seconds` (moved out of `bot/config.py` 2026-08-07 — it is a service concern, and while it sat in bot config it was the last function-level import forcing a `bot` -> `service` edge, see [[package-layering]]), `exit_when_idle` disabled branch, startup launcher `tankpit-bot-service.cmd` in shell:startup); deploy curl-verified on port 8091 (SPA at tankpit root, profile serving `stream: null` + `botVideoUrl`, compiled bundle carrying the `bot-video` glue).
[^5]: code truth on disk: `src/tankpit_bot/service/fleet.py` (FleetManager, routes, `resolve_fleet_port`), `service/_test_hooks.py` (`_CHILD_BOOTSTRAP`, `spawn_bot_process`, `run_web_app` seams), `tests/service/test_fleet.py`; `make fleet` target; landed 2026-08-06.
[^6]: code truth on disk: `src/tankpit_bot/types/rooms.py` (`LOBBY_ROOMS`, `DEFAULT_LOBBY_ROOM`), `service/fleet_manager.py::FleetManager.rooms`, `service/fleet_routes.py` (`GET /rooms`), `service/fleet_page.py` (`<select id="room">` + `fillSelect`), `browser/room_join.py::_resolve_room_entry` (prefix match). Ground truth for the two-room lobby is the ROOM_LIST capture in `runs/bot/arterial/bot-20260813-212329.log`: `+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026` and `5=World (Desert)` — the same capture [[game-rules]] cites. Selector-to-resolver coupling is pinned by `tests/login/test_join.py::test_every_offered_room_selector_resolves_against_a_live_lobby`.
[^4]: `Makefile:265-268` — the `service` target (`service: install`) starts the "long-running SPA-driven HTTP + SSE server", listening on `0.0.0.0:27100`. The Startup-folder `.cmd` described here is machine state on the operator's workstation, not a repo artifact, so it is not verifiable from this checkout; only the `make service` entry point it invokes is.
[^7]: this repo, 2026-09-03: `10f97042` deleted the `service` Makefile target and the `tankpit-bot-service` console script; `ff1ac1be` removed the `_mode_bridge.submit("UNSET")` idle pin from `service/session_runner.py`; `dfdbf310` added `service/demo.py` + `service/demo_routes.py` (the three `/demo/` routes). Cross-repo in `~/PROJECTS/MCPs`: `02cfd967` deleted `fiesta/src/tankbot/`, `fiesta/src/boot/bot-overlay.ts`, `fiesta/profiles/tankpit.json`, the `location /api/tankbot/` block in `fiesta/nginx.conf`, and the `botCommand` / `botVideoUrl` / `botServerLaunchCommand` profile fields; `54925b6d` added `tankpit-public/nginx.conf` (forwards `/demo/`, returns 404 otherwise) and the `tankpit.austinwagner.org` ingress. Fleet children run this service via the child bootstrap in `service/_test_hooks/processes.py`.
[^8]: measured 2026-09-03 in the fleet container, `tankpit-fleet:v0.1.0-76a0f62b`: a headless Chromium loaded `https://tankpit.com/` and issued five `fetch` POSTs to a loopback listener, counted server-side. 0 of 5 arrived under default launch args; 5 of 5 arrived when Chromium was launched with the Local Network Access and Private Network Access feature checks disabled. Frame delivery timing from the same session: binding frames arrive at 31.3/s during `page.wait_for_timeout`, which is how `bot/tick_loop.py` L 241 and L 246 actually wait, so a tick does not starve the stream.
