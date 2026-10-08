---
title: Make Targets
tags: [codebase, cli, tooling]
related:
  - "[[module-map]]"
  - "[[adding-a-probe]]"
source_paths:
  - "Makefile"
  - "scripts/fleet_host.py"
  - "src/tankpit_bot/service/fleet.py"
  - "src/tankpit_bot/service/fleet_config.py"
  - "scripts/fleet_gate.py"
source_git_blobs:
  "Makefile": "d645040986ceba38930aefbab34f836ae5f31475"
  "scripts/fleet_host.py": "86ff7450c3473940c899a0c26d9f1e3d4a943faf"
  "src/tankpit_bot/service/fleet.py": "bfb7fb5e4d92ae57525d14406b672bf33b6c3452"
  "scripts/fleet_gate.py": "d1a74605621980e9c0662e33f67c142572944ed1"
  "src/tankpit_bot/service/fleet_config.py": "74d5793285f55c76e9498db448378a72f58f739e"
fact_checked: "2026-09-28"
verified: 2026-10-08 (Makefile, fleet_host.py and fleet.py re-read against API HEAD for MCPs board task 3d71a8e1; check, lint, execution, the up gate and five new targets corrected or added)
confidence: high
hubs: [codebase]
---

# Make Targets

On Windows Make runs every recipe in PowerShell, elsewhere in `/bin/sh`; the choice lives in the monorepo's shared include, not here. From the terminal, just run `make <target>`.[^1]

## Code health (safe, offline)

| Target | What it does |
|--------|-------------|
| `make check` | **the gate**; run before every commit. Under maketools' time budget (`check-budget`) it runs lint's prep steps, then the guard, mypy and the test suite side by side (`concurrent`), so it is charged the longest of the three rather than their sum (board task 28e47ae3)[^6] |
| `make lint` | venv check + `poetry lock`/`sync` + undecoded-field check + **wiki gates** (`tankpit-check-wiki`: page structure + physics-claim binding) + ruff check `--fix`/format, then the guard (`maketools guard`, the monorepo's shared rules: typing, mock-ban and the rest) + mypy strict over `src tests scripts`[^5] |
| `make test` | pytest + branch coverage (`fail_under = 100`) |
| `make execution` | the one live demo case `make check` skips (it presses the public spawn button on austinwagner.org/tankpit, MCPs board task 46934cd6): `maketools test --host-execution --no-cov`, run by the fleet as `clients/TankpitBot-execution` on a Windows node with ffmpeg; fails with `HOST_EXECUTION_INCOMPLETE` when the case is skipped or none ran[^6] |
| `make install` | `poetry lock` + `poetry sync --with dev` + `playwright install chromium`[^5] |
| `make wiki-anchors` | report which wiki pages have drifted from the trees they were audited against (`tankpit-wiki-anchors`; `ARGS=--all` lists current anchors too, `ARGS=--exit-code` exits 1 on drift). **Never gates** — a stale anchor is a to-do marker, not a defect, so `make check` does not depend on it. |
| `make help` | print every target with a one-line description |

## Live bot (needs browser + accounts.json, touches live server)

| Target | What it does |
|--------|-------------|
| `make bot` | Run the HFSM bot indefinitely (no timeout) |
| `make run` | Timed session + scorecard. Honors a pre-set `TANKPIT_BOT_SESSION_SECONDS` (default 300) and `TANKPIT_BOT_SESSION_KILLS` (wind-down at the Nth kill). Sessions > 120 s end themselves cleanly (`session_complete`): finish the live fight, top off, quit — [[bot-behavior-contract]] §1.2 wind-down. The issue report on the run follows.[^5] |
| `make sim-run` | Production bot vs the simulator on real field01 terrain — no server, no browser, no fuel spent. Artifacts: `runs/probe/latest.sim.*` + `runs/sim/sim-<stamp>.capture_session.json` (standard CaptureSession — `tankpit-audit --runs-dir` can price it). `tankpit-sim-run --rounds N --no-opponent` for variants. See [[physics-module-roadmap]]. |
| `make sim-field` | Several production bots on one sim field, two in the arena by default (`tankpit-sim-run --clients 2 $(ARGS)`; `ARGS=--practice`, `--rounds N`). See [[multiplayer-field]]. |
| `make sim-control` | The N=1 byte-identity control: record a manifest, or compare a BEFORE and AFTER (`scripts/sim_control.py`, `ARGS` passed through). |
| `make sim-run-practice` | Production bot vs a REAL practice room (2026-07-25 rework): a stamp-selected mined layout seeds the full 36-bot roster (ids 500-535, 9/team) at archive-observed positions plus the client's real join spawn, on a static container field (~620-dot exposure atlas at the live ~40% hold rate + measured hidden population (840 fuel, half drained + 180 equipment); no runtime spawning — the respawn law was falsified). Bots driven by the certified `sim/bot_policy`. The fidelity soak: 150/150 rounds sustainably, kills across the map, exposure law 18/18 on the sim's own capture. `tankpit-sim-run --practice`. |
| `make sniff` | WebSocket capture to disk — also the human-session recorder (you play, it records). `OUTPUT=<path>` overrides the capture file location. The former `make play` alias was removed 2026-07-01 (identical command). |
| `make release` | Freeze a runnable snapshot of the bot and its two path-dependency libs at HEAD (`git archive`, so uncommitted work is excluded) into `C:\Users\Test\PROJECTS\tankpit-releases\v<version>-<short-sha>` on the hub, with the checkout's `accounts.json`, `.env` and `data/tank_registry.json` copied in; `make up` runs the newest one. Windows hub only.[^5] |
| `make up` / `make down` / `make dev` | THE fleet lifecycle (consolidated 2026-09-02 by operator order — one command, one system). Since 2026-09-28 `up` and `down` run on the hub and act on SEDONA, the game host, through `scripts/fleet_host.py`.[^4] `up`: newest release → stage the committed compose file and `edge/nginx.conf` plus the release's `.env` + `accounts.json` into `C:/fleet/tankpit` on sedona → build its image there if missing (`tankpit-fleet:v<ver>-<sha>`) → run the fleet CONTAINER (manager + N bot children, page on sedona's `127.0.0.1:27300`[^3]) with its public edge → wait for `https://tankpit.austinwagner.org` to serve → since 2026-09-28 the bring-up gate, `scripts/fleet_gate.py`'s `gate`, which spawns one bounded Practice bot and waits for its first tick, so an `up` that exits 0 has been seen playing the game as well as serving. `down`: SIGTERM drain on sedona, every bot to the lobby, 10 m grace. `dev`: hot-tree foreground manager, development only. Replaced `make fleet`/`fleet-dev` and the host-mode detached pair — see [[fleet-lifecycle]] operator surface for the transition note. Also replaced **`make service`**, the standalone SPA-driven HTTP+SSE server on `0.0.0.0:27100`, deleted 2026-09-03 in `10f97042` along with the config flag that only it justified; `src/tankpit_bot/service/` itself remains and now serves the fleet. |
| `make smoke` | Shortest live join-and-quit check: a bot session of `SMOKE_DURATION` seconds (default 30), then its assertions (`tankpit-smoke`, `scripts/smoke.py`)[^5] |
| `make debug-run` | A bot session of `DEBUG_DURATION` seconds (default 30), then the issue report on it[^5] |

## Live probes (need browser + accounts.json, touches live server)

| Target | What it does |
|--------|-------------|
| `make movement-probe` | Walk to 3 targets — cheapest smoke test |
| `make teleport-probe` | Both safe + aggressive teleport strategies |
| `make teleport-probe-safe` | 3 teleports with sync_before_teleport |
| `make teleport-probe-aggressive` | 3 teleports with immediate_after_map_open |
| `make fuel-probe` | 3 fuel pickups via 9 attempts |
| `make fuel-drill` | Fill tank to 1100 (long-running) |
| `make equipment-probe` | 3 equipment pickups via 9 attempts |
| `make combat-probe` | 3 combat engagements, 20 shots each |
| `make track` | Enemy tracking probe — wire-derived positions vs the JS client's truth. Knobs: `TANKPIT_ENEMY_TRACKING_*`. |
| `make enemy-teleport-probe` | 3 enemy-directed teleports; `-map` and `-nearest` variants pin the acquisition strategy |
| `make teleport-probe-full` | The full teleport probe sweep (all targets, no strategy split) |
| `make larder-probe` | Own-tile equipment pickup vs adjacent control — the probe gating [[larder-plan]]. Knobs: `TANKPIT_LARDER_*`. |
| `make mine-landing-probe` | Teleport onto enemy mines and read the fuel bill off the wire. Knobs: `TANKPIT_MINE_LANDING_*`. |
| `make queue-probe` | Test multi-command batching against server |
| `make cadence-probe` | Server shot-serve rate at four spacings (2000, 1000, 500, 250 ms), six shots each (`tankpit-cadence-probe`)[^5] |
| `make weave-probe` | Does a move cost the queued shot? Eight beats, one burst (`tankpit-weave-probe`)[^5] |
| `make respawn-watch` | Teleport adjacent to a bot, fire a single at its registry position every 2 s for up to 30 s (kills already-damaged bots; full-fuel ones teleport off at 7-8 hits either way), then map-poll every 2 s for 60 s so the 0x4C snapshots pin the same-id reactivation tick and tile. Up to 4 targets per session. Analysis is offline from the capture (0x41 kill vs 0x58 flee). Knobs: `TANKPIT_RESPAWN_WATCH_*`. |
| `make key-probe` | Press each safe physical key once (own capture window per press) and attribute sent frames to keys. Settled R=radar 2026-07-24. |
| `make radar-watch` | Stationary spawn-law watch on the account: slot-5 extras toggled off (verified via wire state; stock preserved), free built-in 5×5 scan per 15 s + free map open per 30 s + 1-tile walk shuffle per beat (immune to the ~12-min never-playing disconnect). |
| `make density-probe` | Budgeted extra-radar density sweep (2026-07-25): teleport a 4×4 map-spread site grid, map-open before every hop, verify each landing before spending an extra, one full-viewport scan per landed site. Funds itself (viewport pickups → dot hops → blind dot-walks), aborts + quits to lobby when marooned, restores the slot-5 enable state, archives per run under `runs/probe/density-<stamp>`. Knobs: `TANKPIT_DENSITY_*`. Run-5 measurement in [[game-economy]]. |
| `make bot-watch` | Teleport adjacent to a practice bot, then dwell 10 min at a 1.5 s walk-shuffle heartbeat (2026-07-24: query heartbeats were falsified — only real gameplay actions hold the push stream open, ~40 fuel/min; each beat drains the CDP buffer then walks 1 tile). See [[server-push-gating]] for the law and the seven-run proof. |
| `make viewport-probe` | Autoscroll/viewport law probe (2026-07-25): normalize autoscroll to OFF via wire-verified 'a' presses (plaintext `A0`/`A1` acks), then per phase (OFF, ON) anchor-teleport with landing verification, walk terrain-routed steps to the window's east edge, attempt one crossing step, and fire long boundary moves. Quits to lobby on success AND abort. Measured the edge-recentering + acceptance-boundary laws in [[viewport-shift-protocol]]. Knobs: `TANKPIT_VIEWPORT_*`. |

## Offline analysis (safe, reads capture files)

| Target | What it does |
|--------|-------------|
| `make analyze` | Issue report + engagements + forage economy + run audit (deterministic verdicts + capture replay diff) + cross-session stats on latest run. The forage-economy section (`tankpit-forage-economy`, 2026-07-26) answers "where did the time go": hunt/collect split, forage viewports per kill, pickups per viewport, weapons per equipment pickup, hop selected/declined breakdown. Pass two events paths to diff runs — built for the 803 s vs 1,187 s 10-kill pair whose deciding number (weapons/pickup 3.34 vs 2.14) was invisible to the issue report. |
| `make digest` | Compact per-run truth table for the latest run (`tankpit-run-digest runs/bot/latest.events.jsonl`), then the cross-session stats (`tankpit-stats`).[^5] Works on CRASHED runs, where `make analyze` cannot complete — that is the reason it exists separately. Added 2026-08-05. |
| `make audit` | Re-derive every validated wiki physics claim from the full runs archive (`tankpit-audit --stamp` rewrites `fact_checked:` on green pages). See [[physics-module-roadmap]] Phase 2. |
| `make shadow` | Price the SIM's laws against the archive — every validator imports its predictor from sim source (sync cadence, grant invariants, kill mercy bundle, corpse window). A failure = sim and real server disagree. See [[physics-module-roadmap]]. |
| `make roundtrip` | Decode→encode→decode every archive message; byte-identity proof for the sim's encoders |
| `make analyze-timing` | Command-response timing analysis |
| `make decode` | Replay a capture through real decoders |
| ~~`make discover`~~ | Extracted command constants from the JS client. Retired 2026-08 in `48cda6bd` with the other 43 ungated one-shot scripts (board task f0c3a532) |
| `make analyze-viewport` | Analyze viewport bounds in captures |
| `make corpus-audit` | Diff every archived run's analyzers against its wire receipts (`tankpit-corpus-audit runs/bot`)[^5] |
| `make sim-baseline` | Fresh one-generation sim corpus plus a fidelity diff (`scripts/build_sim_baseline.py`, `ARGS` passed through)[^5] |
| `make analyze-command-coverage` | Does the sim survive every command a REAL client sends? (`scripts/analyze_command_coverage.py`)[^5] |
| `make analyze-response-shapes` | Response-shape analysis over the captures (`scripts/analyze_response_shapes.py`)[^5] |
| `make analyze-recipient-policy` | Recipient-policy analysis over the captures (`scripts/analyze_recipient_policy.py`)[^5] |
| `make download-fields` | Fetch the field GIFs the terrain loader decodes |
| `make container-census` | Container sites per field from the archive's radar scans, with the terrain shape fitted per field and scored held out (`tankpit-container-census`). See [[container-census]]. |
| `make conformance-baseline` | Rewrite the conformance ratchet from a fresh replay (`tankpit-conformance --write-baseline`). See [[capture-conformance]]. |

## Output

Bot runs save to `runs/bot/`, sniffer to `runs/sniff/`, probes to `runs/probe/`. Each run writes a stable `latest.*` file **and** a timestamped archive copy (`bot-YYYYMMDD-HHMMSS.*`) — these are independent files, not symlinks, so the archive is never clobbered by the next run.[^2] Full layout in `docs/run-artifacts.md`.

[^1]: `Makefile` line 1, `include ../../scripts/make/shell.mk`, whose `ifeq ($(OS),Windows_NT)` sets `SHELL := powershell.exe` and otherwise `SHELL := /bin/sh`; the monorepo's `lint-makefiles` guard refuses a Makefile that sets `SHELL` itself.
[^2]: `runtime_artifacts.py` — `build_bot_run_artifacts` / `build_sniff_run_artifacts` / `build_probe_run_artifacts` each return both an archive path and a `latest_*_path`; no symlink is created anywhere in the module
[^3]: `src/tankpit_bot/service/fleet.py` — `resolve_fleet_host` returns `TANKPIT_FLEET_HOST` when set, else `FLEET_HOST_DEFAULT = "127.0.0.1"` (`:51`); the manager binds through `service_hooks.build_site(..., host, port)` and logs "tankpit-fleet listening on %s:%d" (`:164`). The fleet CONTAINER sets `TANKPIT_FLEET_HOST=0.0.0.0` so its published port reaches the process, and the loopback property moves to the port mapping `127.0.0.1:27300:27300` in `docker-compose.yml`. The port default is `FLEET_PORT_DEFAULT = 27300` in `src/tankpit_bot/service/fleet_config.py`, overridden by `TANKPIT_FLEET_PORT` and rejected outside `[1024, 65535]`. **Found 2026-08-07:** the Makefile's own `fleet` banner claimed `0.0.0.0:27300`, which was never what the host process bound. The banner was corrected to match the code rather than this page to match the banner.
[^4]: `up` and `down` in `scripts/fleet_host.py`, run by the Makefile's `up` and `down` as `poetry run python -m scripts.fleet_host up|down`; its module docstring lists what is staged on sedona and from where. MCPs board task `077204e8`.
[^5]: `Makefile` § the `lint`, `install`, `release`, `run`, `smoke` (`SMOKE_DURATION ?= 30`), `debug-run` (`DEBUG_DURATION ?= 30`), `analyze`, `digest`, `cadence-probe`, `weave-probe`, `corpus-audit`, `sim-baseline` and `analyze-*` recipes, re-read 2026-09-28; the `lint` row re-read 2026-10-08, when its guard no longer carried the package's own rules: `57fd5ab40` (2026-08-24, repo API) made `scripts/guard.py` the shared shim, the physics-claim and wiki rules now run in `tankpit-check-wiki` (`scripts/wiki_gates.py`), and the contract, layer, protocol-constant, state-sentinel, mine-layer and hook-restore rules in `scripts/` are called from their tests only.
[^6]: `Makefile:130-142` (`check` runs `check-budget`; `_check-unbudgeted` runs `_lint-prep` and then `$(MAKETOOLS) concurrent _lint-guard _lint-mypy _test-suite`, with the board task 28e47ae3 note above it) and `:144-153` (`execution` and its comment), re-read 2026-10-08.
