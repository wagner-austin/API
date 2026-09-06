SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
# OUTPUT is an explicit override for the sniffer's capture-file path.
# When unset, the sniffer defaults to runs/sniff/latest.capture_session.json
# and ALSO writes a timestamped archive to
# runs/sniff/sniff-YYYYMMDD-HHMMSS.capture_session.json that is never
# overwritten. Reference historical sessions by their archive path.
OUTPUT ?=

.PHONY: help lint test check install digest sniff teleport-probe teleport-probe-safe teleport-probe-aggressive teleport-probe-full enemy-teleport-probe enemy-teleport-probe-map enemy-teleport-probe-nearest bot-watch respawn-watch key-probe radar-watch density-probe viewport-probe fuel-probe fuel-drill equipment-probe movement-probe combat-probe cadence-probe weave-probe track bot run up down dev analyze analyze-timing queue-probe download-fields decode analyze-viewport analyze-recipient-policy analyze-response-shapes analyze-command-coverage sim-baseline smoke debug-run audit wiki-anchors roundtrip sim-run sim-run-practice shadow

help:
	@Write-Host ""
	@Write-Host "TankpitBot -- available targets" -ForegroundColor Cyan
	@Write-Host "==============================" -ForegroundColor Cyan
	@Write-Host ""
	@Write-Host "Code health:" -ForegroundColor Yellow
	@Write-Host "  make check                       lint + test (the gate)"
	@Write-Host "  make lint                        guard + ruff + mypy"
	@Write-Host "  make test                        pytest + coverage"
	@Write-Host "  make audit                       re-derive wiki physics claims from the runs archive"
	@Write-Host "  make corpus-audit                diff every archived run's analyzers against its wire receipts"
	@Write-Host "  make wiki-anchors                report wiki pages owed an audit (never gates)"
	@Write-Host "  make sim-run                     production bot vs the simulator on real terrain (free soak)"
	@Write-Host "  make sim-run-practice            production bot vs the CERTIFIED practice-bot roster"
	@Write-Host "  make install                     poetry install + playwright chromium"
	@Write-Host ""
	@Write-Host "Live action probes (need browser + accounts.json):" -ForegroundColor Yellow
	@Write-Host "  make movement-probe              walk to 3 targets -- cheapest smoke test"
	@Write-Host "  make teleport-probe-safe         3 teleports, sync_before_teleport strategy"
	@Write-Host "  make teleport-probe-aggressive   3 teleports, immediate_after_map_open"
	@Write-Host "  make teleport-probe              both safe + aggressive"
	@Write-Host "  make enemy-teleport-probe-map    3 enemy teleports via map open"
	@Write-Host "  make enemy-teleport-probe-nearest 3 enemy teleports via nearest"
	@Write-Host "  make bot-watch                   teleport to a bot, sit idle 10 min, capture"
	@Write-Host "  make respawn-watch               engage bots + map-poll their same-id reactivation"
	@Write-Host "  make key-probe                   press physical keys, map real key bindings"
	@Write-Host "  make radar-watch                 free-radar spawn watch (extras stay stocked)"
	@Write-Host "  make density-probe               budgeted extra-radar density sweep (grid sites)"
	@Write-Host "  make larder-probe                own-tile equipment pickup vs adjacent control"
	@Write-Host "  make mine-landing-probe          teleport onto enemy mines, read the bill"
	@Write-Host "  make viewport-probe              edge walks + boundary moves, autoscroll off/on"
	@Write-Host "  make fuel-probe                  3 fuel pickups via 9 attempts"
	@Write-Host "  make fuel-drill                  fill tank to 1100 (long-running)"
	@Write-Host "  make equipment-probe             3 equipment pickups via 9 attempts"
	@Write-Host "  make combat-probe                3 combat engagements, 20 shots each"
	@Write-Host "  make cadence-probe               server shot-serve rate at 4 spacings"
	@Write-Host "  make weave-probe                 does a move cost the queued shot?"
	@Write-Host "  make track                       enemy tracking probe (wire vs JS truth)"
	@Write-Host "  make bot                         live HFSM bot"
	@Write-Host "  make run                         5-min timed session + scorecard"
	@Write-Host "  make up                          THE fleet command: newest release -> image -> container; page on 27300"
	@Write-Host "  make down                        drain every bot to the lobby, then stop the container"
	@Write-Host "  make dev                         fleet manager on the HOT TREE, foreground (development only)"
	@Write-Host "  make analyze                     issue report + run audit + cross-session stats"
	@Write-Host "  make digest                      compact per-run truth table (works on crashed runs)"
	@Write-Host "  make sniff                       capture WebSocket session to disk"
	@Write-Host ""
	@Write-Host "Offline tools:" -ForegroundColor Yellow
	@Write-Host "  make decode                      replay a capture through real decoders"
	@Write-Host "  make analyze-viewport            analyze viewport bounds in captures"
	@Write-Host "  make sim-baseline                fresh one-generation sim corpus + fidelity diff"
	@Write-Host "  make analyze-command-coverage    does the sim survive every command a REAL client sends?"
	@Write-Host ""

install:
	@Write-Host ""
	@Write-Host "==> install" -ForegroundColor Cyan
	poetry lock
	poetry sync --with dev
	poetry run playwright install chromium
	@Write-Host "[install] done" -ForegroundColor Green

lint:
	@Write-Host ""
	@Write-Host "==> lint" -ForegroundColor Cyan
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry sync --with dev
	@Write-Host ""
	@Write-Host "  -> guard" -ForegroundColor DarkCyan
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] guard failed" -ForegroundColor Red; exit $$LASTEXITCODE } }
	@Write-Host ""
	@Write-Host "  -> undecoded-field check" -ForegroundColor DarkCyan
	poetry run tankpit-check-undecoded-fields; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] undecoded-field check failed" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host ""
	@Write-Host "  -> wiki gates (structure + claim binding)" -ForegroundColor DarkCyan
	poetry run tankpit-check-wiki; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] wiki gates failed" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host ""
	@Write-Host "  -> ruff check + format" -ForegroundColor DarkCyan
	poetry run ruff check . --fix
	poetry run ruff format .
	@Write-Host ""
	@Write-Host "  -> mypy" -ForegroundColor DarkCyan
	poetry run mypy src tests scripts
	@Write-Host ""
	@Write-Host "[lint] passed" -ForegroundColor Green

# Coverage data goes to a per-run path, not the default fixed ``.coverage``.
#
# Coverage writes one SQLite file at a fixed path, and two runs that overlap
# corrupt it. The reader then reports a spurious failure on a run whose tests
# all passed, which reads as a code problem and is not one. Several sessions
# work this repo at once, so overlapping runs are the normal case rather than
# the exception.
#
# Per-run paths cannot collide, so no lock is needed. Fragments land beside
# the combined file: xdist writes ``.coverage-<id>.<host>.<pid>.<rand>`` per
# worker and coverage combines them, so cleanup globs rather than naming one
# file. Under runs/ because that is gitignored (``**/runs/``) and is where
# untracked run output already belongs.
test:
	@Write-Host ""
	@Write-Host "==> test" -ForegroundColor Cyan
	poetry lock
	poetry sync --with dev
	& "..\\..\\scripts\\run-tests.ps1"; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }
	@Write-Host ""
	@Write-Host "[test] passed" -ForegroundColor Green

check: lint | test
	@Write-Host ""
	@Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
	@Write-Host ""

sniff: install
	@Write-Host ""
	@if ('$(OUTPUT)') { Write-Host "==> sniff (output override: $(OUTPUT))" -ForegroundColor Cyan } else { Write-Host "==> sniff (writing to runs/sniff/latest.capture_session.json + timestamped archive)" -ForegroundColor Cyan }
	if ('$(OUTPUT)') { $$env:TANKPIT_OUTPUT = '$(OUTPUT)' } else { Remove-Item Env:TANKPIT_OUTPUT -ErrorAction SilentlyContinue }; poetry run tankpit-sniff

# Live health gate. Run a $(SMOKE_DURATION)-second bot session, then
# assert 5 concrete signals on runs/bot/latest.events.jsonl. Override
# the duration with `make smoke SMOKE_DURATION=60`.
SMOKE_DURATION ?= 30
smoke: install
	@Write-Host ""
	@Write-Host "==> smoke (duration: $(SMOKE_DURATION)s)" -ForegroundColor Cyan
	$$env:TANKPIT_BOT_SESSION_SECONDS = '$(SMOKE_DURATION)'; poetry run tankpit-bot
	@Write-Host ""
	@Write-Host "==> smoke assertions" -ForegroundColor Cyan
	poetry run tankpit-smoke

# Live debug session. Same as `make smoke` but skips the assertions
# and chains into `make analyze` so you can read the issue report.
DEBUG_DURATION ?= 30
debug-run: install
	@Write-Host ""
	@Write-Host "==> debug-run (duration: $(DEBUG_DURATION)s)" -ForegroundColor Cyan
	$$env:TANKPIT_BOT_SESSION_SECONDS = '$(DEBUG_DURATION)'; poetry run tankpit-bot
	@Write-Host ""
	@Write-Host "==> debug-run analyze" -ForegroundColor Cyan
	poetry run tankpit-issue-report runs/bot/latest.events.jsonl

teleport-probe: teleport-probe-safe teleport-probe-aggressive
	@Write-Host ""
	@Write-Host "=== TELEPORT PROBES COMPLETE ===" -ForegroundColor Green

teleport-probe-safe: install
	@Write-Host ""
	@Write-Host "==> teleport-probe-safe (strategy: sync_before_teleport)" -ForegroundColor Cyan
	poetry run tankpit-teleport-probe --max-targets 3 --teleport-strategy sync_before_teleport

teleport-probe-aggressive: install
	@Write-Host ""
	@Write-Host "==> teleport-probe-aggressive (strategy: immediate_after_map_open)" -ForegroundColor Cyan
	poetry run tankpit-teleport-probe --max-targets 3 --teleport-strategy immediate_after_map_open

teleport-probe-full: install
	@Write-Host ""
	@Write-Host "==> teleport-probe-full" -ForegroundColor Cyan
	poetry run tankpit-teleport-probe

enemy-teleport-probe: install
	@Write-Host ""
	@Write-Host "==> enemy-teleport-probe (strategy: map_open)" -ForegroundColor Cyan
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy map_open

enemy-teleport-probe-map: install
	@Write-Host ""
	@Write-Host "==> enemy-teleport-probe (strategy: map_open)" -ForegroundColor Cyan
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy map_open

enemy-teleport-probe-nearest: install
	@Write-Host ""
	@Write-Host "==> enemy-teleport-probe (strategy: nearest_enemy)" -ForegroundColor Cyan
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy nearest_enemy

# The bot-watch capture: one teleport to the nearest practice bot,
# then sit IDLE next to it for 10 minutes while the wire records.
# One session feeds three open questions at once: the idle self-sync
# trigger, fine-grained in-viewport bot behavior, and whether an
# untouched bot's fuel ever moves. Analyze afterwards with
# analysis_scripts/mine_bot_policy.py and `make shadow`.
bot-watch: install
	@Write-Host ""
	@Write-Host "==> bot-watch (1 enemy teleport + 10 min heartbeat dwell)" -ForegroundColor Cyan
	$$env:TANKPIT_ENEMY_TELEPORT_SETTLE_MS = "600000"; $$env:TANKPIT_ENEMY_TELEPORT_HEARTBEAT_MS = "1500"; $$env:TANKPIT_ENEMY_TELEPORT_PROBE_OUTPUT = "runs/probe/bot_watch_probe.json"; poetry run tankpit-enemy-teleport-probe --max-attempts 1 --acquisition-strategy map_open

respawn-watch: install
	@Write-Host ""
	@Write-Host "==> respawn-watch (engage adjacent bots, map-poll for same-id reactivation)" -ForegroundColor Cyan
	poetry run tankpit-respawn-watch-probe

radar-watch: install
	@Write-Host ""
	@Write-Host "==> radar-watch (account, extras toggled OFF; free 5x5 scans for spawn law)" -ForegroundColor Cyan
	poetry run tankpit-radar-watch-probe

density-probe: install
	@Write-Host ""
	@Write-Host "==> density-probe (account; budgeted extra-radar sweep of grid-spread sites)" -ForegroundColor Cyan
	poetry run tankpit-density-probe

larder-probe: install
	@Write-Host ""
	@Write-Host "==> larder-probe (account; own-tile equipment pickup vs adjacent control)" -ForegroundColor Cyan
	poetry run tankpit-larder-probe

mine-landing-probe: install
	@Write-Host ""
	@Write-Host "==> mine-landing-probe (account; teleport onto enemy mines, read the bill)" -ForegroundColor Cyan
	poetry run tankpit-mine-landing-probe

viewport-probe: install
	@Write-Host ""
	@Write-Host "==> viewport-probe (account; edge walks + boundary moves, autoscroll off/on, restored)" -ForegroundColor Cyan
	poetry run tankpit-viewport-probe

key-probe: install
	@Write-Host ""
	@Write-Host "==> key-probe (press physical keys as a guest; map the real key bindings)" -ForegroundColor Cyan
	poetry run tankpit-key-probe

fuel-probe: install
	@Write-Host ""
	@Write-Host "==> fuel-probe (target 3 pickups, max 9 attempts)" -ForegroundColor Cyan
	poetry run tankpit-fuel-probe --target-pickups 3 --max-attempts 9

fuel-drill: install
	@Write-Host ""
	@Write-Host "==> fuel-drill (target 3 fills to 1100, max 18 attempts)" -ForegroundColor Cyan
	poetry run tankpit-fuel-drill --target-fills 3 --max-attempts 18 --fuel-full-threshold 1100

equipment-probe: install
	@Write-Host ""
	@Write-Host "==> equipment-probe (target 3 pickups, max 9 attempts)" -ForegroundColor Cyan
	poetry run tankpit-equipment-probe --target-pickups 3 --max-attempts 9

movement-probe: install
	@Write-Host ""
	@Write-Host "==> movement-probe (3 targets)" -ForegroundColor Cyan
	poetry run tankpit-movement-probe --max-targets 3

combat-probe: install
	@Write-Host ""
	@Write-Host "==> combat-probe (3 engagements, 20 shots each)" -ForegroundColor Cyan
	poetry run tankpit-combat-probe --max-engagements 3 --max-shots 20

# Measures the server's shot serve rate: one burst per spacing, served
# shots counted from 0x49 ammo snapshots. Writes runs/probe/cadence-<stamp>.json.
cadence-probe: install
	@Write-Host ""
	@Write-Host "==> cadence-probe (bursts at 2000/1000/500/250 ms)" -ForegroundColor Cyan
	poetry run tankpit-cadence-probe --spacings 2000,1000,500,250 --shots 6

# Does a move cost the queued shot? Alternating shoot-only and
# shoot+move beats on the 2 s serve grid. Writes runs/probe/weave-<stamp>.json.
weave-probe: install
	@Write-Host ""
	@Write-Host "==> weave-probe (8 beats, even beats also move)" -ForegroundColor Cyan
	poetry run tankpit-weave-probe --beats 8 --bursts 1

# Diagnoses wire-presence vs JS-truth divergence. Fires one shot at the
# closest enemy, then samples per-tank belief at 1Hz for 120s. Writes
# runs/probe/enemy-tracking-<stamp>.json. Override duration with
# `make track TRACK_SAMPLE_DURATION_MS=60000`.
TRACK_SAMPLE_DURATION_MS ?= 120000
TRACK_SAMPLE_INTERVAL_MS ?= 1000
track: install
	@Write-Host ""
	@Write-Host "==> track (enemy tracking probe, duration: $(TRACK_SAMPLE_DURATION_MS)ms)" -ForegroundColor Cyan
	poetry run tankpit-enemy-tracking-probe --sample-duration-ms $(TRACK_SAMPLE_DURATION_MS) --sample-interval-ms $(TRACK_SAMPLE_INTERVAL_MS)

bot: install
	@Write-Host ""
	@Write-Host "==> bot (live HFSM)" -ForegroundColor Cyan
	poetry run tankpit-bot

# The fleet manager is the AI-operated control plane: it spawns
# tankpit-bot child processes (one per instance namespace) and exposes
# spawn/list/stop/remove over plain HTTP so an AI session can drive a
# multi-bot fleet without owning any bot's terminal. Runs in a
# user-owned terminal because harness background tasks are killed
# after ~46 minutes.
# Build a frozen, runnable release snapshot of the bot plus its two
# path-dep libs at the CURRENT COMMIT (uncommitted work is excluded
# by construction -- git archive reads HEAD). The fleet manager and
# live sessions run from the snapshot so hot-tree surgery can never
# crash them (2026-08-28: deleting a runtime-read file killed a live
# session). Versioning is the folder name:
# C:\Users\Test\PROJECTS\tankpit-releases\v<version>-<short-sha>.
# Activate by restarting the fleet manager from the new folder.
release:
	@Write-Host ""
	@Write-Host "==> release (frozen runnable snapshot for the fleet)" -ForegroundColor Cyan
	$$sha = (git rev-parse --short HEAD).Trim(); $$ver = (Select-String -Path pyproject.toml -Pattern 'version = .(\d+\.\d+\.\d+).').Matches[0].Groups[1].Value; $$dest = "C:\Users\Test\PROJECTS\tankpit-releases\v$$ver-$$sha"; if (Test-Path $$dest) { Write-Host "[release] $$dest already exists; nothing to do" -ForegroundColor Yellow } else { New-Item -ItemType Directory -Force $$dest | Out-Null; git -C ..\.. archive --format=zip HEAD clients/TankpitBot libs/platform_core libs/monorepo_guards -o "$$dest\snapshot.zip"; Expand-Archive "$$dest\snapshot.zip" -DestinationPath $$dest; Remove-Item "$$dest\snapshot.zip";Copy-Item accounts.json "$$dest\clients\TankpitBot\"; Copy-Item .env "$$dest\clients\TankpitBot\"; if (Test-Path data/tank_registry.json) { New-Item -ItemType Directory -Force "$$dest/clients/TankpitBot/data" | Out-Null; Copy-Item data/tank_registry.json "$$dest/clients/TankpitBot/data/" }; New-Item -ItemType Directory -Force "$$dest\clients\TankpitBot\runs" | Out-Null; Push-Location "$$dest\clients\TankpitBot"; poetry install --only main; Pop-Location; Write-Host "[release] built $$dest" -ForegroundColor Green; Write-Host "[release] activate: 'make down' then 'make up' -- up builds this release's image on first run" -ForegroundColor DarkGray }

# THE fleet lifecycle is ONE pair (operator order 2026-09-02: "one
# single command we use to start this ish"): ``make up`` runs the
# fleet CONTAINER from the newest release — it resolves the release,
# builds its image if that exact tag does not exist yet
# (tankpit-fleet:v<ver>-<sha>, built FROM the snapshot so the tag
# carries the ladder's sha and a hot-tree build can never masquerade
# as a release), then composes it up with the release's own runs/ and
# accounts.json mounted. ``make down`` is a DRAIN, never a kill:
# docker stop's SIGTERM enters drain_on_interrupt, every bot quits to
# the lobby, and the container exits after the last one lands
# (stop_grace_period 10m — interrupting the wait is safe, the manager
# owns the drain). ``make dev`` is the only other way to run a
# manager: hot tree, foreground, for development — live sessions on
# the hot tree can break under mid-surgery edits, which is the whole
# reason ``up`` runs the frozen snapshot.
#
# The retired host-mode targets (fleet / fleet-dev / up-docker /
# down-docker / image, 2026-09-02): one system, not two. TRANSITION
# NOTE: a container manager cannot adopt HOST bot processes — drain
# any pre-container host fleet (poetry run tankpit-fleet-down from
# its release folder) before the first containerized ``make up``, or
# its spawn records on the shared runs/ volume will read as dead.
#
# Both targets check WHO holds 27300, because Windows loopback has no
# exclusive bind by default: Docker's port proxy binds
# 127.0.0.1:27300 without error while a host-mode manager already
# holds it, and the page silently stays the host manager's (observed
# 2026-09-02 — a leftover host manager answered http://127.0.0.1:27300/
# through a full container up/down cycle). A non-Docker owner means a
# host-mode manager: ``up`` refuses to start behind one, ``down``
# names it after the container stops so "still online" is never a
# mystery.
up:
	@Write-Host ""
	@Write-Host "==> up (the fleet: newest release, containerized)" -ForegroundColor Cyan
	$$own = Get-NetTCPConnection -LocalPort 27300 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($$null -ne $$own) { $$proc = Get-Process -Id $$own.OwningProcess; if ($$proc.ProcessName -notlike "*docker*") { Write-Host "[up] port 27300 is already held by $$($$proc.ProcessName) (pid $$($$own.OwningProcess)) -- a HOST-mode fleet manager, not the container." -ForegroundColor Red; Write-Host "[up] drain it first: poetry run tankpit-fleet-down" -ForegroundColor Red; exit 1 } }; $$rel = Get-ChildItem "C:/Users/Test/PROJECTS/tankpit-releases" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($$null -eq $$rel) { Write-Host "[up] no release found -- run 'make release' first" -ForegroundColor Red; exit 1 }; $$tag = "tankpit-fleet:$$($$rel.Name)"; docker image inspect $$tag 2>&1 | Out-Null; if ($$LASTEXITCODE -ne 0) { Write-Host "  Image $$tag not built yet; building from the snapshot (first run takes minutes)" -ForegroundColor DarkGray; docker build -f (Join-Path $$rel.FullName "clients/TankpitBot/Dockerfile") -t $$tag --build-arg BUILD_REF=$$($$rel.Name) $$rel.FullName; if ($$LASTEXITCODE -ne 0) { Write-Host "[up] image build failed" -ForegroundColor Red; exit 1 } }; $$env:FLEET_IMAGE = $$tag; Write-Host "  Image $$tag; 'make down' to drain and stop." -ForegroundColor DarkGray; Push-Location (Join-Path $$rel.FullName "clients/TankpitBot"); docker compose up -d --no-build fleet; $$code = $$LASTEXITCODE; if ($$code -eq 0) { $$ok = $$false; foreach ($$i in 1..5) { Start-Sleep -Seconds 2; try { Invoke-WebRequest -Uri "http://127.0.0.1:27300/bots" -UseBasicParsing -TimeoutSec 3 | Out-Null; $$ok = $$true; break } catch {} }; if (-not $$ok) { Write-Host "  Page unreachable; the host port proxy is stale (a container created behind a squatting host manager keeps a dead proxy) -- recreating once" -ForegroundColor Yellow; docker compose up -d --force-recreate --no-build fleet; foreach ($$i in 1..5) { Start-Sleep -Seconds 2; try { Invoke-WebRequest -Uri "http://127.0.0.1:27300/bots" -UseBasicParsing -TimeoutSec 3 | Out-Null; $$ok = $$true; break } catch {} } }; if ($$ok) { Write-Host "  Fleet page: http://127.0.0.1:27300/" -ForegroundColor Green } else { Write-Host "[up] container started but http://127.0.0.1:27300/ never answered -- inspect: docker logs tankpitbot-fleet-1" -ForegroundColor Red; $$code = 1 } }; Pop-Location; exit $$code

down:
	@Write-Host ""
	@Write-Host "==> down (drain every bot, then stop the container)" -ForegroundColor Cyan
	$$rel = Get-ChildItem "C:/Users/Test/PROJECTS/tankpit-releases" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($$null -eq $$rel) { Write-Host "[down] no release found -- nothing was ever started from one" -ForegroundColor Red; exit 1 }; Write-Host "  Bots quit to the lobby before the container exits; this waits for them (up to 10m)." -ForegroundColor DarkGray; Push-Location (Join-Path $$rel.FullName "clients/TankpitBot"); docker compose stop fleet; $$code = $$LASTEXITCODE; Pop-Location; $$own = Get-NetTCPConnection -LocalPort 27300 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($$null -ne $$own) { $$proc = Get-Process -Id $$own.OwningProcess; if ($$proc.ProcessName -notlike "*docker*") { Write-Host "[down] the container is stopped, but 27300 is STILL held by $$($$proc.ProcessName) (pid $$($$own.OwningProcess)) -- a HOST-mode manager from before the container era. Drain it: poetry run tankpit-fleet-down" -ForegroundColor Yellow } }; exit $$code

dev: install
	@Write-Host ""
	@Write-Host "==> dev (fleet manager, HOT TREE, foreground -- Ctrl+C drains)" -ForegroundColor Yellow
	@Write-Host "  Live sessions can break under mid-surgery edits; use 'make up' to operate." -ForegroundColor DarkGray
	poetry run tankpit-fleet

# One recipe line on purpose: the scorecard must run whatever the
# session's exit code was -- a failed or force-exited run is EXACTLY
# when the issue report matters -- and the target still exits with
# the bot's code so failures stay visible (operator flag 2026-09-03:
# a teardown wedge exited before the scorecard ever printed).
run: install
	@Write-Host ""
	@Write-Host "==> run (timed session; TANKPIT_BOT_SESSION_SECONDS / _KILLS honored, default 300s)" -ForegroundColor Cyan
	if (-not $$env:TANKPIT_BOT_SESSION_SECONDS) { $$env:TANKPIT_BOT_SESSION_SECONDS = '300' }; poetry run tankpit-bot; $$bot = $$LASTEXITCODE; Write-Host ""; Write-Host "==> scorecard" -ForegroundColor Cyan; poetry run tankpit-issue-report runs/bot/latest.events.jsonl; if ($$bot -ne 0) { exit $$bot }; exit $$LASTEXITCODE

digest:
	@Write-Host ""
	@Write-Host "==> digest (latest run)" -ForegroundColor Cyan
	poetry run tankpit-run-digest runs/bot/latest.events.jsonl

# Diff every archived run's analyzers against its own wire receipts
# (kill/death recounts, beat discipline, radar book, tombstone
# re-aims). Exit 1 when any run is flagged.
corpus-audit:
	@Write-Host ""
	@Write-Host "==> corpus-audit (every archived bot run vs its wire receipts)" -ForegroundColor Cyan
	poetry run tankpit-corpus-audit runs/bot

analyze:
	@Write-Host ""
	@Write-Host "==> analyze (latest run)" -ForegroundColor Cyan
	poetry run tankpit-issue-report runs/bot/latest.events.jsonl
	@Write-Host ""
	poetry run tankpit-engagements runs/bot/latest.events.jsonl
	@Write-Host ""
	poetry run tankpit-forage-economy runs/bot/latest.events.jsonl
	@Write-Host ""
	poetry run tankpit-run-audit runs/bot/latest.events.jsonl
	@Write-Host ""
	poetry run tankpit-stats

audit:
	@Write-Host ""
	@Write-Host "==> audit (re-derive wiki physics claims from the runs archive)" -ForegroundColor Cyan
	poetry run tankpit-audit --stamp; if ($$LASTEXITCODE -ne 0) { Write-Host "[audit] a claim failed re-derivation" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host "[audit] every validated claim re-derived from the archive" -ForegroundColor Green

# Reports which wiki pages are owed an audit. NOT part of `make check`:
# a stale anchor is a to-do marker, not a defect, and gating on it would
# redden the build on every source commit. See wiki/SCHEMA.md.
wiki-anchors:
	@Write-Host ""
	@Write-Host "==> wiki-anchors (which pages have drifted from their audited trees)" -ForegroundColor Cyan
	poetry run tankpit-wiki-anchors $(ARGS)

shadow:
	@Write-Host ""
	@Write-Host "==> shadow (price the sim's laws against the real runs archive)" -ForegroundColor Cyan
	poetry run tankpit-shadow; if ($$LASTEXITCODE -ne 0) { Write-Host "[shadow] a sim law diverges from the archive" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host "[shadow] every sim law priced at archive exactness" -ForegroundColor Green

sim-run:
	@Write-Host ""
	@Write-Host "==> sim-run (production bot vs the simulator on real terrain -- no server, no browser)" -ForegroundColor Cyan
	poetry run tankpit-sim-run; if ($$LASTEXITCODE -ne 0) { Write-Host "[sim-run] session failed" -ForegroundColor Red; exit $$LASTEXITCODE }

sim-run-practice:
	@Write-Host ""
	@Write-Host "==> sim-run-practice (production bot vs the certified practice-bot roster)" -ForegroundColor Cyan
	poetry run tankpit-sim-run --practice; if ($$LASTEXITCODE -ne 0) { Write-Host "[sim-run-practice] session failed" -ForegroundColor Red; exit $$LASTEXITCODE }

roundtrip:
	@Write-Host ""
	@Write-Host "==> roundtrip (encode(decode(x)) == x for every archived binary message)" -ForegroundColor Cyan
	poetry run tankpit-roundtrip; if ($$LASTEXITCODE -ne 0) { Write-Host "[roundtrip] an encoder disagrees with its decoder" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host "[roundtrip] every archived binary message re-encodes byte-identically" -ForegroundColor Green

analyze-timing:
	@Write-Host ""
	@Write-Host "==> analyze-timing (command-response latencies)" -ForegroundColor Cyan
	poetry run python -m scripts.analyze_session_timing

queue-probe: install
	@Write-Host ""
	@Write-Host "==> queue-probe (test server command queue)" -ForegroundColor Cyan
	poetry run python -m scripts.queue_probe

download-fields: install
	@Write-Host ""
	@Write-Host "==> download-fields" -ForegroundColor Cyan
	poetry run python -m scripts.download_fields

decode:
	@Write-Host ""
	@Write-Host "==> decode" -ForegroundColor Cyan
	poetry run python -m scripts.decode

analyze-response-shapes:
	@Write-Host ""
	@Write-Host "==> analyze response shapes (live archive vs sim)" -ForegroundColor Cyan
	poetry run python -m scripts.analyze_response_shapes

sim-baseline:
	@Write-Host ""
	@Write-Host "==> sim-baseline (fresh one-generation sim corpus + fidelity diff)" -ForegroundColor Cyan
	poetry run python -m scripts.build_sim_baseline $(ARGS); if ($$LASTEXITCODE -ne 0) { Write-Host "[sim-baseline] failed" -ForegroundColor Red; exit $$LASTEXITCODE }

analyze-command-coverage:
	@Write-Host ""
	@Write-Host "==> analyze command coverage (does the sim survive what a REAL client sends?)" -ForegroundColor Cyan
	poetry run python -m scripts.analyze_command_coverage; if ($$LASTEXITCODE -ne 0) { Write-Host "[command-coverage] an unmapped command byte would crash a hosted server" -ForegroundColor Red; exit $$LASTEXITCODE }

analyze-recipient-policy:
	@Write-Host ""
	@Write-Host "==> analyze recipient policy" -ForegroundColor Cyan
	poetry run python -m scripts.analyze_recipient_policy

analyze-viewport:
	@Write-Host ""
	@Write-Host "==> analyze viewport" -ForegroundColor Cyan
	poetry run python -m scripts.analyze_viewport
