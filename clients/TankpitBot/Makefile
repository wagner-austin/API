SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
# OUTPUT is an explicit override for the sniffer's capture-file path.
# When unset, the sniffer defaults to runs/sniff/latest.capture_session.json
# and ALSO writes a timestamped archive to
# runs/sniff/sniff-YYYYMMDD-HHMMSS.capture_session.json that is never
# overwritten. Reference historical sessions by their archive path.
OUTPUT ?=

.PHONY: help lint test check install sniff teleport-probe teleport-probe-safe teleport-probe-aggressive teleport-probe-full enemy-teleport-probe enemy-teleport-probe-map enemy-teleport-probe-nearest fuel-probe fuel-drill equipment-probe movement-probe combat-probe track bot run analyze analyze-timing queue-probe download-fields decode discover analyze-viewport smoke debug-run

help:
	@Write-Host ""
	@Write-Host "TankpitBot -- available targets" -ForegroundColor Cyan
	@Write-Host "==============================" -ForegroundColor Cyan
	@Write-Host ""
	@Write-Host "Code health:" -ForegroundColor Yellow
	@Write-Host "  make check                       lint + test (the gate)"
	@Write-Host "  make lint                        guard + ruff + mypy"
	@Write-Host "  make test                        pytest + coverage"
	@Write-Host "  make install                     poetry install + playwright chromium"
	@Write-Host ""
	@Write-Host "Live action probes (need browser + accounts.json):" -ForegroundColor Yellow
	@Write-Host "  make movement-probe              walk to 3 targets -- cheapest smoke test"
	@Write-Host "  make teleport-probe-safe         3 teleports, sync_before_teleport strategy"
	@Write-Host "  make teleport-probe-aggressive   3 teleports, immediate_after_map_open"
	@Write-Host "  make teleport-probe              both safe + aggressive"
	@Write-Host "  make enemy-teleport-probe-map    3 enemy teleports via map open"
	@Write-Host "  make enemy-teleport-probe-nearest 3 enemy teleports via nearest"
	@Write-Host "  make fuel-probe                  3 fuel pickups via 9 attempts"
	@Write-Host "  make fuel-drill                  fill tank to 1100 (long-running)"
	@Write-Host "  make equipment-probe             3 equipment pickups via 9 attempts"
	@Write-Host "  make combat-probe                3 combat engagements, 20 shots each"
	@Write-Host "  make track                       enemy tracking probe (wire vs JS truth)"
	@Write-Host "  make bot                         live HFSM bot"
	@Write-Host "  make run                         5-min timed session + scorecard"
	@Write-Host "  make analyze                     issue report + cross-session stats"
	@Write-Host "  make sniff                       capture WebSocket session to disk"
	@Write-Host ""
	@Write-Host "Offline tools:" -ForegroundColor Yellow
	@Write-Host "  make decode                      replay a capture through real decoders"
	@Write-Host "  make discover                    extract commands from JS client"
	@Write-Host "  make analyze-viewport            analyze viewport bounds in captures"
	@Write-Host ""

install:
	@Write-Host ""
	@Write-Host "==> install" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	poetry run playwright install chromium
	@Write-Host "[install] done" -ForegroundColor Green

lint:
	@Write-Host ""
	@Write-Host "==> lint" -ForegroundColor Cyan
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry install --with dev
	@Write-Host ""
	@Write-Host "  -> guard" -ForegroundColor DarkCyan
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] guard failed" -ForegroundColor Red; exit $$LASTEXITCODE } }
	@Write-Host ""
	@Write-Host "  -> undecoded-field check" -ForegroundColor DarkCyan
	poetry run tankpit-check-undecoded-fields; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] undecoded-field check failed" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host ""
	@Write-Host "  -> ruff check + format" -ForegroundColor DarkCyan
	poetry run ruff check . --fix
	poetry run ruff format .
	@Write-Host ""
	@Write-Host "  -> mypy" -ForegroundColor DarkCyan
	poetry run mypy src tests scripts
	@Write-Host ""
	@Write-Host "[lint] passed" -ForegroundColor Green

test:
	@Write-Host ""
	@Write-Host "==> test" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; poetry run python -m pytest -n auto -v @covArgs
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

# Diagnoses wire-presence vs JS-truth divergence. Fires one shot at the
# closest enemy, then samples per-tank belief at 1Hz for 120s. Writes
# enemy_tracking_probe.json. Override duration with
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

run: install
	@Write-Host ""
	@Write-Host "==> run (5-min timed session)" -ForegroundColor Cyan
	$$env:TANKPIT_BOT_SESSION_SECONDS = '300'; poetry run tankpit-bot
	@Write-Host ""
	@Write-Host "==> scorecard" -ForegroundColor Cyan
	poetry run tankpit-issue-report runs/bot/latest.events.jsonl

analyze:
	@Write-Host ""
	@Write-Host "==> analyze (latest run)" -ForegroundColor Cyan
	poetry run tankpit-issue-report runs/bot/latest.events.jsonl
	@Write-Host ""
	poetry run tankpit-stats

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

discover:
	@Write-Host ""
	@Write-Host "==> discover commands" -ForegroundColor Cyan
	poetry run python discover_commands.py

analyze-viewport:
	@Write-Host ""
	@Write-Host "==> analyze viewport" -ForegroundColor Cyan
	poetry run python -m scripts.analyze_viewport
