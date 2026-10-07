include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

# OUTPUT is an explicit override for the sniffer's capture-file path.
# When unset, the sniffer defaults to runs/sniff/latest.capture_session.json
# and ALSO writes a timestamped archive to
# runs/sniff/sniff-YYYYMMDD-HHMMSS.capture_session.json that is never
# overwritten. Reference historical sessions by their archive path.
OUTPUT ?=
CONFORMANCE_BASELINE := wiki/sources/conformance_baseline.json

.PHONY: help lint test check _check-unbudgeted _deps _lint-prep _lint-guard _lint-mypy _test-suite execution install digest sniff teleport-probe teleport-probe-safe teleport-probe-aggressive teleport-probe-full enemy-teleport-probe enemy-teleport-probe-map enemy-teleport-probe-nearest bot-watch respawn-watch key-probe radar-watch density-probe viewport-probe fuel-probe fuel-drill equipment-probe movement-probe combat-probe cadence-probe weave-probe track bot run up down dev analyze analyze-timing queue-probe download-fields decode analyze-viewport analyze-recipient-policy analyze-response-shapes analyze-command-coverage sim-baseline sim-control smoke debug-run audit conformance-baseline container-census wiki-anchors roundtrip sim-run sim-run-practice sim-field shadow larder-probe mine-landing-probe corpus-audit release

help:
	@echo ""
	@echo "TankpitBot -- available targets"
	@echo "=============================="
	@echo ""
	@echo "Code health:"
	@echo "  make check                       lint + test (the gate)"
	@echo "  make lint                        guard + ruff + mypy"
	@echo "  make test                        pytest + coverage"
	@echo "  make audit                       re-derive wiki physics claims, replay captures vs the sim"
	@echo "  make conformance-baseline        rewrite the conformance ratchet from a fresh replay"
	@echo "  make container-census            container sites per field from radar scans, held-out fit"
	@echo "  make corpus-audit                diff every archived run's analyzers against its wire receipts"
	@echo "  make wiki-anchors                report wiki pages owed an audit (never gates)"
	@echo "  make sim-run                     production bot vs the simulator on real terrain (free soak)"
	@echo "  make sim-run-practice            production bot vs the CERTIFIED practice-bot roster"
	@echo "  make sim-field ARGS=...          two production bots on one field (ARGS=--practice, --rounds N)"
	@echo "  make install                     poetry install + playwright chromium"
	@echo ""
	@echo "Live action probes (need browser + accounts.json):"
	@echo "  make movement-probe              walk to 3 targets -- cheapest smoke test"
	@echo "  make teleport-probe-safe         3 teleports, sync_before_teleport strategy"
	@echo "  make teleport-probe-aggressive   3 teleports, immediate_after_map_open"
	@echo "  make teleport-probe              both safe + aggressive"
	@echo "  make enemy-teleport-probe-map    3 enemy teleports via map open"
	@echo "  make enemy-teleport-probe-nearest 3 enemy teleports via nearest"
	@echo "  make bot-watch                   teleport to a bot, sit idle 10 min, capture"
	@echo "  make respawn-watch               engage bots + map-poll their same-id reactivation"
	@echo "  make key-probe                   press physical keys, map real key bindings"
	@echo "  make radar-watch                 free-radar spawn watch (extras stay stocked)"
	@echo "  make density-probe               budgeted extra-radar density sweep (grid sites)"
	@echo "  make larder-probe                own-tile equipment pickup vs adjacent control"
	@echo "  make mine-landing-probe          teleport onto enemy mines, read the bill"
	@echo "  make viewport-probe              edge walks + boundary moves, autoscroll off/on"
	@echo "  make fuel-probe                  3 fuel pickups via 9 attempts"
	@echo "  make fuel-drill                  fill tank to 1100 (long-running)"
	@echo "  make equipment-probe             3 equipment pickups via 9 attempts"
	@echo "  make combat-probe                3 combat engagements, 20 shots each"
	@echo "  make cadence-probe               server shot-serve rate at 4 spacings"
	@echo "  make weave-probe                 does a move cost the queued shot?"
	@echo "  make track                       enemy tracking probe (wire vs JS truth)"
	@echo "  make bot                         live HFSM bot"
	@echo "  make run                         5-min timed session + scorecard"
	@echo "  make up                          THE fleet command: newest release to image to container on sedona, with its public edge (run from the Windows hub)"
	@echo "  make down                        drain every bot to the lobby, then stop the container on sedona (run from the Windows hub)"
	@echo "  make dev                         fleet manager on the HOT TREE, foreground (development only)"
	@echo "  make analyze                     issue report + run audit + cross-session stats"
	@echo "  make digest                      compact per-run truth table (works on crashed runs)"
	@echo "  make sniff                       capture WebSocket session to disk"
	@echo ""
	@echo "Offline tools:"
	@echo "  make decode                      replay a capture through real decoders"
	@echo "  make analyze-viewport            analyze viewport bounds in captures"
	@echo "  make sim-baseline                fresh one-generation sim corpus + fidelity diff"
	@echo "  make sim-control ARGS=...        N=1 control: record MANIFEST, or compare BEFORE AFTER"
	@echo "  make analyze-command-coverage    does the sim survive every command a REAL client sends?"
	@echo ""

install:
	@echo ""
	@echo "=== install"
	poetry lock
	poetry sync --with dev
	poetry run playwright install chromium
	@echo "[install] done"

lint: _lint-prep _lint-guard _lint-mypy
	@echo ""
	@echo "[lint] passed"

# What lint and the suite both need first: a venv that answers, and the
# lock and the dev dependencies synced.
_deps:
	$(MAKETOOLS) venv-check
	poetry lock
	poetry sync --with dev

# The quick gates, and the two steps that REWRITE sources (ruff's fixes and
# its formatter), which must finish before anything reads the tree.
_lint-prep: _deps
	@echo ""
	@echo "=== lint"
	@echo ""
	@echo "  -- undecoded-field check"
	poetry run tankpit-check-undecoded-fields
	@echo ""
	@echo "  -- wiki gates (structure + claim binding)"
	poetry run tankpit-check-wiki
	@echo ""
	@echo "  -- ruff check + format"
	poetry run ruff check . --fix
	poetry run ruff format .

_lint-guard:
	@echo ""
	@echo "  -- guard"
	$(MAKETOOLS) guard

_lint-mypy:
	@echo ""
	@echo "  -- mypy"
	poetry run mypy src tests scripts

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command); see any sibling package's Makefile for what it adds over the
# inline recipe and the incident behind it.
test: _deps _test-suite
	@echo ""
	@echo "[test] passed"

_test-suite:
	@echo ""
	@echo "=== test"
	$(MAKETOOLS) test

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo ""
	@echo "=== ALL CHECKS PASSED ==="
	@echo ""

# The guard, mypy and the suite read the tree and write nothing another of
# them reads, so once _lint-prep has synced and formatted, they run side by
# side (tools/maketools ``concurrent``) and the check is charged the longest
# of the three instead of their sum (board task 28e47ae3: on sedona at
# c20080a69 the suite took 371 s after about 154 s of lint).
_check-unbudgeted: _lint-prep
	$(MAKETOOLS) concurrent _lint-guard _lint-mypy _test-suite

# The live demo case, run against austinwagner.org/tankpit (MCPs board task
# 46934cd6). ``make check`` skips it on every machine, because that run is
# also API's CI and the case presses the public spawn button; the fleet runs
# this target as clients/TankpitBot-execution on a Windows node with ffmpeg.
# It keeps only that case and fails with HOST_EXECUTION_INCOMPLETE when it is
# skipped or none ran. Coverage is off because one case is not the suite;
# ``make check`` holds the 100.
execution:
	poetry sync --with dev
	$(MAKETOOLS) test --host-execution --no-cov

# An empty OUTPUT unsets TANKPIT_OUTPUT (the sniffer then writes its default
# path plus the timestamped archive); a value sets it.
sniff: install
	@echo ""
	@echo "=== sniff (output override: '$(OUTPUT)', blank means the default path + archive)"
	$(MAKETOOLS) env TANKPIT_OUTPUT=$(OUTPUT) -- poetry run tankpit-sniff

# Live health gate. Run a $(SMOKE_DURATION)-second bot session, then
# assert 5 concrete signals on runs/bot/latest.events.jsonl. Override
# the duration with `make smoke SMOKE_DURATION=60`.
SMOKE_DURATION ?= 30
smoke: install
	@echo ""
	@echo "=== smoke (duration: $(SMOKE_DURATION)s)"
	$(MAKETOOLS) env TANKPIT_BOT_SESSION_SECONDS=$(SMOKE_DURATION) -- poetry run tankpit-bot
	@echo ""
	@echo "=== smoke assertions"
	poetry run tankpit-smoke

# Live debug session. Same as `make smoke` but skips the assertions
# and chains into `make analyze` so you can read the issue report.
DEBUG_DURATION ?= 30
debug-run: install
	@echo ""
	@echo "=== debug-run (duration: $(DEBUG_DURATION)s)"
	$(MAKETOOLS) env TANKPIT_BOT_SESSION_SECONDS=$(DEBUG_DURATION) -- poetry run tankpit-bot
	@echo ""
	@echo "=== debug-run analyze"
	poetry run tankpit-issue-report runs/bot/latest.events.jsonl

teleport-probe: teleport-probe-safe teleport-probe-aggressive
	@echo ""
	@echo "=== TELEPORT PROBES COMPLETE ==="

teleport-probe-safe: install
	@echo ""
	@echo "=== teleport-probe-safe (strategy: sync_before_teleport)"
	poetry run tankpit-teleport-probe --max-targets 3 --teleport-strategy sync_before_teleport

teleport-probe-aggressive: install
	@echo ""
	@echo "=== teleport-probe-aggressive (strategy: immediate_after_map_open)"
	poetry run tankpit-teleport-probe --max-targets 3 --teleport-strategy immediate_after_map_open

teleport-probe-full: install
	@echo ""
	@echo "=== teleport-probe-full (every target, default strategy)"
	poetry run tankpit-teleport-probe

enemy-teleport-probe: install
	@echo ""
	@echo "=== enemy-teleport-probe (3 attempts, map_open)"
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy map_open

enemy-teleport-probe-map: install
	@echo ""
	@echo "=== enemy-teleport-probe-map (3 attempts via map open)"
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy map_open

enemy-teleport-probe-nearest: install
	@echo ""
	@echo "=== enemy-teleport-probe-nearest (3 attempts via nearest enemy)"
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy nearest_enemy

# Teleport to one bot, then sit idle for ten minutes capturing what the
# server sends, heartbeating every 1.5s so the session stays alive.
bot-watch: install
	@echo ""
	@echo "=== bot-watch (teleport to a bot, sit idle 10 min, capture)"
	$(MAKETOOLS) env TANKPIT_ENEMY_TELEPORT_SETTLE_MS=600000 TANKPIT_ENEMY_TELEPORT_HEARTBEAT_MS=1500 TANKPIT_ENEMY_TELEPORT_PROBE_OUTPUT=runs/probe/bot_watch_probe.json -- poetry run tankpit-enemy-teleport-probe --max-attempts 1 --acquisition-strategy map_open

respawn-watch: install
	@echo ""
	@echo "=== respawn-watch"
	poetry run tankpit-respawn-watch-probe

radar-watch: install
	@echo ""
	@echo "=== radar-watch"
	poetry run tankpit-radar-watch-probe

density-probe: install
	@echo ""
	@echo "=== density-probe"
	poetry run tankpit-density-probe

larder-probe: install
	@echo ""
	@echo "=== larder-probe"
	poetry run tankpit-larder-probe

mine-landing-probe: install
	@echo ""
	@echo "=== mine-landing-probe"
	poetry run tankpit-mine-landing-probe

viewport-probe: install
	@echo ""
	@echo "=== viewport-probe"
	poetry run tankpit-viewport-probe

key-probe: install
	@echo ""
	@echo "=== key-probe"
	poetry run tankpit-key-probe

fuel-probe: install
	@echo ""
	@echo "=== fuel-probe (3 pickups via 9 attempts)"
	poetry run tankpit-fuel-probe --target-pickups 3 --max-attempts 9

fuel-drill: install
	@echo ""
	@echo "=== fuel-drill (fill to 1100)"
	poetry run tankpit-fuel-drill --target-fills 3 --max-attempts 18 --fuel-full-threshold 1100

equipment-probe: install
	@echo ""
	@echo "=== equipment-probe (3 pickups via 9 attempts)"
	poetry run tankpit-equipment-probe --target-pickups 3 --max-attempts 9

movement-probe: install
	@echo ""
	@echo "=== movement-probe (3 targets)"
	poetry run tankpit-movement-probe --max-targets 3

combat-probe: install
	@echo ""
	@echo "=== combat-probe (3 engagements, 20 shots each)"
	poetry run tankpit-combat-probe --max-engagements 3 --max-shots 20

cadence-probe: install
	@echo ""
	@echo "=== cadence-probe (4 spacings, 6 shots each)"
	poetry run tankpit-cadence-probe --spacings 2000,1000,500,250 --shots 6

weave-probe: install
	@echo ""
	@echo "=== weave-probe (8 beats, 1 burst)"
	poetry run tankpit-weave-probe --beats 8 --bursts 1

TRACK_SAMPLE_DURATION_MS ?= 120000
TRACK_SAMPLE_INTERVAL_MS ?= 1000
track: install
	@echo ""
	@echo "=== track (sample $(TRACK_SAMPLE_DURATION_MS)ms every $(TRACK_SAMPLE_INTERVAL_MS)ms)"
	poetry run tankpit-enemy-tracking-probe --sample-duration-ms $(TRACK_SAMPLE_DURATION_MS) --sample-interval-ms $(TRACK_SAMPLE_INTERVAL_MS)

bot: install
	@echo ""
	@echo "=== bot (live HFSM)"
	poetry run tankpit-bot

# The fleet manager is the AI-operated control plane: it spawns
# tankpit-bot child processes (one per instance namespace) and exposes
# spawn/list/stop/remove over plain HTTP so an AI session can drive a
# multi-bot fleet without owning any bot's terminal. Runs in a
# user-owned terminal because harness background tasks are killed
# after ~46 minutes.
dev: install
	@echo ""
	@echo "=== dev (fleet manager, HOT TREE, foreground -- Ctrl+C drains)"
	@echo "  Live sessions can break under mid-surgery edits, so operate with 'make up'."
	poetry run tankpit-fleet

# The scorecard must run whatever the session's exit code was -- a failed
# or force-exited run is EXACTLY when the issue report matters -- and the
# target still exits with the bot's code so failures stay visible (operator
# flag 2026-09-03: a teardown wedge exited before the scorecard ever
# printed). ``env --then`` is that sequence: the second command runs
# regardless, and the first non-zero status of the two is the verdict.
run: install
	@echo ""
	@echo "=== run (timed session, TANKPIT_BOT_SESSION_SECONDS / _KILLS honored, default 300s, then the scorecard)"
	$(MAKETOOLS) env TANKPIT_BOT_SESSION_SECONDS?=300 --then "poetry run tankpit-issue-report runs/bot/latest.events.jsonl" -- poetry run tankpit-bot

digest:
	@echo "=== digest (latest run)"
	poetry run tankpit-run-digest runs/bot/latest.events.jsonl
	poetry run tankpit-stats

# Diff every archived run's analyzers against its own wire receipts
# (kill/death recounts, beat discipline, radar book, tombstone
# re-aims). Exit 1 when any run is flagged.
corpus-audit:
	@echo "=== corpus-audit (every archived run vs its wire receipts)"
	poetry run tankpit-corpus-audit runs/bot

analyze:
	@echo "=== analyze (issue report + engagements + forage economy + run audit + stats)"
	poetry run tankpit-issue-report runs/bot/latest.events.jsonl
	poetry run tankpit-engagements runs/bot/latest.events.jsonl
	poetry run tankpit-forage-economy runs/bot/latest.events.jsonl
	poetry run tankpit-run-audit runs/bot/latest.events.jsonl
	poetry run tankpit-stats

audit:
	@echo "=== audit (re-derive wiki physics claims from the runs archive)"
	poetry run tankpit-audit --stamp
	@echo "[audit] every validated claim re-derived from the archive"
	@echo "=== conformance (replay every capture through the sim, tick by tick)"
	poetry run tankpit-conformance --baseline $(CONFORMANCE_BASELINE)
	@echo "[audit] no command group fell below its conformance baseline"

# Rewrites the committed conformance baseline from a fresh replay of the
# archive. Run it after a sim change that RAISES a group's match rate, and
# commit the file with that change: the bar only moves on purpose.
conformance-baseline:
	poetry run tankpit-conformance --write-baseline $(CONFORMANCE_BASELINE)

# Where containers stand on each field, read from the archive's radar
# scans, with the terrain shape fitted on each field and scored on every
# other. A measurement, not a gate; see wiki/pages/container-census.md.
container-census:
	poetry run tankpit-container-census

# Reports which wiki pages are owed an audit. NOT part of `make check`:
# a stale anchor is a to-do marker, not a defect, and gating on it would
# redden the build on every source commit. See wiki/SCHEMA.md.
wiki-anchors:
	poetry run tankpit-wiki-anchors $(ARGS)

shadow:
	@echo "=== shadow (price the sim's laws against the real runs archive)"
	poetry run tankpit-shadow
	@echo "[shadow] every sim law priced at archive exactness"

sim-run:
	@echo "=== sim-run (production bot vs the simulator on real terrain -- no server, no browser)"
	poetry run tankpit-sim-run

sim-run-practice:
	@echo "=== sim-run-practice (production bot vs the certified practice-bot roster)"
	poetry run tankpit-sim-run --practice

sim-field:
	@echo "=== sim-field (several production bots on one sim field, two in the arena by default)"
	poetry run tankpit-sim-run --clients 2 $(ARGS)

roundtrip:
	@echo "=== roundtrip (encode(decode(x)) == x for every archived binary message)"
	poetry run tankpit-roundtrip
	@echo "[roundtrip] every archived binary message re-encodes byte-identically"

analyze-timing:
	@echo "=== analyze-timing (command-response latencies)"
	poetry run python -m scripts.analyze_session_timing

queue-probe: install
	@echo "=== queue-probe"
	poetry run python -m scripts.queue_probe

download-fields: install
	@echo "=== download-fields"
	poetry run python -m scripts.download_fields

decode:
	@echo "=== decode (replay a capture through the real decoders)"
	poetry run python -m scripts.decode

analyze-response-shapes:
	@echo "=== analyze-response-shapes"
	poetry run python -m scripts.analyze_response_shapes

sim-baseline:
	@echo "=== sim-baseline (fresh one-generation sim corpus + fidelity diff)"
	poetry run python -m scripts.build_sim_baseline $(ARGS)

sim-control:
	@echo "=== sim-control (record or compare the N=1 byte-identity control)"
	poetry run python -m scripts.sim_control $(ARGS)

analyze-command-coverage:
	@echo "=== analyze command coverage (does the sim survive what a REAL client sends?)"
	poetry run python -m scripts.analyze_command_coverage

analyze-recipient-policy:
	@echo "=== analyze recipient policy"
	poetry run python -m scripts.analyze_recipient_policy

analyze-viewport:
	@echo "=== analyze viewport"
	poetry run python -m scripts.analyze_viewport

# THE fleet lifecycle is ONE pair (operator order 2026-09-02: "one
# single command we use to start this ish"): ``make up`` runs the
# fleet CONTAINER from the newest release ON SEDONA, the game host
# (since 2026-09-28, MCPs board task 077204e8; scripts/fleet_host.py
# says what it stages there and from where) -- it resolves the release,
# builds its image on sedona if that exact tag does not exist yet
# (tankpit-fleet:v<ver>-<sha>, built FROM the snapshot so the tag
# carries the ladder's sha and a hot-tree build can never masquerade
# as a release), then composes it up with its public edge (the
# /demo/-only filter and tankpit's own tunnel) and asks
# https://tankpit.austinwagner.org whether it serves. ``make down`` is
# a DRAIN, never a kill:
# docker stop's SIGTERM enters drain_on_interrupt, every bot quits to
# the lobby, and the container exits after the last one lands
# (stop_grace_period 10m -- interrupting the wait is safe, the manager
# owns the drain). ``make dev`` is the only other way to run a
# manager: hot tree, foreground, for development -- live sessions on
# the hot tree can break under mid-surgery edits, which is the whole
# reason ``up`` runs the frozen snapshot.
#
# ``make release`` builds the frozen, runnable snapshot of the bot plus
# its two path-dep libs at the CURRENT COMMIT (uncommitted work is
# excluded by construction -- git archive reads HEAD). The fleet manager
# and live sessions run from the snapshot so hot-tree surgery can never
# crash them (2026-08-28: deleting a runtime-read file killed a live
# session). Versioning is the folder name:
# C:\Users\Test\PROJECTS\tankpit-releases\v<version>-<short-sha>.
#
# The retired host-mode targets (fleet / fleet-dev / up-docker /
# down-docker / image, 2026-09-02): one system, not two. TRANSITION
# NOTE: a container manager cannot adopt HOST bot processes -- drain
# any pre-container host fleet (poetry run tankpit-fleet-down from
# its release folder) before the first containerized ``make up``, or
# its spawn records on the shared runs/ volume will read as dead.
#
# WINDOWS-ONLY, and fenced as such: the release ladder lives at a Windows
# path on the hub, and the fleet on sedona is operated from the hub,
# which holds the tunnel token (~/.tankpit-secrets). The else arm refuses with the
# reason so a Linux checkout gets an answer rather than a parse error.
ifeq ($(OS),Windows_NT)
release:
	@Write-Host ""
	@Write-Host "==> release (frozen runnable snapshot for the fleet)" -ForegroundColor Cyan
	$$sha = (git rev-parse --short HEAD).Trim(); $$ver = (Select-String -Path pyproject.toml -Pattern 'version = .(\d+\.\d+\.\d+).').Matches[0].Groups[1].Value; $$dest = "C:\Users\Test\PROJECTS\tankpit-releases\v$$ver-$$sha"; if (Test-Path $$dest) { Write-Host "[release] $$dest already exists; nothing to do" -ForegroundColor Yellow } else { New-Item -ItemType Directory -Force $$dest | Out-Null; git -C ..\.. archive --format=zip HEAD clients/TankpitBot libs/platform_core libs/monorepo_guards -o "$$dest\snapshot.zip"; Expand-Archive "$$dest\snapshot.zip" -DestinationPath $$dest; Remove-Item "$$dest\snapshot.zip";Copy-Item accounts.json "$$dest\clients\TankpitBot\"; Copy-Item .env "$$dest\clients\TankpitBot\"; if (Test-Path data/tank_registry.json) { New-Item -ItemType Directory -Force "$$dest/clients/TankpitBot/data" | Out-Null; Copy-Item data/tank_registry.json "$$dest/clients/TankpitBot/data/" }; New-Item -ItemType Directory -Force "$$dest\clients\TankpitBot\runs" | Out-Null; Push-Location "$$dest\clients\TankpitBot"; poetry install --only main; Pop-Location; Write-Host "[release] built $$dest" -ForegroundColor Green; Write-Host "[release] activate: 'make down' then 'make up' -- up builds this release's image on first run" -ForegroundColor DarkGray }

up:
	@Write-Host ""
	@Write-Host "==> up (the fleet on sedona: newest release, containerized, with its public edge)" -ForegroundColor Cyan
	poetry run python -m scripts.fleet_host up

down:
	@Write-Host ""
	@Write-Host "==> down (drain every bot on sedona, then stop the container)" -ForegroundColor Cyan
	poetry run python -m scripts.fleet_host down
else
release:
	@echo "release is Windows-only (the release ladder is C:/Users/Test/PROJECTS/tankpit-releases on the hub)"
	@exit 1

up:
	@echo "up is Windows-only (the fleet container is operated from the hub, where the release ladder lives)"
	@exit 1

down:
	@echo "down is Windows-only (the fleet container is operated from the hub, where the release ladder lives)"
	@exit 1
endif
