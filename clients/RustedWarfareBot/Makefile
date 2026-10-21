include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

# Path to the pinned game working copy. Every game invocation launches
# from here, never from the Steam install: runs rewrite preferences.ini,
# and Steam would re-verify the build out from under the class mappings.
GAME_DIR ?= .game

.PHONY: help lint test check _check-unbudgeted execution sources install boot-probe unit-dump type-flags wire-capture agent agent-selftest sandbox-probe discover-probe decompile play host watch fleet door fleet-up income sweep

help:
	@echo ""
	@echo "RustedWarfareBot -- available targets"
	@echo "====================================="
	@echo ""
	@echo "Code health:"
	@echo "  make check                       lint + sources + test (the gate, on any clean checkout)"
	@echo "  make execution                   agent-selftest against the pinned game (needs it at $(GAME_DIR))"
	@echo "  make lint                        guard + ruff + mypy over src, tests, scripts"
	@echo "  make test                        pytest + coverage (100% statements and branches)"
	@echo "  make sources                     wiki provenance and navigation gate"
	@echo "  make install                     poetry install"
	@echo ""
	@echo "Game probes (need the pinned Linux depot at $(GAME_DIR), PROBE_DISPLAY=N starts an X server):"
	@echo "  make boot-probe                  headless boot, archive the engine log"
	@echo "  make unit-dump                   -printunits stat catalogue"
	@echo "  make type-flags                  placement rules from the live type registry"
	@echo "  make wire-capture                regenerate the archived world-stream fixture"
	@echo "  make sandbox-probe               live skirmish with the agent attached"
	@echo "  make discover-probe              reflective engine-state snapshots at timed offsets"
	@echo ""
	@echo "Play:"
	@echo "  make play                        start a game, play a whole match, stop it"
	@echo "                                   PLAY_SAMPLES=1500 PLAY_SEED=0 PLAY_LOG=runs/play.log"
	@echo "                                   PLAY_SEED pins the opponents' RNG, PLAY_LOCKSTEP locks the tick"
	@echo "  make income                      measure extractor credits/s and payback time"
	@echo "                                   INCOME_STAGES=4 INCOME_IDLE=200 INCOME_LOG=runs/income.log"
	@echo "  make sweep                       play a batch of matches in parallel, resumably"
	@echo "                                   SWEEP_JOBS=sweeps/x.txt SWEEP_NAME=x SWEEP_WORKERS=4"
	@echo "  make watch                       one windowed match a human can watch (needs a desktop)"
	@echo "  make fleet                       match fleet manager UI on 127.0.0.1:27500 (spawn/kill/restart parallel matches)"
	@echo "                                   WATCH_DIFFICULTY=3 WATCH_DOCTRINE=doctrines/aa-counter-guard.doctrine"
	@echo "  make host                        bot-hosted LAN game a human can join"
	@echo "  make door                        match-service HTTP door, detached (port 27501)"
	@echo "  make fleet-up                    start any missing match workers, detached"
	@echo "                                   HOST_MAP=maps/skirmish/[p2]duel_lake.tmx"
	@echo ""
	@echo "Java agent (built with the host JDK on PATH, as the game's depot ships only a JRE):"
	@echo "  make agent                       compile + jar the javaagent"
	@echo "  make agent-selftest              patch the real jar, verify via the JVM verifier (make execution runs it)"
	@echo ""
	@echo "Static analysis:"
	@echo "  make decompile                   CFR the pinned jar into a readable tree (gitignored)"
	@echo ""

install:
	@echo ""
	@echo "=== install"
	poetry lock
	poetry install --with dev
	@echo "[install] done"

lint:
	@echo ""
	@echo "=== lint"
	$(MAKETOOLS) venv-check
	poetry lock
	poetry install --with dev
	@echo ""
	@echo "  -- guard"
	$(MAKETOOLS) guard
	@echo ""
	@echo "  -- ruff check + format"
	poetry run ruff check . --fix
	poetry run ruff format .
	@echo ""
	@echo "  -- mypy"
	poetry run mypy src tests scripts
	@echo ""
	@echo "[lint] passed"

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command): a per-run COVERAGE_FILE under runs/, not the default fixed
# ``.coverage``.
#
# The same reasoning as the absent ``--basetemp`` in pyproject, and it cost a
# confusing hour to rediscover: this working tree is used by more than one agent
# at a time, coverage writes one SQLite file at a fixed path, and two concurrent
# runs corrupt it. The reader then fails with "no such table: arc" and reports
# 0.00% while every one of the 520 tests passes -- a green suite presented as a
# coverage failure, which reads as a code problem and is not one.
test:
	@echo ""
	@echo "=== test"
	poetry lock
	poetry install --with dev
	$(MAKETOOLS) test
	@echo ""
	@echo "[test] passed"

# The wiki's mechanical contract, gated: cited paths resolve with anchors in
# bounds, wikilinks land, footnotes pair, every page is hub-reachable, and the
# index counts are true. Enforced because the 2026-08-14 hand audit found five
# defects that had accumulated silently -- and its own first run then caught
# the hand audit overcounting every hub by the format-comment line. Citations
# into the gitignored artifact store (runs/, .game/, .decompiled/) are
# existence-checked on the machine that holds every one of those roots and
# reported as not applicable, in numbers, anywhere else: the checker decides
# from the tree, because this target runs on CI runners and fleet nodes now
# and the --artifacts flag it used to pass failed the gate on all of them.
sources:
	@echo ""
	@echo "=== sources (wiki provenance and navigation)"
	poetry run python -m scripts.wiki_check

# The gate is what a clean checkout can prove: lint, the wiki's repo tier and
# the suite. It used to chain agent-selftest too, and then it could pass only
# on the hub, because the self-test patches the real pinned jar, a 451 MB
# commercial binary that is never tracked. Every review and fleet re-run of a
# commit touching this package read FAILED at Makefile:460 on a missing
# .game/jvm64/bin/javac.exe (board tasks 4f228608 and 156c4745). The Java half
# is not dropped: it is `make execution` below.
check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo ""
	@echo "=== ALL CHECKS PASSED ==="
	@echo ""

_check-unbudgeted: lint sources | test

# The Java half, executed against the real target, as tools/hpc3 and tools/fleet
# name theirs (board task 465689f5). agent-selftest patches every Targets entry
# in the pinned jar and lets the game's own JVM verifier certify the result, so
# a patcher regression or an obfuscated name that moved in a game update fails
# here rather than inside a live engine. Run it after any agent or game change,
# on a machine that holds the game and a JDK. Nothing skips: without the game
# it fails on the missing runtime, and without a JDK on the missing javac.
execution: agent-selftest

# Pin the engine's generator so repeated runs face the same opponent choices.
# 0 leaves it alone, which is the honest default: a run that silently became
# repeatable would be worse than one that visibly is not.
# Frames between samples when locking the exchange to the tick. 0 free-runs on
# a wall clock, which is what left the coupling nondeterministic even with the
# engine's randomness pinned.
PLAY_LOCKSTEP ?= 0
PLAY_SEED ?= 0
PLAY_SAMPLES ?= 1500
PLAY_SETTLE ?= 22
# The game writes its own log here; the agent writes to stdout, which is a
# different stream and is captured beside it as <PLAY_LOG>.agent. Conflating
# the two cost a wrong conclusion once: a seed that had applied perfectly well
# was reported as never firing, because the confirmation was grepped for in the
# game's file rather than the agent's.
PLAY_LOG ?= runs/play.log

# Which planner the harness drives, and what to pass it after the three paths
# every planner takes. Parameterised so a probe reuses this recipe rather than
# copying it: bringing the game up headless, waiting for the channel, letting
# the map settle and tearing the process down again is the fiddly part, and a
# second copy of it would drift from this one.
# Invoked as a module, not a path. A planner that imports another planner's
# loaders needs the repository root on sys.path, and running a file by path puts
# its own directory there instead -- so "python scripts/income.py" cannot see
# "scripts.play" at all. "python -m" is what `make lint` already uses for the
# guard, so this is the form the tree is known to support.
# Which match to play. Empty leaves the engine's own hardcoded sandbox, which is
# a five-player free-for-all on a ten-player map that nobody chose: the setup is
# read from a GUI document with no values headless, so every figure falls
# through to a Java default of four opponents at Medium ([[policy-determinism]]).
#
# Difficulty is an income multiplier on the AI alone -- 0.4x Very Easy, 0.7x
# Easy, 1.0x Medium, 1.4x Hard, 1.8x Very Hard, 3.7x Impossible -- so at the
# default an opponent earns exactly what the bot does.
PLAY_MAP ?=
PLAY_OPPONENTS ?= 1
PLAY_DIFFICULTY ?= 0

PLAY_MODULE ?= scripts.play
PLAY_ARGS ?= $(PLAY_SAMPLES)

# A frozen code snapshot to play from, or empty to import the working tree.
# Sweeps always pass one (rw_bot.harness.runner.prepare_tree): the planner then
# runs with -P so the repository root cannot outrank the snapshot on sys.path,
# and the agent jar is the snapshot's prebuilt copy rather than a per-match
# compile. The working directory stays the repository root either way, so
# catalogue, type-dump, doctrine and trace paths resolve unchanged.
PLAY_TREE ?=

# Constant frame delta in ms, 0 to leave the engine on wall clock. What makes
# one seed produce one answer ([[policy-determinism]]); trees frozen before the
# option exists must run with 0, so the default stays off until they retire.
PLAY_PINDELTA ?= 0
# Fast-forward: target logic updates per second, 0 for the engine's own
# pacing. With the delta pinned this multiplies wall-clock speed without
# touching the simulation (task #35).
PLAY_FASTFORWARD ?= 0
# Diagnostic draw tap: 1 replaces the engine's generator with a per-caller
# counting one and logs a tally per sample window (task #36).
PLAY_RNGTAP ?= 0

# Diagnostic passthrough for agent options no launch knob names -- e.g.
# discovery snapshots riding a live match to capture full rosters around a
# divergence window: PLAY_EXTRA_AGENT="discoverAtSeconds=24,27,30;stateOutPath=runs/x.ndjson"
PLAY_EXTRA_AGENT ?=

# How many extractors the income probe builds, and how long it stands still
# between each one. Target-specific variables rather than a recursive $(MAKE):
# the argument list contains a space, and quoting it through a sub-make split
# it into a stray target.
INCOME_STAGES ?= 4
INCOME_IDLE ?= 200
INCOME_LOG ?= runs/income.log

# The single-match port band. A sweep leases its ports from clone ordinals
# (rw_bot.harness.clone.leased_port); a single match draws one per invocation
# so two agents playing in one tree cannot collide on a fixed number.
PLAY_PORT_LOW := 27600
PLAY_PORT_HIGH := 27999

# The body lives in rw_bot.harness.play_match_cli -- launch plumbing, read
# from the working tree like this Makefile itself, while what the match RUNS
# comes from the frozen tree when PLAY_TREE is set ([[policy-loop]]).
#
# It was scripts/make/play.ps1 until the cluster port. PowerShell cannot start
# a match on a Linux compute node, and a second shell script beside the first
# would be two launchers to keep in step -- with a missed edit meaning a batch
# that runs the wrong configuration and still files a scorecard. One launcher
# for the workstation and the cluster; this recipe is just the variables it is
# given. Every other game recipe below followed it on 2026-10-21 (board task
# 0eae9619), each one a module under rw_bot.harness or scripts.
#
# The port was ``PLAY_PORT := $(shell python -c "...randint...")`` until the
# portable grammar: ``$(shell`` runs the platform shell, so the draw moves into
# the launcher's ``env --draw``, which substitutes it into ``--port @PLAY_PORT@``
# and exports PLAY_PORT to the child. The ``:=`` lesson that cost a debugging
# round still holds in spirit: one draw per invocation, every reference the
# same number. The launcher prints the port it bound.
#
# The optional flags are appended with $(if ...) rather than passed empty.
# PowerShell, which ran these recipes until 2026-10-21, DROPPED an empty
# argument entirely -- so `--map ""` arrived as a bare `--map` and swallowed
# the next flag as its value ("--map requires a value, got the flag
# '--opponents'"). Omitting them is also what a sweep does
# (rw_bot.harness.sweep.make_argv), so both callers spell "not asked for" the
# same way: absent.
play:
	@echo ""
	@echo "=== play (headless match, port drawn from $(PLAY_PORT_LOW)-$(PLAY_PORT_HIGH), log: $(PLAY_LOG))"
	$(MAKETOOLS) env --draw PLAY_PORT=$(PLAY_PORT_LOW)-$(PLAY_PORT_HIGH) -- poetry run python -m rw_bot.harness.play_match_cli --port @PLAY_PORT@ --game-dir "$(GAME_DIR)" --seed $(PLAY_SEED) --lockstep $(PLAY_LOCKSTEP) --settle $(PLAY_SETTLE) --play-log "$(PLAY_LOG)" --opponents $(PLAY_OPPONENTS) --difficulty $(PLAY_DIFFICULTY) --pin-delta $(PLAY_PINDELTA) --fast-forward $(PLAY_FASTFORWARD) --rng-tap $(PLAY_RNGTAP) --module $(PLAY_MODULE) --play-args "$(PLAY_ARGS)"$(if $(PLAY_MAP), --map "$(PLAY_MAP)")$(if $(PLAY_TREE), --tree "$(PLAY_TREE)")$(if $(PLAY_EXTRA_AGENT), --extra-agent-args "$(PLAY_EXTRA_AGENT)")

income: PLAY_MODULE = scripts.income
income: PLAY_ARGS = $(INCOME_STAGES) $(INCOME_IDLE)
income: PLAY_LOG = $(INCOME_LOG)
income: play

# The match fleet: one user-owned manager process with a control page,
# spawning `make play` children (seed, map, opponents, difficulty,
# fast-forward, frozen tree are spawn parameters). The per-invocation port
# draw is what makes parallel matches safe. Runs in a terminal the operator
# owns; browse to the printed URL.
fleet: install
	@echo ""
	@echo "=== fleet (match fleet manager, http://127.0.0.1:27500/)"
	poetry run rw-fleet

# A batch of matches, played several at a time. Each worker gets its own copy of
# the game directory because a running match writes three fixed-name paths
# inside its own -- everything else the launcher needs was already
# per-invocation ([[harness-nodisplay]]).
#
# Resumable by construction: a match's result is a file named after its job, so
# re-issuing the same command replays only what is missing. There is no progress
# file that could disagree with the results.
SWEEP_JOBS ?= sweeps/jobs.txt
SWEEP_NAME ?= run
SWEEP_WORKERS ?= 4
# Engine frames between samples. Never zero for a sweep: free running, the
# exchange is paced by a wall clock, so parallel matches under CPU contention
# sample at different game-times and running them in parallel would change their
# results ([[policy-determinism]]).
SWEEP_LOCKSTEP ?= 75

# Which match every job in the batch plays: a map path and a difficulty, or
# empty for the engine's own hardcoded ten-player free-for-all. The map decides
# the opponent count, because the engine caps teams by the map's own -- so a
# two-player map is a duel ([[policy-determinism]]).
#
#   make sweep SWEEP_MATCH="maps/skirmish/[p2]duel_lake.tmx -2"
SWEEP_MATCH ?=

# Constant frame delta for every match in the batch, empty to leave the engine
# on the wall clock. Only meaningful with SWEEP_MATCH set (it is the seventh
# positional); trees frozen before the option exists must leave it empty.
SWEEP_PINDELTA ?=

# Wall-clock multiple every match in the batch runs at, empty for realtime.
# The gym knob, certified bit-exact against realtime at 10 (log 2026-08-06).
# Eighth positional, so it requires SWEEP_PINDELTA set; trees frozen before
# the option exists must leave it empty.
SWEEP_FASTFORWARD ?=

# Depends on the agent jar because the batch freezes a copy of it: matches
# play from the snapshot, so the jar has to be current when the snapshot is
# taken rather than compiled per match ([[policy-loop]]).
sweep: agent
	@echo ""
	@echo "=== sweep $(SWEEP_NAME) ($(SWEEP_WORKERS) workers, jobs: $(SWEEP_JOBS))"
	poetry run python -m scripts.sweep "$(SWEEP_JOBS)" "$(SWEEP_NAME)" $(SWEEP_WORKERS) $(SWEEP_LOCKSTEP) $(SWEEP_MATCH) $(SWEEP_PINDELTA) $(SWEEP_FASTFORWARD)

# Decompile the pinned jar into a readable source tree.
#
# Disassembly (javap) is ground truth but reads one method at a time, and
# cross-referencing it by grep has already produced one false negative on this
# codebase -- the real writer of the tick counter was missed because the
# increment compiles against the subclass as a bare field name. A whole-jar
# decompile answers "who calls this" and "what else writes this" directly.
#
# Output is DERIVED and gitignored. Anything a wiki page cites gets copied into
# wiki/sources/ as an excerpt, the same discipline the javap artifacts follow.
#
# CFR is pinned by version: a decompiler is a heuristic, and an unpinned one
# would silently change the source we reason about between runs. The body is
# rw_bot.harness.decompile_cli, which fetches the pinned jar into .tools once.
CFR_VERSION ?= 0.152
DECOMPILED ?= .decompiled

# An X display the probes start their own server on (xvfb-run), or 0 for the
# machine's own. -nodisplay still opens a display ([[harness-nodisplay]]), so a
# headless node needs one; a desktop does not.
PROBE_DISPLAY ?= 0

# Headless engine boot writes the engine log to the path given by LOG
# (default: a timestamped file under runs/). Uses the flag set proven in
# wiki/pages/harness-nodisplay.md.
LOG ?= runs/boot-probe.log
# Live skirmish with the agent attached. Runs until killed -- the engine logs
# only events, so a quiet log here means a running sim, not a stalled one.
SANDBOX_LOG ?= runs/sandbox-probe.log
# Reflective snapshot of the live engine at each requested elapsed time. Two
# times by default because the comparison is the point: fields that change
# between a pre-map and post-map snapshot are the ones holding match state,
# which a single dump cannot distinguish from static configuration.
DISCOVER_AT ?= 5,25
DISCOVER_OUT ?= runs/discovery.txt
# A short capture of the live world stream, kept as the wire contract's fixture.
#
# The decoder is tested against these bytes rather than against a fixture
# written to match it, so this is regenerated whenever the record shape changes
# -- editing the archived capture by hand would defeat the point of having one.
#
# Three consecutive offsets rather than one: the decoder's job is folding a
# sequence into samples, and a single sample cannot exercise the boundary
# between two.
WIRE_CAPTURE_OUT ?= runs/world-sample.ndjson
WIRE_CAPTURE_AT ?= 16,17,18
# The placement rules -printunits does not print.
#
# The stat catalogue carries prices and speeds but nothing about where a
# structure may stand, and the only mention of the resource-pool rule in any
# shipped file is an English sentence in a translations bundle. So it is asked
# of the live engine instead: every registered type is resolved through the
# engine's own by-name lookup -- the same call the build path makes -- and
# asked for its own placement predicate.
#
# Regenerate after a game update or a change to the enabled mod set, for the
# same reasons unit-dump needs regenerating.
TYPE_FLAGS_OUT ?= runs/type-flags.ndjson
TYPE_FLAGS_AT ?= 20
# Host a LAN game the bot plays in, for a human to join. The join address is
# printed at launch; the match starts the moment a player connects. Sparring
# is realtime by necessity -- no lockstep, no hold, no seed
# ([[multiplayer-portability-invariants]]).
HOST_MAP ?= maps/skirmish/[p2]duel_lake.tmx
HOST_SAMPLES ?= 20000
HOST_DOCTRINE ?= doctrines/flame-nocover.doctrine
HOST_LOG ?= runs/host.log
# One match with the window on, so a human can watch the bot fight. The same
# agent-driven skirmish as `play` with rendering on and the reproducibility
# machinery off: lockstep would hold frames on the planner's acks and a fixed
# logic step is pointless when the point is to watch it in real time. Difficulty
# defaults to the rung the champion is measured against.
WATCH_MAP ?= maps/skirmish/[p2]duel_lake.tmx
WATCH_OPPONENTS ?= 1
WATCH_DIFFICULTY ?= 3
WATCH_SAMPLES ?= 20000
WATCH_DOCTRINE ?= doctrines/flame-nocover.doctrine
WATCH_LOG ?= runs/watch.log

# Everything below drives the pinned game copy, the Linux depot at GAME_DIR,
# through a module that composes and runs it: rw_bot.harness.probe_cli for the
# probes, agent_cli for the agent, decompile_cli for CFR, interactive_cli for a
# watched or hosted game, scripts.service_up for the match service. Each was a
# PowerShell script under scripts/make until 2026-10-21 (board task 0eae9619).
PROBE := poetry run python -m rw_bot.harness.probe_cli

boot-probe:
	@echo ""
	@echo "=== boot-probe (headless, log: $(LOG))"
	$(PROBE) --kind boot --game-dir "$(GAME_DIR)" --log "$(LOG)" --display $(PROBE_DISPLAY)

# Builds the runtime jar, which is what a game attaches with -javaagent.
#
# Compiles into a per-invocation directory with the host JDK at
# rw_bot.harness.agent_build.JAVA_RELEASE, the level the depot's Java 8 loads,
# and publishes the finished jar in one rename, so two agents building at once
# cannot interleave class files or publish a half-written jar. The rename is
# read back: the PowerShell recipe once reported a build over a jar it never
# replaced. No third-party libraries by design: the agent loads into the game's
# classloader beside obfuscated 1.15 classes, where every dependency is a
# conflict surface.
agent:
	@echo ""
	@echo "=== agent (host JDK, zero third-party deps)"
	poetry run python -m rw_bot.harness.agent_cli build

# Patches every Targets entry in the real pinned jar and DEFINES each result,
# so the JVM's own bytecode verifier -- not a second reading by the code that
# wrote the bytes -- is what certifies the patch. Run this after any game
# update: obfuscated names move silently between builds.
# Deliberately does NOT depend on the jar.
#
# The self-checks need class files, not an archive, so this compiles into a
# per-invocation directory and runs from it. That decoupling is what lets the
# gate pass while a game is running: a JVM with the agent attached holds
# rw-agent.jar open, and a gate that rebuilt it would fail for a reason that
# has nothing to do with the code under test. It also lets two agents run
# `make execution` on one working tree at the same time. The self-test runs on
# the game's own runtime, the verifier the engine loads a patch under.
agent-selftest:
	@echo ""
	@echo "=== agent-selftest (JVM verifier as oracle)"
	poetry run python -m rw_bot.harness.agent_cli selftest --game-dir "$(GAME_DIR)"

sandbox-probe: agent
	@echo ""
	@echo "=== sandbox-probe (live skirmish, agent attached, log: $(SANDBOX_LOG))"
	$(PROBE) --kind sandbox --game-dir "$(GAME_DIR)" --log "$(SANDBOX_LOG)" --display $(PROBE_DISPLAY)

discover-probe: agent
	@echo ""
	@echo "=== discover-probe (snapshots at $(DISCOVER_AT)s into $(DISCOVER_OUT))"
	$(PROBE) --kind discover --game-dir "$(GAME_DIR)" --log "$(DISCOVER_OUT)" --at "$(DISCOVER_AT)" --display $(PROBE_DISPLAY)

decompile:
	@echo ""
	@echo "=== decompile (CFR $(CFR_VERSION) into $(DECOMPILED))"
	poetry run python -m rw_bot.harness.decompile_cli --game-dir "$(GAME_DIR)" --cfr-version "$(CFR_VERSION)" --out "$(DECOMPILED)"

wire-capture: agent
	@echo ""
	@echo "=== wire-capture (live world stream at $(WIRE_CAPTURE_AT)s into $(WIRE_CAPTURE_OUT))"
	$(PROBE) --kind wire --game-dir "$(GAME_DIR)" --log "runs/wire-capture.log" --at "$(WIRE_CAPTURE_AT)" --out "$(WIRE_CAPTURE_OUT)" --display $(PROBE_DISPLAY)

type-flags: agent
	@echo ""
	@echo "=== type-flags (placement rules from the live registry into $(TYPE_FLAGS_OUT))"
	$(PROBE) --kind type-flags --game-dir "$(GAME_DIR)" --log "runs/type-flags.log" --at "$(TYPE_FLAGS_AT)" --out "$(TYPE_FLAGS_OUT)" --display $(PROBE_DISPLAY)

unit-dump:
	@echo ""
	@echo "=== unit-dump (-printunits stat catalogue)"
	$(PROBE) --kind units --game-dir "$(GAME_DIR)" --log "runs/printunits.log" --display $(PROBE_DISPLAY)

# A watched or hosted game draws its channel port per invocation exactly as
# play does, through the launcher's env --draw.
host:
	@echo ""
	@echo "=== host (bot-hosted LAN game on port 5123, log: $(HOST_LOG))"
	$(MAKETOOLS) env --draw PLAY_PORT=$(PLAY_PORT_LOW)-$(PLAY_PORT_HIGH) -- poetry run python -m rw_bot.harness.interactive_cli --mode host --port @PLAY_PORT@ --game-dir "$(GAME_DIR)" --play-log "$(HOST_LOG)" --map "$(HOST_MAP)" --module $(PLAY_MODULE) --play-args "$(HOST_SAMPLES) $(HOST_DOCTRINE) -"

watch:
	@echo ""
	@echo "=== watch (windowed match at difficulty $(WATCH_DIFFICULTY), log: $(WATCH_LOG))"
	$(MAKETOOLS) env --draw PLAY_PORT=$(PLAY_PORT_LOW)-$(PLAY_PORT_HIGH) -- poetry run python -m rw_bot.harness.interactive_cli --mode watch --port @PLAY_PORT@ --game-dir "$(GAME_DIR)" --play-log "$(WATCH_LOG)" --map "$(WATCH_MAP)" --opponents $(WATCH_OPPONENTS) --difficulty $(WATCH_DIFFICULTY) --module $(PLAY_MODULE) --play-args "$(WATCH_SAMPLES) $(WATCH_DOCTRINE) -"

# The match service's HTTP door, detached from the launching terminal.
# Reads the database password from the platform-postgres container at
# launch; the DSN never lands on disk.
door:
	@echo ""
	@echo "=== door (match service on http://127.0.0.1:27501/)"
	poetry run python -m scripts.service_up door

# The worker fleet, detached: starts any of the 8 workers not already
# running. The whole Docker-crash recovery is `make door` (if the door
# also died) + `make fleet-up` -- the queue rows survive Postgres
# restarts and the startup reap requeues what dead owners held.
fleet-up:
	@echo ""
	@echo "=== fleet-up (8 detached match workers)"
	poetry run python -m scripts.service_up fleet-up
