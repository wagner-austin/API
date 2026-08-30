SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

# Path to the pinned game working copy. Every game invocation launches
# from here, never from the Steam install: runs rewrite preferences.ini,
# and Steam would re-verify the build out from under the class mappings.
GAME_DIR ?= .game

.PHONY: help lint test check sources install boot-probe unit-dump type-flags wire-capture agent agent-selftest sandbox-probe discover-probe decompile play host watch fleet door fleet-up income sweep

help:
	@Write-Host ""
	@Write-Host "RustedWarfareBot -- available targets" -ForegroundColor Cyan
	@Write-Host "=====================================" -ForegroundColor Cyan
	@Write-Host ""
	@Write-Host "Code health:" -ForegroundColor Yellow
	@Write-Host "  make check                       lint + test (the gate)"
	@Write-Host "  make lint                        guard + ruff + mypy over src, tests, scripts"
	@Write-Host "  make test                        pytest + coverage (100% statements and branches)"
	@Write-Host "  make sources                     wiki provenance and navigation gate"
	@Write-Host "  make install                     poetry install"
	@Write-Host ""
	@Write-Host "Game probes (need the pinned copy at $(GAME_DIR)):" -ForegroundColor Yellow
	@Write-Host "  make boot-probe                  headless boot, archive the engine log"
	@Write-Host "  make unit-dump                   -printunits stat catalogue"
	@Write-Host "  make type-flags                  placement rules from the live type registry"
	@Write-Host "  make wire-capture                regenerate the archived world-stream fixture"
	@Write-Host "  make sandbox-probe               live skirmish with the agent attached"
	@Write-Host "  make discover-probe              reflective engine-state snapshots at timed offsets"
	@Write-Host ""
	@Write-Host "Play:" -ForegroundColor Yellow
	@Write-Host "  make play                        start a game, play a whole match, stop it"
	@Write-Host "                                   PLAY_SAMPLES=1500 PLAY_SEED=0 PLAY_LOG=runs/play.log"
	@Write-Host "                                   PLAY_SEED pins the opponents' RNG; PLAY_LOCKSTEP locks the tick"
	@Write-Host "  make income                      measure extractor credits/s and payback time"
	@Write-Host "                                   INCOME_STAGES=4 INCOME_IDLE=200 INCOME_LOG=runs/income.log"
	@Write-Host "  make sweep                       play a batch of matches in parallel, resumably"
	@Write-Host "                                   SWEEP_JOBS=sweeps/x.txt SWEEP_NAME=x SWEEP_WORKERS=4"
	@Write-Host "  make watch                       one windowed match a human can watch"
	@Write-Host "  make fleet                       match fleet manager UI on 127.0.0.1:27500 (spawn/kill/restart parallel matches)"
	@Write-Host "                                   WATCH_DIFFICULTY=3 WATCH_DOCTRINE=doctrines/aa-counter-guard.doctrine"
	@Write-Host "  make host                        bot-hosted LAN game a human can join"
	@Write-Host "  make door                        match-service HTTP door, detached (port 27501)"
	@Write-Host "  make fleet-up                    start any missing match workers, detached"
	@Write-Host "                                   HOST_MAP=maps/skirmish/[p2]duel_lake.tmx"
	@Write-Host ""
	@Write-Host "Java agent:" -ForegroundColor Yellow
	@Write-Host "  make agent                       compile + jar the javaagent"
	@Write-Host "  make agent-selftest              patch the real jar, verify via the JVM verifier"
	@Write-Host ""
	@Write-Host "Static analysis:" -ForegroundColor Yellow
	@Write-Host "  make decompile                   CFR the pinned jar into a readable tree (gitignored)"
	@Write-Host ""

install:
	@Write-Host ""
	@Write-Host "==> install" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	@Write-Host "[install] done" -ForegroundColor Green

lint:
	@Write-Host ""
	@Write-Host "==> lint" -ForegroundColor Cyan
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry install --with dev
	@Write-Host ""
	@Write-Host "  -> guard" -ForegroundColor DarkCyan
	poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] guard failed" -ForegroundColor Red; exit $$LASTEXITCODE }
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
# The same reasoning as the absent ``--basetemp`` in pyproject, and it cost a
# confusing hour to rediscover: this working tree is used by more than one agent
# at a time, coverage writes one SQLite file at a fixed path, and two concurrent
# runs corrupt it. The reader then fails with "no such table: arc" and reports
# 0.00% while every one of the 520 tests passes -- a green suite presented as a
# coverage failure, which reads as a code problem and is not one.
#
# Per-run paths cannot collide, so no lock is needed. Fragments land beside the
# combined file: xdist writes ``.coverage-<id>.<host>.<pid>.<rand>`` per worker
# and coverage combines them, so the cleanup globs rather than naming one file.
# Under runs/ because that is where untracked run output already lives.
test:
	@Write-Host ""
	@Write-Host "==> test" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	& "..\\..\\scripts\\run-tests.ps1"; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }
	@Write-Host ""
	@Write-Host "[test] passed" -ForegroundColor Green

# The wiki's mechanical contract, gated: cited paths resolve with anchors in
# bounds, wikilinks land, footnotes pair, every page is hub-reachable, and the
# index counts are true. Enforced because the 2026-08-14 hand audit found five
# defects that had accumulated silently -- and its own first run then caught
# the hand audit overcounting every hub by the format-comment line.
sources:
	@Write-Host ""
	@Write-Host "==> sources (wiki provenance and navigation)" -ForegroundColor Cyan
	poetry run python -m scripts.wiki_check

# The Java half is gated here too. Without agent-selftest in the chain, a
# patcher regression or an obfuscated name that moved in a game update stays
# invisible until a live run dies in the engine, which is the worst place to
# read a build error. It needs the pinned game copy, which every other game
# target already requires.
check: lint sources | test agent-selftest
	@Write-Host ""
	@Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
	@Write-Host ""

# Headless engine boot. Writes the engine log to the path given by LOG
# (default: a timestamped file under runs/). Uses the flag set proven in
# wiki/pages/harness-nodisplay.md.
LOG ?= runs/boot-probe.log
boot-probe:
	@Write-Host ""
	@Write-Host "==> boot-probe (headless, log: $(LOG))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force (Split-Path -Parent "$(LOG)") | Out-Null
	$$log = (Resolve-Path -LiteralPath ".").Path + "\\$(LOG)"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -width 800 -height 600 -log $$log } finally { Pop-Location }

# The javaagent compiles with the game's OWN bundled JDK 13, so the compiler
# matches the runtime exactly, and targets Java 11 bytecode for headroom. No
# third-party libraries by design: the agent loads into the game's classloader
# beside obfuscated 1.15 classes, where every dependency is a conflict surface.
JAVAC ?= $(GAME_DIR)/jvm64/bin/javac.exe
JAVA ?= $(GAME_DIR)/jvm64/bin/java.exe
JAR ?= $(GAME_DIR)/jvm64/bin/jar.exe
AGENT_JAR := agent/build/rw-agent.jar

# Bytecode level the agent is compiled to, and it is NOT a preference.
#
# The oldest JVM the agent must load into is the Linux depot's JRE 1.8.0_131,
# whose class-file ceiling is 52. A class above a JVM's level does not
# degrade -- it fails at load with UnsupportedClassVersionError, fatally,
# before the game starts.
#
# This recipe built with `--release 11` until 2026-08-30 while
# rw_bot.harness.agent_build.JAVA_RELEASE said 8, and the jar that ships is
# this recipe's. The staged agent was class version 55 and aborted the JVM on
# the first Linux launch. The two are held equal by
# tests/test_agent_release.py, which is what makes this comment load-bearing
# rather than decorative.
AGENT_RELEASE := 8
AGENT_CP := $(AGENT_JAR);$(GAME_DIR)/game-lib.jar;$(GAME_DIR)/libs/*

# Builds the runtime jar, which is what a game attaches with -javaagent.
#
# Compiles into a per-invocation directory and moves the finished jar into
# place, so two agents building at once cannot interleave class files or
# publish a half-written jar. Replacing the jar still fails while a game holds
# it open -- that is the operating system, not something a build can route
# around -- so the failure says which cause to look for instead of surfacing as
# a bare "Device or resource busy".
#
# Move-Item needs -ErrorAction Stop for that catch to fire at all. Without it
# the failure is non-terminating: PowerShell prints the error, skips the catch,
# leaves $failed empty, and the target reports a successful build over a jar it
# never replaced. Observed -- a stale jar reported as built while a game held
# it open.
agent:
	@Write-Host ""
	@Write-Host "==> agent (javac --release $(AGENT_RELEASE), zero third-party deps)" -ForegroundColor Cyan
	& scripts/make/agent.ps1 -Javac "$(JAVAC)" -Jar "$(JAR)" -AgentJar "$(AGENT_JAR)" -Release "$(AGENT_RELEASE)"
	@Write-Host "[agent] built $(AGENT_JAR)" -ForegroundColor Green

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
# `make check` on one working tree at the same time.
agent-selftest:
	@Write-Host ""
	@Write-Host "==> agent-selftest (JVM verifier as oracle)" -ForegroundColor Cyan
	& scripts/make/selftest.ps1 -Javac "$(JAVAC)" -Java "$(JAVA)" -GameDir "$(GAME_DIR)"
	@Write-Host "[agent-selftest] passed" -ForegroundColor Green

# Live skirmish with the agent attached. Runs until killed -- the engine logs
# only events, so a quiet log here means a running sim, not a stalled one.
SANDBOX_LOG ?= runs/sandbox-probe.log
sandbox-probe: agent
	@Write-Host ""
	@Write-Host "==> sandbox-probe (live skirmish, agent attached, log: $(SANDBOX_LOG))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force (Split-Path -Parent "$(SANDBOX_LOG)") | Out-Null
	& scripts/make/probe.ps1 -GameDir "$(GAME_DIR)" -AgentJar "$(AGENT_JAR)" -Log "$(SANDBOX_LOG)"

# Reflective snapshot of the live engine at each requested elapsed time. Two
# times by default because the comparison is the point: fields that change
# between a pre-map and post-map snapshot are the ones holding match state,
# which a single dump cannot distinguish from static configuration.
DISCOVER_AT ?= 5,25
DISCOVER_OUT ?= runs/discovery.txt
discover-probe: agent
	@Write-Host ""
	@Write-Host "==> discover-probe (snapshots at $(DISCOVER_AT)s -> $(DISCOVER_OUT))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force (Split-Path -Parent "$(DISCOVER_OUT)") | Out-Null
	& scripts/make/probe.ps1 -GameDir "$(GAME_DIR)" -AgentJar "$(AGENT_JAR)" -AgentArgs "discoverAtSeconds=$(DISCOVER_AT);exitAfterDiscovery=true" -Log "$(DISCOVER_OUT)"
	@Write-Host "[discover-probe] snapshots written to $(DISCOVER_OUT)" -ForegroundColor Green

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
# would silently change the source we reason about between runs.
CFR_VERSION ?= 0.152
CFR_JAR := .tools/cfr-$(CFR_VERSION).jar
CFR_URL := https://repo1.maven.org/maven2/org/benf/cfr/$(CFR_VERSION)/cfr-$(CFR_VERSION).jar
DECOMPILED ?= .decompiled

decompile:
	@Write-Host ""
	@Write-Host "==> decompile (CFR $(CFR_VERSION) -> $(DECOMPILED))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force ".tools" | Out-Null
	if (-not (Test-Path "$(CFR_JAR)")) { Write-Host "  fetching $(CFR_URL)" -ForegroundColor DarkCyan; Invoke-WebRequest -Uri "$(CFR_URL)" -OutFile "$(CFR_JAR)" -UseBasicParsing }
	if (Test-Path "$(DECOMPILED)") { Remove-Item -Recurse -Force "$(DECOMPILED)" }
	& "$(JAVA)" -Xmx2g -jar "$(CFR_JAR)" "$(GAME_DIR)/game-lib.jar" --outputdir "$(DECOMPILED)" --comments false --silent true; if ($$LASTEXITCODE -ne 0) { Write-Host "[decompile] CFR failed" -ForegroundColor Red; exit $$LASTEXITCODE }
	$$n = (Get-ChildItem "$(DECOMPILED)" -Recurse -Filter *.java | Measure-Object).Count; Write-Host "[decompile] $$n source files" -ForegroundColor Green
	@Write-Host "  Reminder: decompiled Java is a RECONSTRUCTION. Pin load-bearing" -ForegroundColor DarkGray
	@Write-Host "  claims to javap bytecode or observed runtime behaviour." -ForegroundColor DarkGray

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
wire-capture: agent
	@Write-Host ""
	@Write-Host "==> wire-capture (live world stream at $(WIRE_CAPTURE_AT)s -> $(WIRE_CAPTURE_OUT))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force (Split-Path -Parent "$(WIRE_CAPTURE_OUT)") | Out-Null
	if (Test-Path "$(WIRE_CAPTURE_OUT)") { Remove-Item -Force "$(WIRE_CAPTURE_OUT)" }
	& scripts/make/probe.ps1 -GameDir "$(GAME_DIR)" -AgentJar "$(AGENT_JAR)" -AgentArgs "discoverAtSeconds=$(WIRE_CAPTURE_AT);stateOutPath={OUT};exitAfterDiscovery=true" -Out "$(WIRE_CAPTURE_OUT)" -Log "runs/wire-capture.log"
	$$n = (Get-Content "$(WIRE_CAPTURE_OUT)" | Measure-Object -Line).Lines; Write-Host "[wire-capture] $$n record(s) written to $(WIRE_CAPTURE_OUT)" -ForegroundColor Green

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
type-flags: agent
	@Write-Host ""
	@Write-Host "==> type-flags (placement rules from the live registry -> $(TYPE_FLAGS_OUT))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force (Split-Path -Parent "$(TYPE_FLAGS_OUT)") | Out-Null
	& scripts/make/probe.ps1 -GameDir "$(GAME_DIR)" -AgentJar "$(AGENT_JAR)" -AgentArgs "discoverAtSeconds=$(TYPE_FLAGS_AT);typeFlagsPath={OUT};exitAfterDiscovery=true" -Out "$(TYPE_FLAGS_OUT)" -Log "runs/type-flags.log"
	$$n = (Get-Content "$(TYPE_FLAGS_OUT)" | Measure-Object -Line).Lines; Write-Host "[type-flags] $$n type(s) written to $(TYPE_FLAGS_OUT)" -ForegroundColor Green

unit-dump:
	@Write-Host ""
	@Write-Host "==> unit-dump (-printunits stat catalogue)" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$log = (Resolve-Path -LiteralPath ".").Path + "\\runs\\printunits.log"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -printunits -log $$log } finally { Pop-Location }

# Play a whole match: start a headless game with the channel open, run the
# planner against it, then stop the game.
#
# This exists because the alternative was a 400-character PowerShell
# incantation retyped every run, and one of those quietly reintroduced a bug
# the agent target had already fixed -- writing the jar in place while a game
# held it open. A target that is easy to run correctly is the fix for that.
#
# The agent is built into a per-invocation jar rather than reusing the shared
# one, and this target deliberately does not depend on `agent`. A jar held open
# by a running game cannot be replaced, so depending on the shared build makes
# playing impossible whenever any game is already up -- including one started by
# another agent working in the same tree. PLAY_PORT is per-invocation for the
# same reason.
# ":=" and not "?=", which cost a debugging round. A "?=" variable is
# recursively expanded, so $(shell ...) re-runs on every single reference: the
# game bound one port, the readiness poll watched a second, the planner dialled
# a third, and the error message named a fourth. Expanding once is the whole
# fix. A command-line override still wins over ":=".
PLAY_PORT := $(shell python -c "import random; print(random.randint(27600, 27999))")
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
# Owned by rw_bot.harness.launch (CATALOGUE / TYPE_DUMP), which is what the
# play launcher defaults to and what a sweep therefore runs under. These two
# remain only because `host` and `watch` are still PowerShell recipes -- they
# drive an interactive session on this workstation and have no cluster
# equivalent, so they were deliberately not ported with `play`. Keep them
# equal to the module's constants until those recipes follow.
CATALOGUE := wiki/sources/m0-probe/printunits.log
TYPE_DUMP := wiki/sources/m11-pools/type-flags.ndjson

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
# the argument list contains a space, and quoting it through a sub-make under
# PowerShell splits it into a stray target.
INCOME_STAGES ?= 4
INCOME_IDLE ?= 200
INCOME_LOG ?= runs/income.log

# The body lives in rw_bot.harness.play_match_cli -- launch plumbing, read
# from the working tree like this Makefile itself, while what the match RUNS
# comes from the frozen tree when PLAY_TREE is set ([[policy-loop]]).
#
# It was scripts/make/play.ps1 until the cluster port. PowerShell cannot start
# a match on a Linux compute node, and a second shell script beside the first
# would be two launchers to keep in step -- with a missed edit meaning a batch
# that runs the wrong configuration and still files a scorecard. One launcher,
# both platforms; this recipe is now just the variables it is given.
#
# The optional flags are appended with $(if ...) rather than passed empty.
# Make hands the recipe to PowerShell, which DROPS an empty argument entirely
# -- so `--map ""` arrived as a bare `--map` and swallowed the next flag as
# its value ("--map requires a value, got the flag '--opponents'"). Omitting
# them is also what a sweep does (rw_bot.harness.sweep.make_argv), so both
# callers now spell "not asked for" the same way: absent.
play:
	@Write-Host ""
	@Write-Host "==> play (headless match on port $(PLAY_PORT), log: $(PLAY_LOG))" -ForegroundColor Cyan
	poetry run python -m rw_bot.harness.play_match_cli --port $(PLAY_PORT) --game-dir "$(GAME_DIR)" --seed $(PLAY_SEED) --lockstep $(PLAY_LOCKSTEP) --settle $(PLAY_SETTLE) --play-log "$(PLAY_LOG)" --opponents $(PLAY_OPPONENTS) --difficulty $(PLAY_DIFFICULTY) --pin-delta $(PLAY_PINDELTA) --fast-forward $(PLAY_FASTFORWARD) --rng-tap $(PLAY_RNGTAP) --module $(PLAY_MODULE) --play-args "$(PLAY_ARGS)"$(if $(PLAY_MAP), --map "$(PLAY_MAP)")$(if $(PLAY_TREE), --tree "$(PLAY_TREE)")$(if $(PLAY_EXTRA_AGENT), --extra-agent-args "$(PLAY_EXTRA_AGENT)")

# Host a LAN game the bot plays in, for a human to join. The join address is
# printed at launch; the match starts the moment a player connects. Sparring
# is realtime by necessity -- no lockstep, no hold, no seed
# ([[multiplayer-portability-invariants]]).
HOST_MAP ?= maps/skirmish/[p2]duel_lake.tmx
HOST_SAMPLES ?= 20000
HOST_DOCTRINE ?= doctrines/flame-nocover.doctrine
HOST_LOG ?= runs/host.log
host:
	@Write-Host ""
	@Write-Host "==> host (bot-hosted LAN game on port 5123, log: $(HOST_LOG))" -ForegroundColor Cyan
	& scripts/make/host.ps1 -Port $(PLAY_PORT) -GameDir "$(GAME_DIR)" -PlayLog "$(HOST_LOG)" -HostMap "$(HOST_MAP)" -Module $(PLAY_MODULE) -Catalogue "$(CATALOGUE)" -TypeDump "$(TYPE_DUMP)" -PlayArgs "$(HOST_SAMPLES) $(HOST_DOCTRINE) -" -Javac "$(JAVAC)" -Jar "$(JAR)"

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
watch:
	@Write-Host ""
	@Write-Host "==> watch (windowed match at difficulty $(WATCH_DIFFICULTY), log: $(WATCH_LOG))" -ForegroundColor Cyan
	& scripts/make/watch.ps1 -Port $(PLAY_PORT) -GameDir "$(GAME_DIR)" -PlayLog "$(WATCH_LOG)" -Map "$(WATCH_MAP)" -Opponents $(WATCH_OPPONENTS) -Difficulty $(WATCH_DIFFICULTY) -Module $(PLAY_MODULE) -Catalogue "$(CATALOGUE)" -TypeDump "$(TYPE_DUMP)" -PlayArgs "$(WATCH_SAMPLES) $(WATCH_DOCTRINE) -" -Javac "$(JAVAC)" -Jar "$(JAR)"

# The match service's HTTP door, detached from the launching terminal.
# Reads the database password from the platform-postgres container at
# launch; the DSN never lands on disk.
door:
	@Write-Host ""
	@Write-Host "==> door (match service on http://127.0.0.1:27501/)" -ForegroundColor Cyan
	& scripts/make/door.ps1

# The worker fleet, detached: starts any of the 8 workers not already
# running. The whole Docker-crash recovery is `make door` (if the door
# also died) + `make fleet-up` -- the queue rows survive Postgres
# restarts and the startup reap requeues what dead owners held.
fleet-up:
	@Write-Host ""
	@Write-Host "==> fleet-up (8 detached match workers)" -ForegroundColor Cyan
	& scripts/make/fleet-up.ps1

# The match fleet: one user-owned manager process with a control page,
# spawning `make play` children (seed, map, opponents, difficulty,
# fast-forward, frozen tree are spawn parameters). PLAY_PORT already
# self-randomizes per invocation, which is what makes parallel matches
# safe. Runs in a terminal the operator owns; browse to the printed URL.
fleet: install
	@Write-Host ""
	@Write-Host "==> fleet (match fleet manager, http://127.0.0.1:27500/)" -ForegroundColor Cyan
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
	@Write-Host ""
	@Write-Host "==> sweep $(SWEEP_NAME) ($(SWEEP_WORKERS) workers, jobs: $(SWEEP_JOBS))" -ForegroundColor Cyan
	poetry run python -m scripts.sweep "$(SWEEP_JOBS)" "$(SWEEP_NAME)" $(SWEEP_WORKERS) $(SWEEP_LOCKSTEP) $(SWEEP_MATCH) $(SWEEP_PINDELTA) $(SWEEP_FASTFORWARD)

income: PLAY_MODULE = scripts.income
income: PLAY_ARGS = $(INCOME_STAGES) $(INCOME_IDLE)
income: PLAY_LOG = $(INCOME_LOG)
income: play
