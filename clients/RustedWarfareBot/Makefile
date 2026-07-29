SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

# Path to the pinned game working copy. Every game invocation launches
# from here, never from the Steam install: runs rewrite preferences.ini,
# and Steam would re-verify the build out from under the class mappings.
GAME_DIR ?= .game

.PHONY: help lint test check install boot-probe unit-dump type-flags wire-capture agent agent-selftest sandbox-probe decompile play income sweep

help:
	@Write-Host ""
	@Write-Host "RustedWarfareBot -- available targets" -ForegroundColor Cyan
	@Write-Host "=====================================" -ForegroundColor Cyan
	@Write-Host ""
	@Write-Host "Code health:" -ForegroundColor Yellow
	@Write-Host "  make check                       lint + test (the gate)"
	@Write-Host "  make lint                        guard + ruff + mypy over src, tests, scripts"
	@Write-Host "  make test                        pytest + coverage (100% statements and branches)"
	@Write-Host "  make install                     poetry install"
	@Write-Host ""
	@Write-Host "Game probes (need the pinned copy at $(GAME_DIR)):" -ForegroundColor Yellow
	@Write-Host "  make boot-probe                  headless boot, archive the engine log"
	@Write-Host "  make unit-dump                   -printunits stat catalogue"
	@Write-Host "  make type-flags                  placement rules from the live type registry"
	@Write-Host "  make wire-capture                regenerate the archived world-stream fixture"
	@Write-Host "  make sandbox-probe               live skirmish with the agent attached"
	@Write-Host ""
	@Write-Host "Play:" -ForegroundColor Yellow
	@Write-Host "  make play                        start a game, play a whole match, stop it"
	@Write-Host "                                   PLAY_SAMPLES=1500 PLAY_SEED=0 PLAY_LOG=runs/play.log"
	@Write-Host "                                   PLAY_SEED pins the opponents' RNG; PLAY_LOCKSTEP locks the tick"
	@Write-Host "  make income                      measure extractor credits/s and payback time"
	@Write-Host "                                   INCOME_STAGES=4 INCOME_IDLE=200 INCOME_LOG=runs/income.log"
	@Write-Host "  make sweep                       play a batch of matches in parallel, resumably"
	@Write-Host "                                   SWEEP_JOBS=sweeps/x.txt SWEEP_NAME=x SWEEP_WORKERS=4"
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
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$env:COVERAGE_FILE = (Resolve-Path -LiteralPath ".").Path + "\runs\.coverage-" + [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$code = 0; try { poetry run python -m pytest --cov=src --cov=scripts --cov-branch --cov-report=term-missing; $$code = $$LASTEXITCODE } finally { Get-ChildItem -Path "runs" -Filter ".coverage-*" -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; Remove-Item Env:\COVERAGE_FILE -ErrorAction SilentlyContinue }; if ($$code -ne 0) { exit $$code }
	@Write-Host ""
	@Write-Host "[test] passed" -ForegroundColor Green

# The Java half is gated here too. Without agent-selftest in the chain, a
# patcher regression or an obfuscated name that moved in a game update stays
# invisible until a live run dies in the engine, which is the worst place to
# read a build error. It needs the pinned game copy, which every other game
# target already requires.
check: lint | test agent-selftest
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
	@Write-Host "==> agent (javac --release 11, zero third-party deps)" -ForegroundColor Cyan
	$$dir = "agent/build/classes-" + [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$tmp = "$(AGENT_JAR)." + [System.Guid]::NewGuid().ToString("N").Substring(0,8) + ".new"; $$failed = ""; try { New-Item -ItemType Directory -Force $$dir | Out-Null; & "$(JAVAC)" --release 11 -Xlint:all -Werror -d $$dir (Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $$_.FullName }); if ($$LASTEXITCODE -ne 0) { $$failed = "javac failed" } else { & "$(JAR)" cfm $$tmp agent/manifest.mf -C $$dir .; if ($$LASTEXITCODE -ne 0) { $$failed = "jar failed" } else { try { Move-Item -Force -ErrorAction Stop -LiteralPath $$tmp -Destination "$(AGENT_JAR)" } catch { $$failed = "cannot replace $(AGENT_JAR): a JVM has it attached with -javaagent. Stop the running game, then retry." } } } } finally { if (Test-Path $$dir) { Remove-Item -Recurse -Force $$dir }; if (Test-Path $$tmp) { Remove-Item -Force $$tmp } }; if ($$failed) { Write-Host "[agent] $$failed" -ForegroundColor Red; exit 1 }
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
	$$dir = "agent/build/verify-" + [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$failed = ""; try { New-Item -ItemType Directory -Force $$dir | Out-Null; & "$(JAVAC)" --release 11 -Xlint:all -Werror -d $$dir (Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $$_.FullName }); if ($$LASTEXITCODE -ne 0) { $$failed = "javac failed" } else { & "$(JAVA)" -cp "$$dir;$(GAME_DIR)/game-lib.jar;$(GAME_DIR)/libs/*" rwbot.agent.SelfTest "$(GAME_DIR)/game-lib.jar"; if ($$LASTEXITCODE -ne 0) { $$failed = "self-checks failed" } } } finally { if (Test-Path $$dir) { Remove-Item -Recurse -Force $$dir } }; if ($$failed) { Write-Host "[agent-selftest] $$failed" -ForegroundColor Red; exit 1 }
	@Write-Host "[agent-selftest] passed" -ForegroundColor Green

# Live skirmish with the agent attached. Runs until killed -- the engine logs
# only events, so a quiet log here means a running sim, not a stalled one.
SANDBOX_LOG ?= runs/sandbox-probe.log
sandbox-probe: agent
	@Write-Host ""
	@Write-Host "==> sandbox-probe (live skirmish, agent attached, log: $(SANDBOX_LOG))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force (Split-Path -Parent "$(SANDBOX_LOG)") | Out-Null
	$$root = (Resolve-Path -LiteralPath ".").Path; $$agent = $$root + "\\$(AGENT_JAR)"; $$log = $$root + "\\$(SANDBOX_LOG)"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." "-javaagent:$$agent" -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -sandbox -width 800 -height 600 -log $$log } finally { Pop-Location }

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
	$$root = (Resolve-Path -LiteralPath ".").Path; $$agent = $$root + "\\$(AGENT_JAR)"; $$log = $$root + "\\$(DISCOVER_OUT)"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." "-javaagent:$$agent=discoverAtSeconds=$(DISCOVER_AT);exitAfterDiscovery=true" -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -sandbox -width 800 -height 600 -log $$log } finally { Pop-Location }
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
	$$root = (Resolve-Path -LiteralPath ".").Path; $$agent = $$root + "\\$(AGENT_JAR)"; $$out = $$root + "\\$(WIRE_CAPTURE_OUT)"; $$log = $$root + "\\runs\\wire-capture.log"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." "-javaagent:$$agent=discoverAtSeconds=$(WIRE_CAPTURE_AT);stateOutPath=$$out;exitAfterDiscovery=true" -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -sandbox -width 800 -height 600 -log $$log } finally { Pop-Location }
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
	$$root = (Resolve-Path -LiteralPath ".").Path; $$agent = $$root + "\\$(AGENT_JAR)"; $$out = $$root + "\\$(TYPE_FLAGS_OUT)"; $$log = $$root + "\\runs\\type-flags.log"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." "-javaagent:$$agent=discoverAtSeconds=$(TYPE_FLAGS_AT);typeFlagsPath=$$out;exitAfterDiscovery=true" -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -sandbox -width 800 -height 600 -log $$log } finally { Pop-Location }
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

# How many extractors the income probe builds, and how long it stands still
# between each one. Target-specific variables rather than a recursive $(MAKE):
# the argument list contains a space, and quoting it through a sub-make under
# PowerShell splits it into a stray target.
INCOME_STAGES ?= 4
INCOME_IDLE ?= 200
INCOME_LOG ?= runs/income.log

play:
	@Write-Host ""
	@Write-Host "==> play (headless match on port $(PLAY_PORT), log: $(PLAY_LOG))" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$stamp = [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$dir = "agent/build/play-$$stamp"; $$jar = "agent/build/rw-agent-play-$$stamp.jar"; $$root = (Resolve-Path -LiteralPath ".").Path; $$log = $$root + "\$(PLAY_LOG)"; try { New-Item -ItemType Directory -Force $$dir | Out-Null; & "$(JAVAC)" --release 11 -Xlint:all -Werror -d $$dir (Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $$_.FullName }); if ($$LASTEXITCODE -ne 0) { Write-Host "[play] javac failed" -ForegroundColor Red; exit 1 }; & "$(JAR)" cfm $$jar agent/manifest.mf -C $$dir .; if ($$LASTEXITCODE -ne 0) { Write-Host "[play] jar failed" -ForegroundColor Red; exit 1 }; $$agentArgs = "channelPort=$(PLAY_PORT)"; if ($(PLAY_SEED) -ne 0) { $$agentArgs = $$agentArgs + ";randomSeed=$(PLAY_SEED)"; Write-Host "[play] engine random pinned to seed $(PLAY_SEED)" -ForegroundColor DarkGray }; if ($(PLAY_LOCKSTEP) -ne 0) { $$agentArgs = $$agentArgs + ";lockstepFrames=$(PLAY_LOCKSTEP)"; Write-Host "[play] lockstep every $(PLAY_LOCKSTEP) frames" -ForegroundColor DarkGray }; if ("$(PLAY_MAP)" -ne "") { $$agentArgs = $$agentArgs + ";matchMap=$(PLAY_MAP);matchOpponents=$(PLAY_OPPONENTS);matchDifficulty=$(PLAY_DIFFICULTY)"; Write-Host "[play] $(PLAY_OPPONENTS) opponent(s) at difficulty $(PLAY_DIFFICULTY) on $(PLAY_MAP)" -ForegroundColor DarkGray }; $$gameArgs = @("-Xmx1000M", "--add-opens", "java.base/java.lang=ALL-UNNAMED", "-Djava.library.path=.", "-javaagent:$$root\$$jar=$$agentArgs", "-cp", "game-lib.jar;libs/*", "com.corrodinggames.rts.java.Main", "-nodisplay", "-nosound"); if ("$(PLAY_MAP)" -eq "") { $$gameArgs += "-sandbox" } else { Write-Host "[play] -sandbox withheld; the agent starts the match" -ForegroundColor DarkGray }; $$gameArgs += @("-width", "800", "-height", "600", "-log", "$$log"); $$game = Start-Process -FilePath "$$root\$(GAME_DIR)\jvm64\bin\java.exe" -ArgumentList $$gameArgs -WorkingDirectory "$$root\$(GAME_DIR)" -PassThru -RedirectStandardOutput "$$root\$(PLAY_LOG).agent" -RedirectStandardError "$$root\$(PLAY_LOG).err"; try { $$sw = [Diagnostics.Stopwatch]::StartNew(); $$open = $$false; while ($$sw.Elapsed.TotalSeconds -lt 90) { try { $$c = New-Object Net.Sockets.TcpClient("127.0.0.1", $(PLAY_PORT)); $$c.Close(); $$open = $$true; break } catch { Start-Sleep -Milliseconds 1000 } }; if (-not $$open) { Write-Host "[play] the agent never opened port $(PLAY_PORT)" -ForegroundColor Red; exit 1 }; Write-Host "[play] channel open; letting the map settle $(PLAY_SETTLE)s" -ForegroundColor DarkGray; Start-Sleep -Seconds $(PLAY_SETTLE); poetry run python -m $(PLAY_MODULE) $(PLAY_PORT) "$(CATALOGUE)" "$(TYPE_DUMP)" $(PLAY_ARGS) } finally { Stop-Process -Id $$game.Id -Force -ErrorAction SilentlyContinue; Write-Host "[play] game stopped" -ForegroundColor DarkGray } } finally { if (Test-Path $$dir) { Remove-Item -Recurse -Force $$dir }; Start-Sleep -Milliseconds 500; if (Test-Path $$jar) { Remove-Item -Force $$jar -ErrorAction SilentlyContinue } }

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

sweep:
	@Write-Host ""
	@Write-Host "==> sweep $(SWEEP_NAME) ($(SWEEP_WORKERS) workers, jobs: $(SWEEP_JOBS))" -ForegroundColor Cyan
	poetry run python -m scripts.sweep "$(SWEEP_JOBS)" "$(SWEEP_NAME)" $(SWEEP_WORKERS) $(SWEEP_LOCKSTEP) $(SWEEP_MATCH)

income: PLAY_MODULE = scripts.income
income: PLAY_ARGS = $(INCOME_STAGES) $(INCOME_IDLE)
income: PLAY_LOG = $(INCOME_LOG)
income: play
