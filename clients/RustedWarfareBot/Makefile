SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

# Path to the pinned game working copy. Every game invocation launches
# from here, never from the Steam install: runs rewrite preferences.ini,
# and Steam would re-verify the build out from under the class mappings.
GAME_DIR ?= .game

.PHONY: help lint test check install boot-probe unit-dump agent agent-selftest sandbox-probe

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
	@Write-Host "  make sandbox-probe               live skirmish with the agent attached"
	@Write-Host ""
	@Write-Host "Java agent:" -ForegroundColor Yellow
	@Write-Host "  make agent                       compile + jar the javaagent"
	@Write-Host "  make agent-selftest              patch the real jar, verify via the JVM verifier"
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

test:
	@Write-Host ""
	@Write-Host "==> test" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	poetry run python -m pytest --cov=src --cov=scripts --cov-branch --cov-report=term-missing
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

agent:
	@Write-Host ""
	@Write-Host "==> agent (javac --release 11, zero third-party deps)" -ForegroundColor Cyan
	if (Test-Path "agent/build/classes") { Remove-Item -Recurse -Force "agent/build/classes" }
	New-Item -ItemType Directory -Force "agent/build/classes" | Out-Null
	& "$(JAVAC)" --release 11 -Xlint:all -Werror -d agent/build/classes (Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $$_.FullName }); if ($$LASTEXITCODE -ne 0) { Write-Host "[agent] javac failed" -ForegroundColor Red; exit $$LASTEXITCODE }
	& "$(JAR)" cfm "$(AGENT_JAR)" agent/manifest.mf -C agent/build/classes .
	@Write-Host "[agent] built $(AGENT_JAR)" -ForegroundColor Green

# Patches every Targets entry in the real pinned jar and DEFINES each result,
# so the JVM's own bytecode verifier -- not a second reading by the code that
# wrote the bytes -- is what certifies the patch. Run this after any game
# update: obfuscated names move silently between builds.
agent-selftest: agent
	@Write-Host ""
	@Write-Host "==> agent-selftest (JVM verifier as oracle)" -ForegroundColor Cyan
	& "$(JAVA)" -cp "$(AGENT_CP)" rwbot.agent.SelfTest "$(GAME_DIR)/game-lib.jar"; if ($$LASTEXITCODE -ne 0) { Write-Host "[agent-selftest] FAILED" -ForegroundColor Red; exit $$LASTEXITCODE }
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

unit-dump:
	@Write-Host ""
	@Write-Host "==> unit-dump (-printunits stat catalogue)" -ForegroundColor Cyan
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$log = (Resolve-Path -LiteralPath ".").Path + "\\runs\\printunits.log"; Push-Location "$(GAME_DIR)"; try { & ".\\jvm64\\bin\\java.exe" -Xmx1000M "-Djava.library.path=." -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -nodisplay -nosound -printunits -log $$log } finally { Pop-Location }
