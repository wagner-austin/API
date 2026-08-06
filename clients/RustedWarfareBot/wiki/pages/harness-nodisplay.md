---
title: "Headless Mode (`-nodisplay`)"
tags: [harness, headless, cli, boot]
related:
  - "[[engine-name-oracle]]"
  - "[[agent-render-callback-noop]]"
  - "[[issuing-orders]]"
  - "[[multiplayer-portability-invariants]]"
source_paths:
  - "wiki/sources/m0-probe/nodisplay-boot.log:309"
  - "wiki/sources/m0-probe/main-strings.txt:264"
  - "wiki/sources/m0-probe/printunits.log:1511"
  - "wiki/sources/m1-sandbox/sandbox-crash.log"
  - ".game/fallback64.bat"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [headless-harness]
---

# Headless Mode (`-nodisplay`)

Rusted Warfare's desktop build is a plain Java program — `com.corrodinggames.rts.java.Main` on a classpath of `game-lib.jar` plus `libs/*`, launched by a bundled OpenJDK 13.[^1] It accepts a `-nodisplay` flag that boots the entire engine and runs the simulation with no game window. No virtual framebuffer, no screen scraping, no input synthesis.

## What was verified

A run with `-nodisplay -nosound` completed engine init in **1.32 seconds**, then simulated a live match — the animated menu-background map is a real game with AI players in it — for the full 35 seconds before the process was killed.[^2] Two independent signals show the simulation itself was running, not merely the shell: the built-in AI evaluated its opening,[^3] and the mission trigger system fired scripted unit moves at game-time offsets from 50 ms through 8154 ms.[^4]

Under `-nodisplay` the asset index identifies itself as `packageName:dedicatedServer`.[^5] This is a supported mode with its own asset profile, not an accidental code path.

The process logs `steam not requested` and runs to completion with no Steam client present,[^6] so a copied game directory works standalone.

## What still initialises

`-nodisplay` is not a no-OpenGL mode. Slick2D still opens a display — at 10×10 — and creates framebuffer objects during boot.[^7] On a desktop with a GPU that costs nothing. On a headless Linux CI box it would still need Xvfb or one of the software-rendering flags (`-allowsoftwarerender`, `-canvasgl`). Both were later tried against the in-game render crash and neither avoided it, so neither is a substitute for the agent's patch ([[agent-render-callback-noop]]).

Music tracks load and the font/glyph pipeline runs even under `-nosound` — the engine substitutes a null sound factory rather than skipping the load.[^11] Both are cheap and not worth suppressing.

## Argument inventory

Thirty-five flags appear as string constants in `Main.class`, archived at `wiki/sources/m0-probe/main-strings.txt`: `-nodisplay`,[^8] `-sandbox`, `-printunits`, `-outputunitimages`, `-debug`, `-debugscript`, `-devdebug`, `-replay_debug`, `-oldreplays`, `-nomods`, `-noresources`, `-nosound`, `-nomusic`, `-safemode`, `-extrasafemode`, `-canvasgl`, `-allowsoftwarerender`, `-disable_vbos`, `-force_vbos`, `-disable_atlas`, `-disabletextureread`, `-nopostprocessing`, `-noteamshaders`, `-postprocessing`, `-teamshaders`, `-nobackground`, `-lang`, `-log`, `-nologfile`, `-logcolor`, `-width`, `-height`, `-fullscreen`, `-steam`, `-nopreferipv4`.

Nine have been exercised: `-nodisplay`, `-nosound`, `-printunits`, `-log`, `-nologfile`, `-width`, `-height`, `-sandbox`, and the render flags noted above. `-sandbox` proved the most consequential of them and is now the standard way a session starts.[^12] The rest are inventory — the string exists in the parser, nothing more.

## `-printunits` as a stat oracle

`-printunits` boots the engine, emits an HTML catalogue of every loaded unit, and exits without entering the game loop. Each entry carries price, HP, speed, turn speed, mass, shoot delay, attack range and direct damage — for example the Experimental Spider at $70000.[^9] Custom units from enabled mods are included, so the dump reflects the active mod set rather than vanilla alone. This is the ground truth for unit-stat claims: read it rather than measuring stats from play.

## Side effects to contain

Every run rewrites `preferences.ini` in the directory it launches from — the engine saves settings twice during boot and once more after the menu map loads.[^10] Two consequences follow: a run mutates its own game tree, and a Steam-managed install would drift and could be re-verified out from under a pinned build. Mitigation in place: the game is copied to `.game/` and every run launches from there, never from the Steam install.

**The shared `preferences.ini` also carries the engine's failed-load counter, and concurrent launches trip it.** `numLoadsSinceRunningGameOrNormalExit` increments when a load starts and resets on a normal exit, so four engines loading from one `.game` at the same moment each read the others' increments: the last loaders decide previous loads failed and boot into safe mode — "Started game in safe mode due to failed loading attempts. Mods have been disabled." — a popup the headless match never gets past, which the planner sees as a socket timeout on the first sample. Two runs died this way before the mechanism was read out of the log (2026-08-01); a killed-mid-load engine leaves the same footprint for the *next* single launch.[^14] Mitigation in place: launches are staggered so no two engines load at once, and a wedged engine is killed rather than left to hold the counter.

[^14]: `runs/reflex-imp2.log` — the queued messageBox line and the `Showing popup: Safe mode` line, 2026-08-01 00:54; `.game/preferences.ini` — `numLoadsSinceRunningGameOrNormalExit`, read back at 0 after the batch's normal exits.

## Open questions, and one that closed

Starting a skirmish headless was the first question here, and `-sandbox` closed it: the engine loads `maps/skirmish/[z;p10]Crossing Large (10p).tmx` with no human, the same default the single-player menu wires to that button.[^13] It needs an explicit `-width`/`-height`, because the 10x10 display `-nodisplay` selects on its own fails once in-game UI renders ([[agent-render-callback-noop]]). The route this section originally proposed — driving the libRocket script bindings — turned out to be unnecessary for starting a game, though the same surface later supplied the order path and `Root.hostStart(boolean)` ([[issuing-orders]]).

Two remain, each with a concrete test ([[engine-name-oracle]]).

Clean self-termination has not been observed — the 35-second run was killed externally.[^2] The test: run a mission that can end, and check the process exit code.

Faster-than-realtime ticking has not been attempted. The engine exposes `slower` and `faster` as bindable key actions, so a speed control exists; whether it is reachable without the UI is unknown.[^15] The test: locate the speed setter via the boot-log class mapping ([[engine-name-oracle]]) and call it from the agent.

[^1]: `.game/fallback64.bat` — `jvm64\bin\java -Xmx1000M -Dfile.encoding=UTF-8 -Djava.library.path=. -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main -width 800 -height 600`. JVM version from `.game/jvm64/bin/java.exe -version` → "openjdk version 13 2019-09-17".
[^2]: `wiki/sources/m0-probe/nodisplay-boot.log:309` — "----- Game init finished in:1322.0735 ms". The process was still alive at the 35 s kill. Build pinned at `wiki/sources/m0-probe/nodisplay-boot.log:12` — "Game Version: 1.15".
[^3]: `wiki/sources/m0-probe/nodisplay-boot.log:310` — "ai_debug(3):firstRun: no command center found".
[^4]: `wiki/sources/m0-probe/nodisplay-boot.log:336` — "MissionEngine:triggerLog:firstActivation: move at:8154 for teamId:1 to targetId:6 (#units:4)".
[^5]: `wiki/sources/m0-probe/nodisplay-boot.log:53` — "packageName:dedicatedServer".
[^6]: `wiki/sources/m0-probe/nodisplay-boot.log:18` — "steam not requested".
[^7]: `wiki/sources/m0-probe/nodisplay-boot.log:20` — "--- ERROR: Skipping display mode call" — followed at `:27` by "INFO:Starting display 10x10"; FBO-creation lines recur throughout boot.
[^8]: `wiki/sources/m0-probe/main-strings.txt:264` — "-nodisplay". The file is the sorted, deduplicated set of printable string constants extracted from `com/corrodinggames/rts/java/Main.class` inside `.game/game-lib.jar`; `-printunits` at `:297` and `-sandbox` at `:306`.
[^9]: `wiki/sources/m0-probe/printunits.log:1509` — "<h4>Experimental Spider</h4>", with "<pre>Price: $70000" at `:1511` and the HP/speed/mass/range block immediately following.
[^11]: `wiki/sources/m0-probe/nodisplay-boot.log:33` — "Disabling sound with NullSoundFactory"; the music-track load lines follow at `:75`–`:93`.
[^12]: `wiki/sources/m0-probe/main-strings.txt:306` — "-sandbox", with "-debugscript" at `:160`. The former is exercised by every session since; the latter is still only a string in the parser.
[^13]: `wiki/sources/m1-sandbox/sandbox-crash.log` — `Mapfile: assets/maps/skirmish/[z;p10]Crossing Large (10p).tmx` under `-sandbox`, against `assets/maps/menu_background/menu2.tmx` at `wiki/sources/m0-probe/nodisplay-boot.log:262` without it.
[^15]: `.game/preferences.ini:148` — `slower:DEFAULT,DEFAULT`, with `faster:DEFAULT,DEFAULT` at `:149`, both in the `[keys]` section.
[^10]: `wiki/sources/m0-probe/nodisplay-boot.log:62` — "Saving settings to: C:\Program Files (x86)\Steam\steamapps\common\Rusted Warfare\preferences.ini", repeated at `:256` and `:334` — three writes in a single boot. The probe ran against the Steam install before the `.game/` copy existed.
