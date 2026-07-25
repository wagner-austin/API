---
title: "Headless Mode (`-nodisplay`)"
tags: [harness, headless, cli, boot]
related:
  - "[[engine-name-oracle]]"
  - "[[multiplayer-portability-invariants]]"
source_paths:
  - "wiki/sources/m0-probe/nodisplay-boot.log:309"
  - "wiki/sources/m0-probe/main-strings.txt:264"
  - "wiki/sources/m0-probe/printunits.log:1511"
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

`-nodisplay` is not a no-OpenGL mode. Slick2D still opens a display — at 10×10 — and creates framebuffer objects during boot.[^7] On a desktop with a GPU that costs nothing. On a headless Linux CI box it would still need Xvfb or one of the software-rendering flags (`-allowsoftwarerender`, `-canvasgl`), both present in the argument table and both untested.

Music tracks load and the font/glyph pipeline runs even under `-nosound` — the engine substitutes a null sound factory rather than skipping the load.[^11] Both are cheap and not worth suppressing.

## Argument inventory

Thirty-five flags appear as string constants in `Main.class`, archived at `wiki/sources/m0-probe/main-strings.txt`: `-nodisplay`,[^8] `-sandbox`, `-printunits`, `-outputunitimages`, `-debug`, `-debugscript`, `-devdebug`, `-replay_debug`, `-oldreplays`, `-nomods`, `-noresources`, `-nosound`, `-nomusic`, `-safemode`, `-extrasafemode`, `-canvasgl`, `-allowsoftwarerender`, `-disable_vbos`, `-force_vbos`, `-disable_atlas`, `-disabletextureread`, `-nopostprocessing`, `-noteamshaders`, `-postprocessing`, `-teamshaders`, `-nobackground`, `-lang`, `-log`, `-nologfile`, `-logcolor`, `-width`, `-height`, `-fullscreen`, `-steam`, `-nopreferipv4`.

Five have been exercised: `-nodisplay`, `-nosound`, `-printunits`, `-log`, `-nologfile`. The remaining thirty are inventory — the string exists in the parser, nothing more. `-sandbox` and `-debugscript` are the two most likely to matter next, and neither has been run.[^12]

## `-printunits` as a stat oracle

`-printunits` boots the engine, emits an HTML catalogue of every loaded unit, and exits without entering the game loop. Each entry carries price, HP, speed, turn speed, mass, shoot delay, attack range and direct damage — for example the Experimental Spider at $70000.[^9] Custom units from enabled mods are included, so the dump reflects the active mod set rather than vanilla alone. This is the ground truth for unit-stat claims: read it rather than measuring stats from play.

## Side effects to contain

Every run rewrites `preferences.ini` in the directory it launches from — the engine saves settings twice during boot and once more after the menu map loads.[^10] Two consequences follow: a run mutates its own game tree, and a Steam-managed install would drift and could be re-verified out from under a pinned build. Mitigation in place: the game is copied to `.game/` and every run launches from there, never from the Steam install.

## What is not yet known

Three questions are open, each with a concrete test that would settle it ([[engine-name-oracle]]).

Starting a skirmish headless has not been done. The menu loads and the script engine processes `onShowNewScreen()`, but nothing has driven it further.[^13] The test: invoke the libRocket script bindings directly (see [[engine-name-oracle]]), or run `-sandbox`, or load a save through GameSaver, and check whether the boot log reaches a real map rather than the menu background it loads today.[^14]

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
[^12]: `wiki/sources/m0-probe/main-strings.txt:306` — "-sandbox", with "-debugscript" at `:160`. Presence in the parser's string table is all this establishes; no run has exercised either.
[^13]: `wiki/sources/m0-probe/nodisplay-boot.log:305` — "ScriptEngine:HandleEvent:onShowNewScreen();", the last engine-driven event before the run entered its loop.
[^14]: `wiki/sources/m0-probe/nodisplay-boot.log:262` — "Mapfile: assets/maps/menu_background/menu2.tmx", the map the unattended process loads on its own.
[^15]: `.game/preferences.ini:148` — `slower:DEFAULT,DEFAULT`, with `faster:DEFAULT,DEFAULT` at `:149`, both in the `[keys]` section.
[^10]: `wiki/sources/m0-probe/nodisplay-boot.log:62` — "Saving settings to: C:\Program Files (x86)\Steam\steamapps\common\Rusted Warfare\preferences.ini", repeated at `:256` and `:334` — three writes in a single boot. The probe ran against the Steam install before the `.game/` copy existed.
