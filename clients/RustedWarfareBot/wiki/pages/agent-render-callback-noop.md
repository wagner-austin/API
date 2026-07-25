---
title: "Agent: Render-Callback No-Op"
tags: [agent, harness, bytecode, instrumentation, headless]
related:
  - "[[harness-nodisplay]]"
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[multiplayer-portability-invariants]]"
source_paths:
  - "wiki/sources/m1-sandbox/sandbox-crash.log:405"
  - "wiki/sources/m2-agent/sandbox-agent-stdout.txt:1"
  - "wiki/sources/m2-agent/sandbox-agent-running.log:396"
  - "wiki/sources/m2-agent/sandbox-agent-running.log:402"
  - "wiki/sources/m2-agent/sandbox-agent-jstack.txt:3"
  - "wiki/sources/m2-agent/sandbox-agent-jstack.txt:19"
  - "agent/src/rwbot/agent/Targets.java"
  - "agent/src/rwbot/agent/ClassFilePatcher.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [bot-architecture, headless-harness]
---

# Agent: Render-Callback No-Op

The javaagent's first job is not dispatch but survival: it neutralises the GUI render callbacks that dereference a display which does not exist headless, so the engine reaches a live simulation instead of dying on its first in-game frame ([[harness-nodisplay]]).

## The blocker

Booting `-sandbox` reached a fully loaded skirmish and then threw `NullPointerException` in `com.corrodinggames.rts.java.d.a.EnableScissorRegion`, called from the native method `com.LibRocket.render`.[^1] It is a JNI callback: LibRocket's native renderer calls back into Java once per GUI frame, and the callback touches field `j`, an `org.newdawn.slick.Graphics` that is only populated when a real display exists.[^1]

## Why a bare `return` is the correct patch

Disassembly shows the method does exactly two things — `setWorldClip`/`clearWorldClip` on field `j`, and maintaining the boolean flag `h`.[^7] Neither touches simulation state, which is what makes neutralising it multiplayer-legal rather than a rules change ([[multiplayer-portability-invariants]]).

Preserving the flag as `h = enabled` looks more faithful but is actively worse.[^7] Field `h` has exactly three references in the class: the two writes inside this method, and one read in `RenderGeometryPossiblyCompiled` guarding a branch that dereferences field `g`.[^7] Leaving `h` permanently `false` therefore disarms the only reader; preserving it would arm a *second* null dereference rather than avoid one.[^7]

## Patching without a bytecode library

The patch is applied by a hand-rolled class-file rewriter rather than ASM or Javassist, because the agent loads into the game's own classloader beside obfuscated classes where every added dependency is a conflict surface ([[runtime-split-java-agent-python-brain]]).[^8]

What keeps that tractable is a single property: **a no-op body references no constant-pool entries.** The pool is therefore parsed only to find where it ends and is copied through byte-for-byte, and the edit stays local to one `Code` attribute — no length field outside it needs recomputing, because `method_info` carries no total length and the class file has no trailing size.[^8]

## The verifier is the oracle

Correctness is certified by defining and linking each patched class, so HotSpot's own bytecode verifier passes judgement rather than a second reading of the bytes by the code that wrote them.[^8] `make agent-selftest` runs this against the real pinned jar and is the check to re-run after any game update, since obfuscated names move silently between builds.[^6]

## Result

With the agent attached, the engine loads the map, builds PathEngine costs, reaches `--- setRunning ---`, creates the minimap and reports `--- Mouse API succeeded` — no exception.[^2][^3] The agent reports its own patch on stdout before the engine starts.[^4]

The simulation genuinely ticks rather than sitting live-but-frozen: two thread dumps twelve seconds apart show the game thread's CPU time advancing from 4,984 ms to 7,968 ms.[^5] Both samples caught it inside `Display.sync` → `Thread.sleep`, meaning the loop is frame-rate limited — a healthy running loop, and a standing hint that uncapping that sync is available if the bot ever needs faster-than-realtime play.[^5]

## What this does not settle

`RenderGeometryPossiblyCompiled` dereferences the same null field `j` at roughly ten further offsets, so the render path is not proven clean — only proven to survive the frames observed here.[^7] The target list is deliberately a list for that reason: the next callback to surface is a one-line addition to `Targets`.[^6]

[^1]: `wiki/sources/m1-sandbox/sandbox-crash.log:405` — `java.lang.NullPointerException at com.corrodinggames.rts.java.d.a.EnableScissorRegion(SourceFile:650)`, immediately below `at com.LibRocket.render(Native Method)`.
[^2]: `wiki/sources/m2-agent/sandbox-agent-running.log:396` — `--- setRunning ---`, following `PathEngine: Ready` at `:393`.
[^3]: `wiki/sources/m2-agent/sandbox-agent-running.log:402` — `--- Mouse API succeeded`, following `Minimap map render took:10.969ms` at `:401`. The log has no exception entry.
[^4]: `wiki/sources/m2-agent/sandbox-agent-stdout.txt:1` — `[rw-agent] patched com/corrodinggames/rts/java/d/a [EnableScissorRegion(Z)V]`.
[^5]: `wiki/sources/m2-agent/sandbox-agent-jstack.txt:3` and `:19` — `Thread-2` at `cpu=4984.38ms elapsed=12.00s` and `cpu=7968.75ms elapsed=24.13s`, both stacked `Thread.sleep` ← `org.lwjgl.opengl.Sync.sync` ← `Display.sync` ← `updateAndRender`.
[^6]: `agent/src/rwbot/agent/Targets.java` — the class-to-methods map, one entry per neutralised callback, with the pinned-build warning and the hard-failure contract described in its javadoc.
[^7]: `javap -p -c -cp .game/game-lib.jar com.corrodinggames.rts.java.d.a` against the pinned build [synthesis] — `EnableScissorRegion` is 64 bytes ending at offset 63; the three `Field h:Z` references are the writes at offsets 45 and 60 and the read at 275, whose `ifeq` guards `getfield g:Landroid/graphics/RectF;` at 284. Ten `getfield j` sites appear at offsets 418 through 1045. The `.game/` tree is untracked by design ([[harness-nodisplay]]), so this command is the reproduction path rather than an archived artifact.
[^8]: `agent/src/rwbot/agent/ClassFilePatcher.java` — the constant-pool walk, the local `Code`-attribute splice, and `SelfTest`'s define-and-link verification, each documented in the class javadoc.
