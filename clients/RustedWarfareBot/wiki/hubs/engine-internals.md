# Engine Internals

What is actually inside `game-lib.jar`: the engine objects the game constructs at boot, how the ProGuard obfuscation was applied and where it wasn't, the reflection-driven surfaces that survived with readable names, and the techniques for recovering a mapping without decompiling everything.

Scope: the JVM-side game code (`com.corrodinggames.rts.*`) and its libRocket script bindings. The rules the engine *implements* (unit costs, movement, combat) belong in [Game Mechanics](game-mechanics.md); how we launch the process belongs in [Headless Harness](headless-harness.md).

Every page here is pinned to a specific game build. An obfuscated class name is a fact about one build only.

[Engine Name Oracle](../pages/engine-name-oracle.md) -- the boot log names engine objects and prints their obfuscated classes, replacing most manual mapping work
[Engine Tick Method and Clock](../pages/engine-tick-and-clock.md) -- `game.i.a(float)` is the simulation tick; `bx` counts it at ~300 Hz, `by` is the millisecond clock, and five other writers only restore them

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
