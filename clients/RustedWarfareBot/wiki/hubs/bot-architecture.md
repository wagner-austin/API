# Bot Architecture

The bot's own design: the perception layer and its visibility filter, the planner (build orders, economy/army arbitration, attack timing), the dispatch path into the game's command queue, the ledger correlating decisions to outcomes, coding standards, and the behaviour contracts that make bad states unreachable.

Scope: code we write. The game side is [Engine Internals](engine-internals.md) and [Headless Harness](headless-harness.md).

Two inherited principles from the sibling TankpitBot project, adopted here deliberately rather than rediscovered: **fail hard on state entry, not soft on state observation** (contract violations raise on the transition, never after N observations of the consequences), and **single-owner decisions** (any question like "can this unit path here" has exactly one owner; a second validator downstream produces silent rejection loops).

[Runtime Split — Java Agent, Python Brain](../pages/runtime-split-java-agent-python-brain.md) -- why two processes, why those languages, decimated decision rate, standing orders for per-tick reaction
[Multiplayer Portability Invariants](../pages/multiplayer-portability-invariants.md) -- the four rules that keep single-player work from stranding us outside multiplayer
[Agent: Render-Callback No-Op](../pages/agent-render-callback-noop.md) -- the agent's first job: bytecode-patching the render callbacks that kill a headless engine, and why the verifier is the oracle
[Engine Tick Method and Clock](../pages/engine-tick-and-clock.md) -- the tick basis a decimated planner decimates against, and the safe read path to the live engine

[Issuing Orders](../pages/issuing-orders.md) -- the three-call command path, its threading rule, and the order that finally moved a unit

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
