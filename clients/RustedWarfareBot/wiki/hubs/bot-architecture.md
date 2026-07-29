# Bot Architecture

The bot's own design: the perception layer and its visibility filter, the planner (build orders, economy/army arbitration, attack timing), the dispatch path into the game's command queue, the ledger correlating decisions to outcomes, coding standards, and the behaviour contracts that make bad states unreachable.

Scope: code we write. The game side is [Engine Internals](engine-internals.md) and [Headless Harness](headless-harness.md).

Two inherited principles from the sibling TankpitBot project, adopted here deliberately rather than rediscovered: **fail hard on state entry, not soft on state observation** (contract violations raise on the transition, never after N observations of the consequences), and **single-owner decisions** (any question like "can this unit path here" has exactly one owner; a second validator downstream produces silent rejection loops).

[Runtime Split — Java Agent, Python Brain](../pages/runtime-split-java-agent-python-brain.md) -- why two processes, why those languages, decimated decision rate, standing orders for per-tick reaction
[Multiplayer Portability Invariants](../pages/multiplayer-portability-invariants.md) -- the four rules that keep single-player work from stranding us outside multiplayer
[Agent: Render-Callback No-Op](../pages/agent-render-callback-noop.md) -- the agent's first job: bytecode-patching the render callbacks that kill a headless engine, and why the verifier is the oracle
[Engine Tick Method and Clock](../pages/engine-tick-and-clock.md) -- the tick basis a decimated planner decimates against, and the safe read path to the live engine

[Perception: Visible Entities, Economy and Health](../pages/perception-visibility.md) -- enemies, credits and hit points, gated through the engine's own fog test rather than the master list
[Wire Contract — NDJSON World Stream](../pages/wire-contract-ndjson.md) -- the agent publishes the roster as flat NDJSON; why flatness is forced by the planner's type checker
[Unit Catalogue and the Mobility Predicate](../pages/mechanics-unit-catalogue.md) -- what a unit costs and whether it can move at all, read rather than discovered by ordering
[Issuing Orders](../pages/issuing-orders.md) -- the three-call command path, its threading rule, and the order that finally moved a unit

[Command Channel](../pages/command-channel.md) -- orders originate in Python: one loopback socket, id-addressed units, and the backpressure rule that protects the tick

[The Budget — One Authority Over a Tick's Credits](../pages/policy-budget.md) -- two spenders and one balance, and the ordered claim that fixed it
[The Verdict — Asking the Engine Who Won](../pages/policy-verdict.md) -- the engine's own flags, and why they replaced every proxy for them
[The Policy Loop](../pages/policy-loop.md) -- the bot plays: pure decisions from observed state, one order per plan slot, and a scorecard

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
[The Build Tree, and Planning From Goals](../pages/mechanics-build-tree.md) -- goals in, executable plan out: prerequisites derived rather than hand-written
[Threat: Choosing Ground the Builder Survives](../pages/policy-threat.md) -- pools are chosen by who can shoot the walk there, with hostility read from the engine's alliance test rather than from ownership
[How the Built-in AI Plays](../pages/ai-opponent-strategy.md) -- what the bot is losing to, read from the jar rather than reasoned about
[Fighting: the Attack Verb, and Keeping an Army Alive](../pages/policy-combat.md) -- how the bot attacks, why it reinforces, and the target churn that is left
[The AI Zone Probe, and Why It Is Not Perception](../pages/engine-ai-probe.md) -- reading the opposing AI's plans is research, never perception: why, and the two structural guarantees that keep it that way
[Movement Layers and Reachability](../pages/mechanics-movement-layers.md) -- eight layers named by the engine, reachability as a component comparison, and the twelve pools no land builder can reach
[What a Credit Buys — The Unit Value Table](../pages/mechanics-unit-value.md) -- the ranking the army composition is argued from, and the squared-damage error that briefly inverted it
[Production — Keeping the Queues Full](../pages/policy-production.md) -- why a priority list cannot express an army, and the three measured failures behind the composition rule
[Playing Matches in Parallel](../pages/harness-parallel-matches.md) -- the sweep harness: jobs as data, resumable by construction, one result file per match
[Holding Ground - 44 of 46 Pools, and Why the Bot Loses](../pages/policy-holding-ground.md) -- the bot loses at full length; expansion without defence is a credit shredder
