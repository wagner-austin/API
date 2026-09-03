# Headless Harness

Running Rusted Warfare as an unattended process: command-line flags, what boots and what doesn't without a display, the pinned working copy, tick pacing and speed control, N-game batch runs, and the artifacts each run leaves behind.

Scope: everything about *operating* the game process. What the process computes internally is [Engine Internals](engine-internals.md); what the bot decides is [Bot Architecture](bot-architecture.md).

Run artifacts are archived under `runs/` and are the primary sources this wiki cites — a claim about headless behaviour should point at a log line, not at reasoning.

[Headless Mode (`-nodisplay`)](../pages/harness-nodisplay.md) -- the full engine boots and simulates with no window; flag inventory, what still needs a GL context, side effects
[Agent: Render-Callback No-Op](../pages/agent-render-callback-noop.md) -- neutralising the GUI callbacks that dereference a missing display; what unblocked a live headless skirmish

[Determinism policy](../pages/policy-determinism.md) -- what the harness pins per run so a verdict is attributable to the change and not the environment
[Trace policy](../pages/policy-trace.md) -- the r01..r12 worth traces, peak-vs-final shape, and what a trace is allowed to conclude
<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
[Playing Matches in Parallel](../pages/harness-parallel-matches.md) -- what actually had to be separated to run several headless matches at once, and why lockstep is not optional for a batch

[The Exact-Timing Regime — the Ladder Re-Founded](../pages/policy-exact-timing.md) -- the certified seed-pure simulation, the un-handicapped AI it revealed, the boot-sandbox compile trap, and the honest ladder every new rate is measured against

[The Match Service: Engine Slots Become a Queue](../pages/harness-match-service.md) -- the Postgres queue, leased clones and ports, the HTTP door, the results mirror, and the dashboard; how a panel became one submission
[Run Lifecycle: Why Ad-Hoc Jobs Stopped Being Enough](../pages/harness-run-lifecycle.md) -- the failure modes of detached shell jobs at concurrency, and the requirements that produced the match service

[The Doctrine Search: Screening With a Confirmatory Backstop](../pages/harness-doctrine-search.md) -- the dense margin, successive halving over knob combinations, what the method is and is not, and the corners it cuts, named
[Population Search — The Parameterization Design](../pages/harness-population-search.md) -- the "learn by 10,000 matches" answer: a genome (composition simplex + knobs + lifted constants) compiling to ordinary doctrine files, CMA-ES over the existing array machinery; v1 built and running as scripts/evolve.py
[Driving the Research — The Operating Page](../pages/harness-driving-the-research.md) -- what a fresh session runs: the four canonical commands, the boundary-relaunch posture, seed law, and where every verdict must land
