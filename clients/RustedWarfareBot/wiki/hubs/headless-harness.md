# Headless Harness

Running Rusted Warfare as an unattended process: command-line flags, what boots and what doesn't without a display, the pinned working copy, tick pacing and speed control, N-game batch runs, and the artifacts each run leaves behind.

Scope: everything about *operating* the game process. What the process computes internally is [Engine Internals](engine-internals.md); what the bot decides is [Bot Architecture](bot-architecture.md).

Run artifacts are archived under `runs/` and are the primary sources this wiki cites — a claim about headless behaviour should point at a log line, not at reasoning.

[Headless Mode (`-nodisplay`)](../pages/harness-nodisplay.md) -- the full engine boots and simulates with no window; flag inventory, what still needs a GL context, side effects
[Agent: Render-Callback No-Op](../pages/agent-render-callback-noop.md) -- neutralising the GUI callbacks that dereference a missing display; what unblocked a live headless skirmish

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
[Playing Matches in Parallel](../pages/harness-parallel-matches.md) -- what actually had to be separated to run several headless matches at once, and why lockstep is not optional for a batch
