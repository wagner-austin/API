"""The fifth wake bridge: the fleet audit's health journal to the agent board.

The family, and where this one sits:

    hpc-wake           slurm ledger          -> board
    fleet-wake         dispatch ledger       -> board
    ci-wake            GitHub Actions        -> board
    lock-wake          fleet-lock journal    -> board
    fleet-health-wake  fleet health journal  -> board   THIS PACKAGE

The journal is MCPs ``fleet-mcp/state/health-events.jsonl``, appended by the
fleet audit (``fleet-mcp/src/health-journal.ts``) whenever a run's outcome
is news: a row newly failing or recovered, a node gone unreachable or
answering again, a previous snapshot that could not be read, or the audit
refusing to run on a stale build. Each line carries its note already
rendered, so this package decodes and posts; it never re-derives what
changed (MCPs board task ebc80a03).

Same bones as lock-wake, lifted not forked: ``platform_core.journal_cursor``
for the byte-offset cursor, ``platform_core.board`` for identity and the
post, ``board_watch.config.load_credentials`` for secrets, the post before
the position (at-least-once), the standing task as configuration, and a
deterministic service identity. Its cycle runs as a row in the pump's
PUBLISHERS table (``tools/hpc-wake/scripts/run_cycle.py``), never as its own
scheduled task.
"""
