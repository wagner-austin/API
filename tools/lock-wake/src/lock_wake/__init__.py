"""The fourth wake bridge: fleet-lock transitions to the agent board.

The family, and where this one sits:

    hpc-wake     slurm ledger       -> board
    fleet-wake   dispatch ledger    -> board
    ci-wake      GitHub Actions     -> board
    lock-wake    fleet-lock journal -> board   THIS PACKAGE

The journal is ``scripts/with-fleet-lock.ps1``'s append-only record in the
MCPs repo root (``.fleet-events.jsonl``): one JSON object per transition —
requested, waiting, acquired, step, released, failed, timeout — written by
every locked compose target. It exists precisely so a subscriber "keeps a
byte offset, reads from it, and cannot miss a transition at any polling
interval". This package is that subscriber, publishing the transitions the
journal already records so a cascade starting or ending WAKES the sessions
that care instead of each one tailing the file (board task 9406cfd9,
publisher b).

Same bones as the siblings, lifted not forked: ``platform_core.board`` for
identity and the post shape, ``board_watch.config.load_credentials`` for
secrets, announcements POST before the position advances (at-least-once),
the standing task as configuration, and a deterministic service identity.
Its cycle runs as a row in the pump's PUBLISHERS table
(``tools/hpc-wake/scripts/run_cycle.py``), never as its own scheduled task.
"""
