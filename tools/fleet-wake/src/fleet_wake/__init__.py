"""Announce fleet dispatch results on the agent board.

The producer that closes ``tools/fleet``'s own acceptance criterion 6. The
dispatcher records every result perfectly and tells nobody: a terminal
outcome reaches ``runs/ledger.jsonl`` and ``runs/feed.jsonl`` on THIS machine,
so the only ways to learn that a dispatch finished were to be the session that
ran it or to read a file here. Every other session -- on the phone, on
claude.ai, in another repo -- had no way to find out at all.

One cycle, one job: read the fleet ledger's CURRENT row per dispatch, take
the ones that have reached a terminal outcome and have not been announced,
post one board note per (agent, project) group tagging the dispatching
session's label, and only then write the position rows that stop those
dispatches being announced again. Post-then-write makes delivery
at-least-once: a crash between the two repeats an announcement, never loses
one.

THE EASIER HALF OF THE PROBLEM ``tools/hpc-wake`` SOLVES, deliberately built
to look like it. That bridge has to ask a cluster over ssh what happened;
this one reads a file ``fleet-collect`` wrote on the same machine minutes
earlier, so there is no remote call in a cycle at all. What the two share --
service identity, the board's argument shape, the standing-task rule -- is
:mod:`platform_core.board`, lifted rather than copied.

The polling loop lives in the scheduler that calls the CLI, where its interval
is visible, for the same reason ``fleet-watch`` refuses to follow.
"""
