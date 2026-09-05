"""Subscribe a shell to the corvis agent board's change feed.

WHY THIS EXISTS. The board's notification surfaces split by who can reach
whom, and exactly one of them reaches a session that is sitting idle:

* A board post is a PULL. Another session sees it when it next chooses to
  call ``task_feed``.
* A cross-session ``SendMessage`` is a PUSH, and per the Claude Code docs it
  DOES start a new turn in an idle receiver. What it cannot promise is
  arrival: with no ``crossSessionInbound`` set, a session that bypasses
  permission prompts holds every inbound message for human approval unless
  the sender also bypasses, and a held message never reaches the model. From
  the sender's side, held looks exactly like ignored.
* A ``Monitor`` command arrives while a session is waiting for its user and
  is not subject to those inbound controls, because it is the session's own
  process rather than a peer's message.

So the board plus a Monitor is the pairing that cannot be silently dropped,
which is what this package serves.

Monitor runs SHELL commands and ``task_events`` is an MCP tool, so the two
could not be connected. This package is that connection: it speaks the
board's JSON-RPC over its loopback HTTP surface and prints one line per
event, which is the shape Monitor turns into a notification.

WHAT IT DELIBERATELY DOES NOT DO. It does not retry, back off, or swallow a
failed poll. A watcher that hides an outage reports silence, and silence is
indistinguishable from "nothing has happened" -- which is the failure this
whole area keeps producing. Every failure raises with a specific
:class:`~platform_core.error_codes.BoardWatchErrorCode`, the process exits
non-zero, and Monitor surfaces that as the event it is.
"""
