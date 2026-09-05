"""Subscribe a shell to the corvis agent board's change feed.

WHY THIS EXISTS. The board's notification surfaces split by who can reach
whom, and exactly one of them reaches a session that is sitting idle:

* A board post is a PULL. Another session sees it when it next chooses to
  call ``task_feed``.
* A cross-session ``SendMessage`` is a PUSH that lands in a queue. Measured
  2026-09-05: the send returns ``success`` and the target's status stays
  ``idle``, because the documented drain condition is the receiver's next
  tool round and an idle session has none until its operator types.
* A ``Monitor`` command is the only surface whose events arrive while a
  session is waiting for its user.

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
