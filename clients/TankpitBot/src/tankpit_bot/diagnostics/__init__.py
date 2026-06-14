"""Post-run diagnostic and analysis tools for bot and action_lab event streams.

This package consumes the JSONL event artifacts that
:mod:`tankpit_bot.runtime_logging` writes during ``make bot`` and
``make <name>-probe`` runs and turns them into structured reports that
identify what happened and where the bot's view diverged from the
game's actual state.

The first tool is :mod:`tankpit_bot.diagnostics.issue_report` -- a
strict-typed analyzer that loads every event from a JSONL artifact and
produces an :class:`IssueReportDict` listing every teleport (success
and failure), every ``map_open`` dispatch and its outcome, every fuel
target selection (and the reason it was rejected when applicable), and
the session room/field metadata so the analysis terrain is never
ambiguous.
"""
