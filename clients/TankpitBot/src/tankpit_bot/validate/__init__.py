"""Physics claim validators — Phase 2 of the physics-module roadmap.

Re-derives the wiki's physics claims from the recorded evidence on
every ``make audit`` run: the capture archive
(``runs/**/*.capture_session.json``, decoded with the production
protocol decoders) and the bot diagnostic logs
(``runs/bot/*.events.jsonl``). Each validator promotes the one-off
analysis method recorded in ``wiki/log.md`` (2026-07-20 entries) to a
re-runnable check. See ``wiki/pages/physics-module-roadmap.md``.
"""

from __future__ import annotations

from tankpit_bot.validate.audit import run_audit
from tankpit_bot.validate.types import ClaimEvidenceDict

__all__ = [
    "ClaimEvidenceDict",
    "run_audit",
]
