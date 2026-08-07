"""Unified action-outcome fabric: one recorded event per attempt.

Phase 2 of the self-observing architecture. Replaces the three
parallel diagnostic mechanisms (``emit_wire_complete``,
``teleport_attempt``, ``combat_feedback``) and the invisible fourth
(executor ``emit_ai`` discards) with one ``action_outcome`` diagnostic
kind + per-kind ring records.

Import from the submodules directly; the re-export block that used to
live here had no importers.
"""

__all__: tuple[str, ...] = ()
