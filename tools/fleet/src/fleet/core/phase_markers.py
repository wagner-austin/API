"""The phase lines every build writes into its transcript.

WHY (MCPs board task 74b13c20). A row's transcript said what each step
printed and never when: 2,446 fleet check jobs in the week to 2026-10-04 ran
330 h, the shortest MCPs rows took 340-360 s whatever the package, and the
only way to learn that fleet-mcp's tests were 39.6 s of a 361 s row was to
read vitest's own summary. Every build now writes one line as each phase
starts and one as it ends::

    fleet-phase install started 2026-10-04T12:00:00Z
    fleet-phase install ended 2026-10-04T12:00:31Z after 31 s, exit 0

for each install step under the phase the registry names
(:class:`fleet.contracts.source.InstallStep`), then for the recipe itself
as the phase ``check``. The stamps are UTC and the duration whole seconds of
wall clock, read by the node's own shell, so the three renders
(:mod:`fleet.core.dialect_linux`, :mod:`fleet.core.linux_isolated_build`,
:mod:`fleet.core.windows_build`) write the same text and one grep reads any
transcript.

WHY ``check`` IS ONE PHASE AND NOT LINT AND TEST. Splitting it would mean
running ``make lint`` and ``make test`` as separate makes, and an MCPs
package's check lock charges its five-minute budget from the moment make
read the Makefile of ``make check`` (MCPs ``packages/maketools``
``check_budget.py``, board task 4080d695): a split would restart that clock
at ``make test`` and pass a check whose lint and suite together are over
budget, a different verdict from the same commit's ``make check`` anywhere
else. The check lock times the two inside ``check`` instead, in the same
transcript and in this grammar: ``fleet-phase lint`` from make reading the
Makefile to the lock being asked for, and ``fleet-phase test`` from there
to the suite returning, with the suite's own exit (MCPs
``packages/maketools`` ``check_phases.py``, the review of 2026-10-04 16:58Z),
beside its ``CHECK BUDGET: <package> took Ns of 300s (...)`` line.
"""

from __future__ import annotations

from typing import Final

#: The word every phase line starts with.
PHASE_MARKER: Final[str] = "fleet-phase"

#: ``date``'s spelling of a UTC stamp in the phase lines' shape.
SH_UTC_STAMP: Final[str] = '"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'


def sh_phase_lines(*, phase: str, command: str, log: str) -> list[str]:
    """The ``sh`` lines that run one command as a timed phase.

    Not under ``set -e`` for the command itself, as the builds were before
    the phases: a failing step is a result to record, so its status is left
    in ``$status`` for the caller to act on.

    Args:
        phase: The phase's name, in the source grammar or ``check``, which
            carries nothing ``sh`` reads.
        command: The command line, already rendered for ``sh``.
        log: The transcript's path, appended to.

    Returns:
        The lines, ending with the phase's closing line; ``$status`` holds
        the command's exit status after them.
    """
    return [
        'phase_started="$(date +%s)"',
        f"printf '{PHASE_MARKER} %s started %s\\n' '{phase}' {SH_UTC_STAMP} >> '{log}'",
        "set +e",
        f"{command} >> '{log}' 2>&1",
        "status=$?",
        "set -e",
        f"printf '{PHASE_MARKER} %s ended %s after %s s, exit %s\\n' '{phase}' {SH_UTC_STAMP} "
        f'"$(( $(date +%s) - phase_started ))" "$status" >> \'{log}\'',
    ]


__all__ = ["PHASE_MARKER", "SH_UTC_STAMP", "sh_phase_lines"]
