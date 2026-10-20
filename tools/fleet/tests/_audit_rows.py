"""Running a runner audit's rows under a real bash, each command they ask a stand-in.

Support module, not a test module. The audit (:mod:`fleet.core.runner_audit`)
is one bash script of rows, each a ``probe`` of one command and a ``check``
that reads its answer through the functions the script defines first
(:data:`~fleet.core.runner_audit.AUDIT_FUNCTIONS`). A row's suite runs its
lines after those very functions, with every command the row asks replaced
by a stand-in (tests/_stand_ins.py), so what is asserted is the verdict the
row reaches, not its text.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

from fleet.core.runner_audit import AUDIT_FUNCTIONS
from tests._stand_ins import run_with_stand_ins


def run_rows(
    tmp_path: pathlib.Path, lines: Sequence[str], commands: Mapping[str, str]
) -> list[str]:
    """Run audit rows with stand-in commands, and return the lines they print.

    Args:
        tmp_path: The case's directory.
        lines: The rows' bash lines, as the audit renders them.
        commands: Each command the rows ask, as
            :func:`tests._stand_ins.run_with_stand_ins` takes them.

    Returns:
        The ``CHECK`` lines printed, in order. The rows run under ``set -u``
        as the audit does, and must exit 0: drift is data.
    """
    ran = run_with_stand_ins(tmp_path, [*AUDIT_FUNCTIONS, *lines], commands, options=("-u",))
    assert ran.returncode == 0, ran.stderr
    return ran.stdout.splitlines()


__all__ = ["run_rows"]
