"""What a Linux build's unit records when it ends without the build's own status.

MCPs board task c8585623. A Linux build writes its exit status to the result
file as its LAST act (:meth:`fleet.core.dialect_linux.LinuxDialect.build_script`),
and the collector read an absent result as a run still going. A build whose
processes are killed writes nothing, so the runner renewed its claim until the
lease ran out and closed it ``LEASE_NOT_HELD`` with exit 124, which reads as a
suite that hung.

MEASURED, fleet job 859e7ab3 (MCPs mcp-shared at 110005706, diphtheria).
Its transcript ended in npm ci's deprecation warnings at 2026-10-05T03:42Z,
and the runner stopped it at 04:06:51Z, 37 s past its lease, as a 24-minute
silent install. diphtheria's journal says the build was dead at 03:44:39Z,
2 min 8 s after the unit started::

    systemd-oomd: Killed /user.slice/user-1000.slice/user@1000.service/app.slice/
      fleet-MCPs-mcp-shared-diphtheria-1791171734.service due to memory pressure for
      /user.slice/user-1000.slice/user@1000.service being 67.99% > 50.00% for > 20s
      with reclaim activity
    fleet-MCPs-mcp-shared-diphtheria-1791171734.service: Main process exited,
      code=killed, status=9/KILL
    fleet-MCPs-mcp-shared-diphtheria-1791171734.service: Failed with result 'oom-kill'.

The 50 percent limit is Ubuntu's own, ``ManagedOOMMemoryPressure=kill`` in
``/usr/lib/systemd/system/user@.service.d/10-oomd-user-service-defaults.conf``,
and every fleet unit runs under ``user@1000.service``. The node's load record
(``logs/load/2026-10-05.jsonl``) shows host memory pressure of 37.4 s in the
minute to 03:44:44Z, against 0.1 to 0.5 s in the minutes before it, with swap
at 6.4 of 7 GB. Killing a unit under that pressure protects the production
stack on the same host. The defect was that the runner then took 22 minutes
to notice, and named the result as a lease expiry.

THE UNIT NOW RECORDS ITS OWN ENDING. The launch gives the transient unit an
``ExecStopPost=`` running :func:`unit_end_script`. systemd runs it after the
main process ends for any reason (exit, signal, oom-kill, a stop) and sets
``SERVICE_RESULT``, ``EXIT_CODE`` and ``EXIT_STATUS`` for it (systemd.exec(5),
"Environment Variables Set or Propagated by the Service Manager"). When the
build already wrote its status, the script does nothing. Otherwise it appends
one :data:`UNIT_ENDED_MARKER` line to the transcript, and writes the status a
shell reports for that ending to the result file. The collector's next poll
then reads a finished run, and the verdict line carries the marker's text
(:func:`fleet.core.verdict.read_unit_end`). Probed on diphtheria (systemd 255)
on 2026-10-07: a unit whose cgroup was sent SIGKILL ran it with
``signal killed KILL``, and one that exited 3 ran it with
``exit-code exited 3``.

What still reaches the lease: a node that cannot run this script at all, for
example one whose user manager is itself gone. The lease remains the outer
bound for that case, as it was for every case before.

A BUILD ENDED FOR MEMORY IS NOT A FAILING SUITE (MCPs board task d111690a).
Every check now starts at ``OOMScoreAdjust=1000`` with ``OOMPolicy=stop``
(MCPs ``scripts/host/lib/cpu-shares.sh``), so a host that runs out ends the
check before any stack container, and the kill of any one of its processes
stops the whole unit with the result ``oom-kill``, as systemd-oomd's kill
does. Measured on lavender on 2026-10-25T14:50:49Z: a unit started through
the helper, capped at 256 MiB and filled by a child, ended with
``SERVICE_RESULT=oom-kill EXIT_CODE=killed EXIT_STATUS=TERM``.
:func:`killed_for_memory` reads that ending from the line, which the runner
then gives back to the queue rather than closing as failed
(:func:`fleet.cli.node_settle.settle`).
"""

from __future__ import annotations

import re
import shlex
from typing import Final

from fleet.core import names

#: The stem of the script the unit runs as its ``ExecStopPost=``, beside the
#: build's own in the dispatch directory, so the retire removes both.
UNIT_END_STEM: Final = "unit-end"

#: The word that opens the transcript line, so a reader can grep for it.
UNIT_ENDED_MARKER: Final = "FLEET_UNIT_ENDED"

#: The words between the unit's name and systemd's ``SERVICE_RESULT`` on the
#: line, written by :func:`unit_end_script` and read by
#: :func:`killed_for_memory`.
RESULT_WORDS: Final = "ended with systemd result"

#: The ``SERVICE_RESULT`` of a unit the kernel's OOM killer or systemd-oomd
#: ended (systemd.exec(5), "Environment Variables Set or Propagated by the
#: Service Manager").
OOM_KILL_RESULT: Final = "oom-kill"

#: A Linux unit's line when its unit was ended for memory.
MEMORY_KILLED_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{UNIT_ENDED_MARKER}: unit \S+ {RESULT_WORDS} {OOM_KILL_RESULT} \("
)

#: The line, as :func:`unit_end_script` and a Windows node's result script
#: (:mod:`fleet.core.windows_result`) append it; the verdict reads the whole
#: line, marker included. It ends before a carriage return as well as a line
#: feed: a Windows tail crosses ssh with CRLF endings and is decoded as it
#: came (:mod:`fleet.core._command`), so a ``$`` would leave the return in
#: the verdict line.
UNIT_ENDED_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{UNIT_ENDED_MARKER}: [^\r\n]+", re.MULTILINE
)

#: The status recorded when the unit ended without a failing status of its
#: own: its main process exited 0 without writing a result, which the build's
#: construction never does, or it never ran one. A zero here would close a
#: check that never ran its recipe as passed. 125 is what GNU ``timeout`` and
#: ``env`` exit with when they themselves fail, which is the case here: the
#: fleet's own machinery, not the recipe.
UNEXPLAINED_EXIT_CODE: Final = 125

#: The offset a shell adds to a signal's number for a process it killed.
SIGNAL_EXIT_BASE: Final = 128

#: The highest signal number Linux has (``SIGRTMAX``), where the walk from a
#: signal's name back to its number stops.
SIGNAL_LIMIT: Final = 64


def unit_end_path(target: str) -> str:
    """Where a dispatch's unit-end script is written.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        ``<target>/unit-end.sh``.
    """
    return f"{target}/{UNIT_END_STEM}.sh"


def unit_end_script(*, target: str, unit: str, prologue: str) -> str:
    """The script a unit runs once its main process has ended.

    Args:
        target: Absolute remote directory holding the staged tree, where the
            result file and the transcript are.
        unit: The transient unit's name, written into the line.
        prologue: The dialect's fail-fast prologue.

    Returns:
        The script's text. It reads ``EXIT_CODE`` and ``EXIT_STATUS`` as
        empty when unset, which systemd does for a unit whose main process
        never ran; that case records :data:`UNEXPLAINED_EXIT_CODE`. A
        signal's name is turned into its number by walking ``kill -l <n>``
        from 1 to :data:`SIGNAL_LIMIT`. POSIX specifies only that direction,
        and dash, diphtheria's ``/bin/sh``, refuses the other: measured
        2026-10-07, ``kill -l KILL`` answered ``Illegal number: KILL`` while
        ``kill -l 9`` answered ``KILL``. A name no number has records
        :data:`UNEXPLAINED_EXIT_CODE`.
    """
    result = shlex.quote(f"{target}/{names.RESULT_NAME}")
    log = shlex.quote(names.log_path(target))
    return (
        f"{prologue}if [ -f {result} ]; then\n"
        "  exit 0\n"
        "fi\n"
        'service_result="${SERVICE_RESULT:-}"\n'
        'exit_code="${EXIT_CODE:-}"\n'
        'exit_status="${EXIT_STATUS:-}"\n'
        f"code={UNEXPLAINED_EXIT_CODE}\n"
        'case "$exit_code" in\n'
        "  exited)\n"
        '    code="$exit_status"\n'
        "    ;;\n"
        "  killed|dumped)\n"
        "    signal=1\n"
        f'    while [ "$signal" -le {SIGNAL_LIMIT} ]; do\n'
        '      if [ "$(kill -l "$signal")" = "$exit_status" ]; then\n'
        f'        code="$(({SIGNAL_EXIT_BASE} + signal))"\n'
        "        break\n"
        "      fi\n"
        '      signal="$((signal + 1))"\n'
        "    done\n"
        "    ;;\n"
        "esac\n"
        'if [ "$code" -eq 0 ]; then\n'
        f"  code={UNEXPLAINED_EXIT_CODE}\n"
        "fi\n"
        f"printf '%s: unit %s {RESULT_WORDS} %s (%s %s) before the build wrote "
        f"its status; recorded exit %s\\n' {UNIT_ENDED_MARKER} {shlex.quote(unit)} "
        '"$service_result" "$exit_code" "$exit_status" "$code" '
        f">> {log}\n"
        f"printf '%s\\n' \"$code\" > {result}\n"
    )


def killed_for_memory(unit_end: str) -> bool:
    """Whether a unit-end line says the kernel or systemd-oomd ended the unit for memory.

    Args:
        unit_end: The line, as :func:`fleet.core.verdict.read_unit_end`
            returns it.

    Returns:
        True for a Linux unit whose ``SERVICE_RESULT`` was
        :data:`OOM_KILL_RESULT`; False for any other ending, and for every
        Windows task's line, which names no systemd result.
    """
    return MEMORY_KILLED_LINE.match(unit_end) is not None


__all__ = [
    "MEMORY_KILLED_LINE",
    "OOM_KILL_RESULT",
    "RESULT_WORDS",
    "SIGNAL_EXIT_BASE",
    "SIGNAL_LIMIT",
    "UNEXPLAINED_EXIT_CODE",
    "UNIT_ENDED_LINE",
    "UNIT_ENDED_MARKER",
    "UNIT_END_STEM",
    "killed_for_memory",
    "unit_end_path",
    "unit_end_script",
]
