"""The script that retires one settled run on its node, and the answer it prints.

Split from :mod:`fleet.core.dialect` (MCPs board task 1526ece1) when the
retire grew its two questions to the node: whether a process still uses the
run's directory, and how many bytes removing it frees.

A PROCESS THE NODE RUNS IS ASKED, NOT INFERRED. A run is sent its retire only
once no lease or running ledger row names it (:func:`fleet.core.retire_owed.live_runs`),
but neither names every process in its tree: a build's child that outlives
the build's lease, or a shell a person left in the directory. So before it
moves or removes anything the script reads every process's working directory
and open files under ``/proc`` and keeps the run whole when one is inside its
enclosure, answering which. ``find -lname`` reads a link's target without
following it and forks nothing per process; ``/proc`` shows the links of the
fleet user's own processes, which are every build's in the stage root, while
another user's are unreadable there and are not reported. The script first
reads its own working directory the same way and stops with a named error
when that fails, so a node whose ``find`` cannot read the links fails the
retire rather than answering that nothing uses the run.

THE BYTES ARE COUNTED BEFORE THE REMOVAL, by ``du -sb`` of the enclosure once
its directories are listable again, and printed as the answer, so the hub
records what each retire freed (:mod:`fleet.contracts.freed_record`).
"""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Final

from typing_extensions import TypedDict

from fleet.core import names
from fleet.core.check_results import retained_results_path, sh_retire_lines
from fleet.core.dialect import PROLOGUE
from fleet.core.linux_isolated_build import EXEC_HOME_LINE, exec_root

#: The answer's word when a process uses the run's directory: then
#: ``in-use <link> <target>``, the ``/proc`` link and the path it names.
IN_USE_WORD: Final = "in-use"

#: The answer's word when the run was retired: then ``freed <bytes>``.
FREED_WORD: Final = "freed"

#: The error a node whose ``find`` cannot read a ``/proc`` link stops with.
SCAN_BLIND: Final = "FLEET_RETIRE_SCAN_BLIND"

_IN_USE: Final = re.compile(rf"{IN_USE_WORD} (/proc/[0-9]+/\S+ /.*)")
_FREED: Final = re.compile(rf"{FREED_WORD} ([0-9]+)")


class RetireAnswer(TypedDict):
    """What a node said its retire of one run did.

    Attributes:
        holder: ``<link> <target>`` of the process found using the run's
            directory, which was then left whole; None when it was retired.
        freed_bytes: The bytes its enclosure held when removed; 0 when a
            process held it, since nothing was removed.
    """

    holder: str | None
    freed_bytes: int


def in_use_lines(enclosure: str) -> list[str]:
    """The lines that keep a run whole while a process uses its enclosure.

    Args:
        enclosure: The enclosure, already quoted for ``sh``.

    Returns:
        A self-check that ``find`` reads this shell's own working directory
        from ``/proc``, then the scan, which on a hit prints
        ``in-use <link> <target>`` and exits 0 before anything is touched.
    """
    own = "here=$(pwd -P)"
    seen = 'seen=$(find "/proc/$$/cwd" -maxdepth 0 -printf "%l")'
    blind = (
        f'if [ "$seen" != "$here" ]; then echo "{SCAN_BLIND}: find read /proc/$$/cwd '
        "as '$seen', not '$here'\" >&2; exit 1; fi"
    )
    scan = (
        f"holder=$( {{ find /proc/[0-9]*/cwd /proc/[0-9]*/fd -maxdepth 1 "
        f"\\( -lname {enclosure} -o -lname {enclosure}'/*' \\) -printf '%p %l\\n' "
        "2>/dev/null || true; } | head -n 1 )"
    )
    hit = f'if [ -n "$holder" ]; then echo "{IN_USE_WORD} $holder"; exit 0; fi'
    return [own, seen, blind, scan, hit]


def retire_script(
    *, stage_root: str, run_id: str, retained: str, scripts: tuple[str, ...], isolated: bool
) -> str:
    """Keep a settled run's transcript, then remove its directories and scripts.

    Run once the runner has read the transcript's tail, when a run has
    finished, been stopped past its lease or been cancelled (MCPs board task
    bfca20e6): nothing reads the export again, and a stage root that kept
    every one filled diphtheria's disk. The run's transient unit needs no
    removal: the launch starts it with ``--collect``, so the manager unloads
    it when the build exits, failed or not.

    Args:
        stage_root: The node's declared stage root.
        run_id: The dispatch; its enclosure (:func:`fleet.core.names.run_enclosure`)
            goes whole, named by the run so no path can make it the stage root.
        retained: Where its transcript is kept, under the stage root's logs
            directory, which is created if absent; its check results go
            beside it (:func:`fleet.core.check_results.sh_retire_lines`).
        scripts: The scripts it left under the stage root, this one included.
        isolated: Whether the node runs docker projects' builds as
            execdocker; the run's copy in that user's home then goes too, as
            root, which such a build removes itself only when it runs to its
            end (:mod:`fleet.core.linux_isolated_build`).

    Returns:
        The script's text, which first keeps the run whole when a process
        uses it (:func:`in_use_lines`) and otherwise ends printing
        ``freed <bytes>``. ``mv`` is guarded because a run cancelled before
        its build started wrote no transcript; ``rm -rf`` and ``rm -f`` need
        no guard for what is already gone, and ``-f`` removes the read-only
        objects of the repository staged there. ``rm -rf`` empties a
        directory only with all three of its owner's bits: read to list its
        entries, search to reach them and write to unlink them. So each of the
        run's own directories lacking any of the three gets all three back
        first, each as ``find`` reaches it and before it descends, since
        ``find`` cannot list a directory it may not read either (``-exec ...
        ;``, never a batched ``+``). The write bit alone came first (MCPs
        board task 05dd6588): since MCPs f27b4822e seals release payloads
        read-only, every MCPs packages/maketools run leaves fixture releases
        sealed under its ``.pytest-tmp``, and from 2026-10-23T23:26Z to
        2026-10-24T03:51Z the retires of 15 such runs on six nodes stopped at
        ``rm -rf``, stranding them. The read bit followed (MCPs board task
        1526ece1): on 2026-11-20T14:14Z loki's retire of
        MCPs-packages-maketools-loki-1795212710 stopped at that ``find``, on
        two ``.pytest-tmp`` directories of mode 0355, and left 1.2 GB no
        later pass removed. Neither ``find`` nor ``rm`` follows a link, so a
        linked companion is untouched. Every step tolerates what an earlier
        attempt already did, so a retire that failed part way succeeds when
        rerun, then reporting 0 bytes for an enclosure already gone.
    """
    target = names.dispatch_directory(stage_root, run_id)
    log = shlex.quote(names.log_path(target))
    enclosure = shlex.quote(names.run_enclosure(stage_root, run_id))
    results = retained_results_path(stage_root, run_id)
    lines = [
        *in_use_lines(enclosure),
        f"mkdir -p {shlex.quote(posixpath.dirname(retained))}",
        f"if [ -e {log} ]; then mv -f {log} {shlex.quote(retained)}; fi",
        *sh_retire_lines(target=target, retained=results),
        "freed=0",
        f"if [ -d {enclosure} ]; then "
        f"find {enclosure} -type d ! -perm -u=rwx -exec chmod u+rwx {{}} \\; ; "
        f"freed=$(du -sb {enclosure}); fi",
        f"rm -rf {enclosure}",
        "rm -f " + " ".join(shlex.quote(script) for script in scripts),
    ]
    if isolated:
        copy = exec_root(run_id)
        lines += [EXEC_HOME_LINE, f'sudo -n rm -rf "{copy}" "{names.staging_directory(copy)}"']
    lines.append(f'echo "{FREED_WORD} ${{freed%%[[:space:]]*}}"')
    return PROLOGUE + "".join(f"{line}\n" for line in lines)


def _last_line(output: str) -> str:
    """The last line a script printed, or ``""`` when it printed none."""
    lines = output.strip().splitlines()
    return lines[-1] if lines else ""


def parse_retire_answer(output: str) -> RetireAnswer | None:
    """Decode what a node's retire printed.

    Args:
        output: The script's standard output.

    Returns:
        The process that kept the run whole, or the bytes its removal freed,
        from the output's last line; None when that line is neither answer.
    """
    last = _last_line(output)
    held = _IN_USE.fullmatch(last)
    if held is not None:
        return RetireAnswer(holder=held.group(1), freed_bytes=0)
    freed = _FREED.fullmatch(last)
    if freed is not None:
        return RetireAnswer(holder=None, freed_bytes=int(freed.group(1)))
    return None


def answer_missing(host: str, output: str) -> str:
    """Say that a node's retire exited 0 without either answer.

    Args:
        host: The node.
        output: The script's standard output.

    Returns:
        The message, quoting the line it ended on.
    """
    return (
        f"{host}'s retire ended without its answer: its last line was {_last_line(output)!r}, "
        f"not '{IN_USE_WORD} <link> <target>' or '{FREED_WORD} <bytes>'"
    )


__all__ = [
    "FREED_WORD",
    "IN_USE_WORD",
    "SCAN_BLIND",
    "RetireAnswer",
    "answer_missing",
    "in_use_lines",
    "parse_retire_answer",
    "retire_script",
]
