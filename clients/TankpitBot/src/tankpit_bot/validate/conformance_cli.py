"""The ``tankpit-conformance`` entrypoint, run by ``make audit``.

Replays every archived capture under the given roots through the sim
(:mod:`tankpit_bot.validate.conformance`), prints the match rate per
command group, the most frequent divergences with an example of each and
the captures it could not replay, and writes the whole report as JSON.

``make audit`` re-derives the wiki's physics claims from the archive; this
is the same archive asked the other question, whether the sim's server
answers each recorded tick the way the real one did. With ``--baseline``
it is a ratchet: a command group whose match rate falls below the
baseline's fails the run, so a sim change that breaks a law the archive
proves is caught by the archive, not by a live run. ``--write-baseline``
records this run's group rates as the next baseline, which is how an
intended improvement raises the bar; the file holds the rates alone, not
the per-capture detail, because the rates are all the ratchet reads.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.validate.conformance import discover_captures, run_conformance
from tankpit_bot.validate.conformance_types import (
    ConformanceReportDict,
    GroupTallyDict,
    decode_report,
    encode_report,
)

DEFAULT_ROOTS = (Path("runs") / "bot", Path("runs") / "sniff")
"""The archive the spec names: every bot and sniff capture, recursively."""

DEFAULT_OUT = Path("runs") / "analysis" / "conformance.json"
"""Where the report is written unless ``--out`` says otherwise."""

DEFAULT_TOP = 25
"""How many divergences the printed report lists."""

EXIT_USAGE = 2
"""A command line this entrypoint does not accept."""


class ConformanceUsageError(ValueError):
    """An argument this entrypoint does not accept (``CONFORMANCE_USAGE``)."""


class _Args:
    """The parsed command line."""

    def __init__(self) -> None:
        self.roots: list[Path] = []
        self.out = DEFAULT_OUT
        self.baseline: Path | None = None
        self.write_baseline: Path | None = None
        self.top = DEFAULT_TOP


def _parse(args: list[str]) -> _Args:
    """Parse the command line, refusing anything unknown.

    Args:
        args: The arguments after the program name.

    Returns:
        The parsed arguments; ``roots`` defaults to :data:`DEFAULT_ROOTS`.

    Raises:
        ConformanceUsageError: For an unknown flag or a flag missing its value.
    """
    parsed = _Args()
    index = 0
    while index < len(args):
        flag = args[index]
        if flag not in ("--root", "--out", "--baseline", "--write-baseline", "--top"):
            raise ConformanceUsageError(f"CONFORMANCE_USAGE: unknown argument {flag!r}")
        if index + 1 >= len(args):
            raise ConformanceUsageError(f"CONFORMANCE_USAGE: {flag} needs a value")
        value = args[index + 1]
        if flag == "--root":
            parsed.roots.append(Path(value))
        elif flag == "--out":
            parsed.out = Path(value)
        elif flag == "--baseline":
            parsed.baseline = Path(value)
        elif flag == "--write-baseline":
            parsed.write_baseline = Path(value)
        elif not value.isdigit():
            raise ConformanceUsageError(f"CONFORMANCE_USAGE: --top needs a count, not {value!r}")
        else:
            parsed.top = int(value)
        index += 2
    if not parsed.roots:
        parsed.roots = list(DEFAULT_ROOTS)
    return parsed


def _rate(group: GroupTallyDict) -> float:
    """A group's match rate.

    Args:
        group: The group (``ticks`` is at least one in any report).

    Returns:
        Matched ticks over compared ticks.
    """
    return group["matched"] / group["ticks"]


def regressions(report: ConformanceReportDict, baseline: ConformanceReportDict) -> list[str]:
    """Every command group whose match rate fell below the baseline's.

    A group the current run did not compare at all is not a regression of
    a law: the archive the run was given simply held no such tick.

    Args:
        report: This run.
        baseline: The run it must not fall below.

    Returns:
        One line per regressed group; empty when none regressed.
    """
    current = {g["commands"]: g for g in report["groups"]}
    lines: list[str] = []
    for before in baseline["groups"]:
        now = current.get(before["commands"])
        if now is not None and _rate(now) < _rate(before):
            lines.append(
                f"{before['commands']}: {now['matched']}/{now['ticks']} ({_rate(now):.1%})"
                f" below baseline {before['matched']}/{before['ticks']} ({_rate(before):.1%})"
            )
    return lines


def format_report(report: ConformanceReportDict, top: int) -> str:
    """Render a run for a terminal.

    Args:
        report: The run.
        top: How many divergences to list.

    Returns:
        The rendered report.
    """
    ticks = sum(s["ticks"] for s in report["sessions"])
    matched = sum(s["matched"] for s in report["sessions"])
    unmodelled = sum(s["unmodelled"] for s in report["sessions"])
    overall = matched / ticks if ticks else 0.0
    lines = [
        f"conformance: {len(report['sessions'])} captures replayed, "
        f"{len(report['skipped'])} skipped; {matched}/{ticks} commanded ticks matched "
        f"({overall:.1%}), {unmodelled} unmodelled",
        "",
        f"{'commands':<40} {'matched':>8} {'ticks':>7} {'rate':>7}",
    ]
    for group in report["groups"]:
        lines.append(
            f"{group['commands']:<40} {group['matched']:>8} {group['ticks']:>7}"
            f" {_rate(group):>7.1%}"
        )
    lines.extend(["", f"top {min(top, len(report['divergences']))} divergences:"])
    for d in report["divergences"][:top]:
        lines.append(
            f"{d['count']:>6}  {d['commands']}  LIVE {' '.join(d['live']) or '(silent)'}"
            f"  SIM {' '.join(d['sim']) or '(silent)'}"
            f"  e.g. {d['example_session']} @{d['example_timestamp_ms']}"
        )
    if report["skipped"]:
        lines.extend(["", "skipped:"])
        for s in report["skipped"]:
            lines.append(f"  {s['reason'].value}: {s['session']} ({s['detail']})")
    return "\n".join(lines)


def run(args: _Args) -> int:
    """Replay the archive, print and write the report, apply the ratchet.

    Args:
        args: The parsed command line.

    Returns:
        0 on success; 1 when nothing could be replayed or a group regressed.
    """
    baseline: ConformanceReportDict | None = None
    if args.baseline is not None:
        # Read before anything is written, so a baseline at the --out or
        # --write-baseline path is the previous run, not this one.
        baseline_text = _test_hooks.read_text(args.baseline)
        baseline = decode_report(narrow_json_to_dict(load_json_str(baseline_text)))
    report = run_conformance(discover_captures(args.roots))
    sys.stdout.write(format_report(report, args.top) + "\n")
    _test_hooks.write_text(args.out, dump_json_str(encode_report(report)))
    sys.stdout.write(f"\nwrote {args.out}\n")
    if not report["sessions"]:
        sys.stdout.write("CONFORMANCE_EMPTY: no capture under the roots could be replayed\n")
        return 1
    if args.write_baseline is not None:
        rates = ConformanceReportDict(
            sessions=[], skipped=[], groups=report["groups"], divergences=[]
        )
        _test_hooks.write_text(args.write_baseline, dump_json_str(encode_report(rates)))
        sys.stdout.write(f"wrote baseline {args.write_baseline}\n")
    if baseline is None:
        return 0
    regressed = regressions(report, baseline)
    for line in regressed:
        sys.stdout.write(f"CONFORMANCE_REGRESSED: {line}\n")
    return 1 if regressed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: ``--root DIR`` (repeatable), ``--out PATH``, ``--baseline
            PATH``, ``--write-baseline PATH``, ``--top N``. Uses
            ``sys.argv[1:]`` when None.

    Returns:
        The process exit code; :data:`EXIT_USAGE` for a bad command line.
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    try:
        parsed = _parse(args)
    except ConformanceUsageError as error:
        sys.stderr.write(f"{error}\n")
        return EXIT_USAGE
    return run(parsed)


__all__ = [
    "DEFAULT_OUT",
    "DEFAULT_ROOTS",
    "DEFAULT_TOP",
    "EXIT_USAGE",
    "ConformanceUsageError",
    "format_report",
    "main",
    "regressions",
    "run",
]
