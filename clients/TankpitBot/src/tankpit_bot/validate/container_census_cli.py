"""The ``tankpit-container-census`` entrypoint, run by ``make container-census``.

Reads every radar scan in the capture archive
(:mod:`tankpit_bot.validate.container_census`), prints each field's
census and every held-out transfer score, and writes the report as JSON.
It is a measurement, not a gate: nothing here fails on what the archive
holds, only on an archive that holds no scanned field at all.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.validate.conformance import discover_captures
from tankpit_bot.validate.conformance_cli import DEFAULT_ROOTS, EXIT_USAGE
from tankpit_bot.validate.container_census import run_census
from tankpit_bot.validate.container_census_types import (
    CensusReportDict,
    TerrainCountsDict,
    encode_census,
)

DEFAULT_CENSUS_OUT = Path("runs") / "analysis" / "container_census.json"
"""Where the census is written unless ``--out`` says otherwise."""


class CensusUsageError(ValueError):
    """An argument this entrypoint does not accept (``CENSUS_USAGE``)."""


def _parse(args: list[str]) -> tuple[list[Path], Path]:
    """Parse ``--root DIR`` (repeatable) and ``--out PATH``.

    Args:
        args: The arguments after the program name.

    Returns:
        The roots (``DEFAULT_ROOTS`` when none was given) and the output path.

    Raises:
        CensusUsageError: For an unknown flag or a flag missing its value.
    """
    roots: list[Path] = []
    out = DEFAULT_CENSUS_OUT
    index = 0
    while index < len(args):
        flag = args[index]
        if flag not in ("--root", "--out"):
            raise CensusUsageError(f"CENSUS_USAGE: unknown argument {flag!r}")
        if index + 1 >= len(args):
            raise CensusUsageError(f"CENSUS_USAGE: {flag} needs a value")
        if flag == "--root":
            roots.append(Path(args[index + 1]))
        else:
            out = Path(args[index + 1])
        index += 2
    return (roots or list(DEFAULT_ROOTS)), out


def _rate(sites: TerrainCountsDict, observed: TerrainCountsDict, kind: str) -> str:
    """One terrain class's site rate as a percentage, or a dash when unobserved."""
    seen = observed["ground"] if kind == "ground" else observed["water"]
    held = sites["ground"] if kind == "ground" else sites["water"]
    return f"{held / seen:.2%}" if seen else "-"


def format_census(report: CensusReportDict) -> str:
    """Render a census for a terminal.

    Args:
        report: The census.

    Returns:
        The rendered census.
    """
    lines = [
        f"{'field':<14} {'caps':>5} {'scans':>6} {'observed':>9} {'sites':>6}"
        f" {'ground':>7} {'water':>7} {'rock':>5} {'density':>8} {'disp':>6} {'miss':>5}"
    ]
    for f in report["fields"]:
        observed, sites = f["observed"], f["sites"]
        seen = observed["ground"] + observed["water"] + observed["rock"]
        held = sites["ground"] + sites["water"] + sites["rock"]
        lines.append(
            f"{f['field']:<14} {f['captures']:>5} {f['scans']:>6} {seen:>9} {held:>6}"
            f" {_rate(sites, observed, 'ground'):>7} {_rate(sites, observed, 'water'):>7}"
            f" {sites['rock']:>5} {f['container_reads'] / f['tile_reads']:>8.2%}"
            f" {f['block_dispersion']:>6.3f} {f['footprint_misses']:>5}"
        )
    lines.extend(["", "terrain shape fitted on one field, scored on another (log loss per tile):"])
    for t in report["transfers"]:
        lines.append(
            f"  {t['train_field']} -> {t['test_field']}: water factor {t['water_factor']:.3f},"
            f" shape {t['log_loss_shape']:.5f} vs constant {t['log_loss_constant']:.5f};"
            f" density {t['train_density']:.2%} -> {t['test_density']:.2%}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: ``--root DIR`` (repeatable) and ``--out PATH``. Uses
            ``sys.argv[1:]`` when None.

    Returns:
        0 when at least one field was censused; 1 when none was; the usage
        code for a bad command line.
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    try:
        roots, out = _parse(args)
    except CensusUsageError as error:
        sys.stderr.write(f"{error}\n")
        return EXIT_USAGE
    report = run_census(discover_captures(roots))
    sys.stdout.write(format_census(report) + "\n")
    _test_hooks.write_text(out, dump_json_str(encode_census(report)))
    sys.stdout.write(f"\nwrote {out}\n")
    if not report["fields"]:
        sys.stdout.write("CENSUS_EMPTY: no capture under the roots scanned a known field\n")
        return 1
    return 0


__all__ = ["DEFAULT_CENSUS_OUT", "CensusUsageError", "format_census", "main"]
