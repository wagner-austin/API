"""Measure what the readable OptionMetrics volatility surfaces cover, as aggregates only.

One WRDS Grid Engine job runs one read-only transaction under ``psql`` with
``ON_ERROR_STOP``. For each relation in :data:`VOLATILITY_SURFACES` it writes
one CSV line: the row count, distinct securities and first and last date,
overall and within :data:`SLICE`, the surface point a one-year at-the-money
put calibration reads (365 days, delta -50, put). No dataset row leaves WRDS,
only those counts and the two bounding dates.

Which relations to measure is fixed here rather than chosen per run: these
are the volatility-surface relations the 2026-10-28 SQL inventory found
readable, and a study that needs a surface from another library adds it here,
where the test pins the query.

Authentication is WRDS Cloud's own, exactly as for the SQL inventory: the job
exports the login environment, which names an ident-authenticated database
host, so no password is stored, sent or logged.
"""

from __future__ import annotations

import csv
import io
import pathlib

from platform_core.json_utils import dump_json_str

from hpc3.contracts.wrds_coverage import (
    CoverageReceipt,
    CoverageRow,
    decode_coverage_row,
    encode_coverage_receipt,
)
from hpc3.contracts.wrds_sql import psql_argv
from hpc3.core import _test_hooks, artifacts, wrds_grid
from hpc3.core.wrds_sql import quote_identifier

#: The readable volatility-surface relations, as ``(library, relation)``.
VOLATILITY_SURFACES: tuple[tuple[str, str], ...] = (
    ("optionmsamp_us", "vsurfd2014"),
    ("omtrial", "vsurfd2014"),
    ("omtrial", "volatility_surface_2013"),
)

#: The surface point a one-year at-the-money put calibration reads.
SLICE = "days = 365 AND delta = -50 AND cp_flag = 'P'"


def _literal(text: str) -> str:
    """Quote a SQL string literal.

    Args:
        text: Any text.

    Returns:
        The text in single quotes with embedded quotes doubled.
    """
    return "'" + text.replace("'", "''") + "'"


def coverage_query(surfaces: tuple[tuple[str, str], ...]) -> str:
    """Write the aggregates-only coverage query.

    Args:
        surfaces: ``(library, relation)`` pairs, at least one.

    Returns:
        The psql script: one read-only transaction copying one CSV line per
        relation, in the order given.

    Raises:
        ValueError: ``WRDS_COVERAGE_EMPTY`` for no relations.
    """
    if not surfaces:
        raise ValueError("WRDS_COVERAGE_EMPTY: name at least one relation")
    selects = [
        f"SELECT {_literal(library)}, {_literal(relation)}, count(*), count(DISTINCT secid), "
        "min(date)::text, max(date)::text, "
        f"count(*) FILTER (WHERE {SLICE}), count(DISTINCT secid) FILTER (WHERE {SLICE}) "
        f"FROM {quote_identifier(library)}.{quote_identifier(relation)}"
        for library, relation in surfaces
    ]
    return (
        "BEGIN READ ONLY;\n"
        "SET LOCAL statement_timeout = '1800s';\n"
        "COPY (\n" + "\nUNION ALL\n".join(selects) + "\n) TO STDOUT WITH (FORMAT CSV);\n"
        "COMMIT;\n"
    )


def read_coverage(text: str, surfaces: tuple[tuple[str, str], ...]) -> list[CoverageRow]:
    """Decode the job's CSV and require exactly the relations asked for.

    Args:
        text: The whole artifact.
        surfaces: The relations the query named, in its order.

    Returns:
        One row per relation, in that order.

    Raises:
        ValueError: ``WRDS_COVERAGE_RELATIONS`` when the lines do not name
            exactly those relations in order, and the row decoder's refusals.
    """
    rows = [decode_coverage_row(fields) for fields in csv.reader(io.StringIO(text, newline=""))]
    named = [(row["library"], row["relation"]) for row in rows]
    if named != list(surfaces):
        raise ValueError(f"WRDS_COVERAGE_RELATIONS: expected {list(surfaces)}, got {named}")
    return rows


def option_coverage(host: str, remote_dir: str, output: pathlib.Path) -> CoverageReceipt:
    """Measure the surfaces and save the verified aggregates locally.

    Writes ``output`` (the CSV exactly as transported) and the receipt beside
    it as ``<output>.receipt.json``.

    Args:
        host: WRDS SSH alias.
        remote_dir: New absolute directory in the user's WRDS home.
        output: New local CSV path. Its parent must already exist.

    Returns:
        The saved receipt.

    Raises:
        ValueError: For an unsafe remote path, an unreadable scheduler id or
            a malformed or mismatched coverage file.
        FileExistsError: If either local output exists.
        AppError: If transport or the job fails, including a relation or
            column the account cannot read.
    """
    remote = wrds_grid.remote_directory(remote_dir)
    receipt_file = wrds_grid.receipt_path(output)
    wrds_grid.refuse_existing([output, receipt_file])
    wrds_grid.create_directory(host, remote)
    query = coverage_query(VOLATILITY_SURFACES)
    script = str(remote / "coverage.sql")
    remote_output = str(remote / "coverage.csv")
    query_sha = wrds_grid.upload(host, script, query.encode("utf-8"))
    job = wrds_grid.submit(
        host,
        remote,
        "wrds_option_coverage",
        psql_argv(script, remote_output),
        environment=True,
        execution_host=None,
    )
    artifact = artifacts.download(host, remote_output)
    receipt = CoverageReceipt(
        host=host,
        scheduler="grid-engine",
        job_id=job,
        query_sha256=query_sha,
        coverage_sha256=artifact["sha256"],
        remote_output=remote_output,
        slice=SLICE,
        rows=read_coverage(artifact["text"], VOLATILITY_SURFACES),
        scope="aggregate counts and bounding dates in one read-only transaction; no dataset rows",
    )
    _test_hooks.write_text(output, artifact["text"])
    _test_hooks.write_text(receipt_file, dump_json_str(encode_coverage_receipt(receipt)) + "\n")
    return receipt


__all__ = [
    "SLICE",
    "VOLATILITY_SURFACES",
    "coverage_query",
    "option_coverage",
    "read_coverage",
]
