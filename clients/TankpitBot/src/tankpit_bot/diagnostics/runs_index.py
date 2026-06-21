"""Per-run index for bot sessions.

After every bot session ends, :func:`append_index_row` appends one
TSV row to ``runs/bot/_index.tsv`` summarising the run -- enough
context to find a specific run without grepping per-session JSONL.

The index is one line per session, tab-separated. Columns (in order):

  1. ``stamp`` -- the canonical run stamp (``YYYYMMDD-HHMMSS``).
  2. ``duration_s`` -- session wall-clock seconds (integer).
  3. ``exit_reason`` -- ``completed`` / ``interrupted`` / ``stop_file``.
  4. ``ticks`` -- total ticks executed.
  5. ``stalls`` -- WIRE_COMPLETE events with ``signal=stall_timeout``.
  6. ``shots_fired`` -- hits + misses (from the AI scorecard).
  7. ``kills`` -- ``session_kill_count`` from the AI scorecard.
  8. ``kills_per_min`` -- ``kills / max(1, duration_s) * 60`` rounded to
     two decimal places.

The TSV format is human-friendly (one ``awk``/``column`` pipeline reads
it) and append-safe (no rewrites). A header row is written when the
file is first created and never duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)

from tankpit_bot import _test_hooks

#: Default index path for bot sessions.
DEFAULT_INDEX_PATH: Path = Path("runs/bot/_index.tsv")

#: Column order matched by :func:`encode_row` and :func:`decode_row`.
INDEX_COLUMNS: tuple[str, ...] = (
    "stamp",
    "duration_s",
    "exit_reason",
    "ticks",
    "stalls",
    "shots_fired",
    "kills",
    "kills_per_min",
)

#: Header line (with trailing newline) prepended to a newly-created index.
HEADER_LINE: str = "\t".join(INDEX_COLUMNS) + "\n"


class BotRunIndexRowDict(TypedDict):
    """One row of the bot run index.

    Attributes:
        stamp: Canonical run stamp matching ``runs/bot/bot-<stamp>.*``.
        duration_s: Session wall-clock seconds (integer; partial seconds
            rounded down).
        exit_reason: How the run ended -- ``completed`` (tick budget
            exhausted), ``interrupted`` (SIGINT/SIGTERM), or
            ``stop_file`` (graceful sentinel file consumed).
        ticks: Total ticks executed before exit.
        stalls: Count of ``stall_timeout`` WIRE_COMPLETE events.
        shots_fired: Hits + misses from the AI scorecard.
        kills: ``session_kill_count`` from the AI scorecard.
        kills_per_min: ``kills / max(1, duration_s) * 60`` rounded to
            two decimal places (float).
    """

    stamp: str
    duration_s: int
    exit_reason: str
    ticks: int
    stalls: int
    shots_fired: int
    kills: int
    kills_per_min: float


def make_index_row(
    *,
    stamp: str,
    duration_s: int,
    exit_reason: str,
    ticks: int,
    stalls: int,
    shots_fired: int,
    kills: int,
) -> BotRunIndexRowDict:
    """Build a typed index row, computing ``kills_per_min`` automatically.

    Args:
        stamp: Canonical run stamp.
        duration_s: Session wall-clock seconds.
        exit_reason: Lifecycle outcome (see :class:`BotRunIndexRowDict`).
        ticks: Total ticks executed.
        stalls: stall_timeout count.
        shots_fired: hits + misses.
        kills: session_kill_count.

    Returns:
        Populated row with ``kills_per_min`` derived from the inputs.
    """
    rate = round(kills / max(1, duration_s) * 60, 2)
    return BotRunIndexRowDict(
        stamp=stamp,
        duration_s=duration_s,
        exit_reason=exit_reason,
        ticks=ticks,
        stalls=stalls,
        shots_fired=shots_fired,
        kills=kills,
        kills_per_min=rate,
    )


def encode_row(row: BotRunIndexRowDict) -> str:
    """Encode an index row into one TSV line (with trailing newline).

    Args:
        row: Index row to encode.

    Returns:
        TSV-formatted line containing the column values in
        :data:`INDEX_COLUMNS` order.
    """
    return (
        "\t".join(
            [
                row["stamp"],
                str(row["duration_s"]),
                row["exit_reason"],
                str(row["ticks"]),
                str(row["stalls"]),
                str(row["shots_fired"]),
                str(row["kills"]),
                f"{row['kills_per_min']:.2f}",
            ]
        )
        + "\n"
    )


def decode_row(line: str) -> BotRunIndexRowDict:
    """Decode one TSV line into a typed index row.

    Args:
        line: A TSV row (with or without trailing newline) containing
            the columns in :data:`INDEX_COLUMNS` order.

    Returns:
        Validated index row.

    Raises:
        JSONTypeError: When the line has the wrong number of columns,
            or any column fails to parse as its declared type.
    """
    stripped = line.rstrip("\n")
    parts = stripped.split("\t")
    if len(parts) != len(INDEX_COLUMNS):
        raise JSONTypeError(f"index row has {len(parts)} columns, expected {len(INDEX_COLUMNS)}")
    raw: JSONObject = {
        "stamp": parts[0],
        "duration_s": _parse_int_column(parts[1], "duration_s"),
        "exit_reason": parts[2],
        "ticks": _parse_int_column(parts[3], "ticks"),
        "stalls": _parse_int_column(parts[4], "stalls"),
        "shots_fired": _parse_int_column(parts[5], "shots_fired"),
        "kills": _parse_int_column(parts[6], "kills"),
        "kills_per_min": _parse_float_column(parts[7], "kills_per_min"),
    }
    return BotRunIndexRowDict(
        stamp=require_str(raw, "stamp"),
        duration_s=require_int(raw, "duration_s"),
        exit_reason=require_str(raw, "exit_reason"),
        ticks=require_int(raw, "ticks"),
        stalls=require_int(raw, "stalls"),
        shots_fired=require_int(raw, "shots_fired"),
        kills=require_int(raw, "kills"),
        kills_per_min=require_float(raw, "kills_per_min"),
    )


def _parse_int_column(text: str, column: str) -> JSONValue:
    """Parse an integer column value or raise with column context.

    Args:
        text: Raw column text from a TSV row.
        column: Column name for the error message.

    Returns:
        Parsed int as a JSONValue.

    Raises:
        JSONTypeError: When ``text`` is not a base-10 integer.
    """
    if not _looks_like_int(text):
        raise JSONTypeError(f"index row column {column!r} expected int, got {text!r}")
    return int(text)


def _parse_float_column(text: str, column: str) -> JSONValue:
    """Parse a float column value or raise with column context.

    Args:
        text: Raw column text from a TSV row.
        column: Column name for the error message.

    Returns:
        Parsed float as a JSONValue.

    Raises:
        JSONTypeError: When ``text`` is not a finite float literal.
    """
    if not _looks_like_float(text):
        raise JSONTypeError(f"index row column {column!r} expected float, got {text!r}")
    return float(text)


def _looks_like_int(text: str) -> bool:
    """Return True when ``text`` matches a signed base-10 integer."""
    body = text[1:] if text.startswith(("-", "+")) else text
    return len(body) > 0 and body.isdigit()


def _looks_like_float(text: str) -> bool:
    """Return True when ``text`` matches a signed decimal number."""
    body = text[1:] if text.startswith(("-", "+")) else text
    if body.count(".") != 1:
        return False
    integer_part, fractional_part = body.split(".", 1)
    return integer_part.isdigit() and fractional_part.isdigit()


def append_index_row(row: BotRunIndexRowDict, index_path: Path = DEFAULT_INDEX_PATH) -> None:
    """Append a row to the index, writing the header if needed.

    File operations go through :mod:`tankpit_bot._test_hooks` so tests
    can inject fakes. ``append_text`` is used so concurrent writes are
    safe under OS append-mode semantics; the header is only written on
    the first call (idempotent guard via ``path_exists``).

    Args:
        row: Row to append.
        index_path: Index path. Defaults to :data:`DEFAULT_INDEX_PATH`.
    """
    if not _test_hooks.path_exists(index_path):
        _test_hooks.append_text(index_path, HEADER_LINE)
    _test_hooks.append_text(index_path, encode_row(row))


def load_index_rows(index_path: Path = DEFAULT_INDEX_PATH) -> list[BotRunIndexRowDict]:
    """Read every data row from the index, skipping the header.

    Args:
        index_path: Index path. Defaults to :data:`DEFAULT_INDEX_PATH`.

    Returns:
        Index rows in file order. Returns ``[]`` when the index file
        does not exist (no runs recorded yet).

    Raises:
        JSONTypeError: When a row fails to decode.
    """
    if not _test_hooks.path_exists(index_path):
        return []
    text = _test_hooks.read_text(index_path)
    rows: list[BotRunIndexRowDict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line_no == 1 and line.rstrip("\n") == HEADER_LINE.rstrip("\n"):
            continue
        rows.append(decode_row(line))
    return rows


def count_stall_timeouts(events_path: Path) -> int:
    """Count ``stall_timeout`` WIRE_COMPLETE events in a JSONL file.

    Walks the file once, parses each line as JSON, and counts records
    with ``channel == "WIRE_COMPLETE"`` and ``signal == "stall_timeout"``.

    Args:
        events_path: Path to a runtime events JSONL file.

    Returns:
        Number of qualifying records, or ``0`` when the file is
        missing. Blank lines are skipped silently. Any line that fails
        to parse propagates the underlying ``JSONTypeError``.

    Raises:
        JSONTypeError: When a line is not a JSON object.
    """
    from platform_core.json_utils import (
        load_json_str as _load_json_str,
    )
    from platform_core.json_utils import (
        narrow_json_to_dict as _narrow,
    )
    from platform_core.json_utils import (
        optional_str as _optional_str,
    )

    if not _test_hooks.path_exists(events_path):
        return 0
    text = _test_hooks.read_text(events_path)
    count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = _narrow(_load_json_str(line))
        if _optional_str(parsed, "channel") != "WIRE_COMPLETE":
            continue
        if _optional_str(parsed, "signal") == "stall_timeout":
            count += 1
    return count


def find_row(
    stamp: str,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> BotRunIndexRowDict | None:
    """Return the row whose ``stamp`` matches ``stamp``, or ``None``.

    Args:
        stamp: Run stamp to match (exact).
        index_path: Index path. Defaults to :data:`DEFAULT_INDEX_PATH`.

    Returns:
        The matching row, or ``None`` when no row has that stamp.
    """
    for row in load_index_rows(index_path):
        if row["stamp"] == stamp:
            return row
    return None


__all__ = [
    "DEFAULT_INDEX_PATH",
    "HEADER_LINE",
    "INDEX_COLUMNS",
    "BotRunIndexRowDict",
    "append_index_row",
    "count_stall_timeouts",
    "decode_row",
    "encode_row",
    "find_row",
    "load_index_rows",
    "make_index_row",
]
