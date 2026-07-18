"""Tests for the bot runs index module.

File I/O goes through ``_test_hooks`` (save-and-restore on the global
``_test_hooks.path_exists`` / ``_test_hooks.read_text`` /
``_test_hooks.append_text``). No monkeypatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    AppendTextProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
)
from tankpit_bot.diagnostics import runs_index
from tankpit_bot.diagnostics.runs_index import (
    DEFAULT_INDEX_PATH,
    HEADER_LINE,
    INDEX_COLUMNS,
    BotRunIndexRowDict,
    append_index_row,
    count_stall_timeouts,
    decode_row,
    encode_row,
    find_row,
    load_index_rows,
    make_index_row,
)


class _FakeFileSystem:
    """Save-and-restore fake for ``_test_hooks.path_exists`` / ``read_text``
    / ``append_text``."""

    def __init__(self) -> None:
        """Initialise with no virtual files registered."""
        self._files: dict[str, str] = {}

    def write(self, path: Path, content: str) -> None:
        """Register a virtual file's contents."""
        self._files[str(path)] = content

    def path_exists(self, path: Path) -> bool:
        """Return True when ``path`` was registered via :meth:`write`."""
        return str(path) in self._files

    def read_text(self, path: Path) -> str:
        """Return the contents of ``path``."""
        return self._files[str(path)]

    def append_text(self, path: Path, content: str) -> None:
        """Append ``content`` to the virtual file at ``path``."""
        existing = self._files.get(str(path), "")
        self._files[str(path)] = existing + content


def _install_fake_filesystem() -> tuple[
    _FakeFileSystem, PathExistsProtocol, ReadTextProtocol, AppendTextProtocol
]:
    """Swap the real script hooks for a fake; return originals for restore."""
    fake = _FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    original_append_text: AppendTextProtocol = _test_hooks.append_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    _test_hooks.append_text = fake.append_text
    return (fake, original_path_exists, original_read_text, original_append_text)


def _sample_row() -> BotRunIndexRowDict:
    """Return a representative row for round-trip tests."""
    return BotRunIndexRowDict(
        stamp="20260620-150138",
        duration_s=155,
        exit_reason="completed",
        ticks=1543,
        stalls=2,
        shots_fired=16,
        kills=3,
        kills_per_min=1.16,
    )


class TestIndexColumns:
    """Tests for the public column-order contract."""

    def test_columns_match_typed_dict_keys(self) -> None:
        """Column order matches :class:`BotRunIndexRowDict` declaration order."""
        # The TypedDict declares fields in this exact order; encode_row
        # depends on it. If the contract drifts, the encode/decode
        # round trip will break -- this test is the early-warning gate.
        assert INDEX_COLUMNS == (
            "stamp",
            "duration_s",
            "exit_reason",
            "ticks",
            "stalls",
            "shots_fired",
            "kills",
            "kills_per_min",
        )

    def test_header_line_uses_column_order(self) -> None:
        """``HEADER_LINE`` is the columns joined by tabs plus a newline."""
        assert "\t".join(INDEX_COLUMNS) + "\n" == HEADER_LINE


class TestMakeIndexRow:
    """Tests for the row constructor's derived ``kills_per_min``."""

    def test_zero_kills_yields_zero_rate(self) -> None:
        """No kills -> rate is exactly ``0.0`` regardless of duration."""
        row = make_index_row(
            stamp="s",
            duration_s=60,
            exit_reason="completed",
            ticks=600,
            stalls=0,
            shots_fired=0,
            kills=0,
        )
        assert row["kills_per_min"] == 0.0

    def test_three_kills_in_sixty_seconds_is_three_per_min(self) -> None:
        """Trivial sanity check on the per-minute scaling."""
        row = make_index_row(
            stamp="s",
            duration_s=60,
            exit_reason="completed",
            ticks=600,
            stalls=0,
            shots_fired=10,
            kills=3,
        )
        assert row["kills_per_min"] == 3.0

    def test_zero_duration_guard_uses_max_one(self) -> None:
        """A zero-duration run treats duration as 1 second to avoid /0."""
        row = make_index_row(
            stamp="s",
            duration_s=0,
            exit_reason="completed",
            ticks=0,
            stalls=0,
            shots_fired=0,
            kills=2,
        )
        # 2 kills / max(1, 0) * 60 = 120.0
        assert row["kills_per_min"] == 120.0


class TestEncodeRow:
    """Tests for TSV encoding."""

    def test_encodes_in_column_order(self) -> None:
        """Encoded line matches the documented TSV layout."""
        row = _sample_row()
        encoded = encode_row(row)
        assert encoded.endswith("\n")
        parts = encoded.rstrip("\n").split("\t")
        assert parts == [
            "20260620-150138",
            "155",
            "completed",
            "1543",
            "2",
            "16",
            "3",
            "1.16",
        ]

    def test_floats_have_two_decimal_places(self) -> None:
        """``kills_per_min`` is always rendered to 2 decimal places."""
        row = BotRunIndexRowDict(
            stamp="s",
            duration_s=60,
            exit_reason="completed",
            ticks=600,
            stalls=0,
            shots_fired=0,
            kills=0,
            kills_per_min=0.0,
        )
        assert encode_row(row).split("\t")[-1] == "0.00\n"


class TestDecodeRow:
    """Tests for TSV decoding."""

    def test_round_trips_through_encode(self) -> None:
        """``decode_row(encode_row(row)) == row`` for the sample row."""
        row = _sample_row()
        assert decode_row(encode_row(row)) == row

    def test_strips_trailing_newline(self) -> None:
        """Trailing newlines do not affect decoding."""
        row = _sample_row()
        encoded = encode_row(row)
        assert decode_row(encoded.rstrip("\n")) == row

    def test_rejects_short_row(self) -> None:
        """A row missing a column raises with column-count context."""
        with pytest.raises(JSONTypeError, match="7 columns, expected 8"):
            decode_row("s\t1\tcompleted\t1\t1\t1\t1")

    def test_rejects_long_row(self) -> None:
        """A row with extra columns raises with column-count context."""
        encoded = encode_row(_sample_row()).rstrip("\n") + "\textra"
        with pytest.raises(JSONTypeError, match="9 columns, expected 8"):
            decode_row(encoded)

    def test_rejects_non_int_duration(self) -> None:
        """A non-integer integer-column value raises with column context."""
        bad = "s\tNOT_AN_INT\tcompleted\t1\t1\t1\t1\t1.00"
        with pytest.raises(JSONTypeError, match="duration_s"):
            decode_row(bad)

    def test_rejects_non_float_rate(self) -> None:
        """A non-float ``kills_per_min`` raises with column context."""
        bad = "s\t1\tcompleted\t1\t1\t1\t1\tNOT_A_FLOAT"
        with pytest.raises(JSONTypeError, match="kills_per_min"):
            decode_row(bad)

    def test_accepts_signed_integers(self) -> None:
        """``-`` and ``+`` prefixes parse correctly."""
        # The schema only emits non-negatives, but the parser is permissive.
        encoded = "s\t-5\tcompleted\t+10\t0\t0\t0\t0.00"
        decoded = decode_row(encoded)
        assert decoded["duration_s"] == -5
        assert decoded["ticks"] == 10


class TestParseHelpers:
    """Direct branch coverage for the private parse helpers."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0", True),
            ("123", True),
            ("-5", True),
            ("+10", True),
            ("", False),
            ("-", False),
            ("1.5", False),
            ("abc", False),
        ],
    )
    def test_looks_like_int(self, text: str, expected: bool) -> None:
        """Each predicate verdict matches the documented rules."""
        assert runs_index._looks_like_int(text) is expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1.5", True),
            ("-0.25", True),
            ("+12.34", True),
            ("0.00", True),
            ("1", False),  # no dot
            ("1.2.3", False),  # too many dots
            (".5", False),  # missing integer part
            ("1.", False),  # missing fractional part
            ("", False),
            ("abc.def", False),
        ],
    )
    def test_looks_like_float(self, text: str, expected: bool) -> None:
        """Each predicate verdict matches the documented rules."""
        assert runs_index._looks_like_float(text) is expected


class TestAppendIndexRow:
    """Tests for index append + header guard via the fake filesystem."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
            self._original_append_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text
        _test_hooks.append_text = self._original_append_text

    def test_writes_header_when_file_missing(self) -> None:
        """First append writes the header before the row."""
        path = Path("runs/bot/_index.tsv")
        append_index_row(_sample_row(), path)
        text = self._fake.read_text(path)
        assert text.startswith(HEADER_LINE)
        assert text.endswith(encode_row(_sample_row()))

    def test_subsequent_appends_skip_header(self) -> None:
        """Second append does not duplicate the header."""
        path = Path("runs/bot/_index.tsv")
        append_index_row(_sample_row(), path)
        append_index_row(_sample_row(), path)
        text = self._fake.read_text(path)
        # Exactly one header, exactly two data rows.
        assert text.count(HEADER_LINE) == 1
        assert text.count(encode_row(_sample_row())) == 2


class TestLoadIndexRows:
    """Tests for reading every row, skipping the header and blank lines."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
            self._original_append_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text
        _test_hooks.append_text = self._original_append_text

    def test_returns_empty_when_file_missing(self) -> None:
        """A missing index yields an empty list (no runs recorded)."""
        assert load_index_rows(Path("runs/bot/missing.tsv")) == []

    def test_skips_header_and_blanks(self) -> None:
        """The header line is skipped; blank lines are skipped silently."""
        path = Path("runs/bot/_index.tsv")
        self._fake.write(
            path,
            HEADER_LINE + "\n" + encode_row(_sample_row()) + "\n" + encode_row(_sample_row()),
        )
        rows = load_index_rows(path)
        assert rows == [_sample_row(), _sample_row()]

    def test_handles_index_with_no_header(self) -> None:
        """Indices without a header (legacy file) still decode every line."""
        path = Path("runs/bot/_index.tsv")
        self._fake.write(path, encode_row(_sample_row()))
        assert load_index_rows(path) == [_sample_row()]


class TestFindRow:
    """Tests for the find_row stamp lookup."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
            self._original_append_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text
        _test_hooks.append_text = self._original_append_text

    def test_returns_matching_row(self) -> None:
        """Exact stamp match returns the row."""
        path = Path("runs/bot/_index.tsv")
        self._fake.write(path, HEADER_LINE + encode_row(_sample_row()))
        assert find_row("20260620-150138", path) == _sample_row()

    def test_returns_none_when_stamp_missing(self) -> None:
        """No match returns ``None`` rather than raising."""
        path = Path("runs/bot/_index.tsv")
        self._fake.write(path, HEADER_LINE + encode_row(_sample_row()))
        assert find_row("20990101-000000", path) is None


class TestCountStallTimeouts:
    """Tests for the stall counter helper."""

    def setup_method(self) -> None:
        """Install the fake filesystem."""
        (
            self._fake,
            self._original_path_exists,
            self._original_read_text,
            self._original_append_text,
        ) = _install_fake_filesystem()

    def teardown_method(self) -> None:
        """Restore the real ``_test_hooks`` bindings."""
        _test_hooks.path_exists = self._original_path_exists
        _test_hooks.read_text = self._original_read_text
        _test_hooks.append_text = self._original_append_text

    def test_returns_zero_when_file_missing(self) -> None:
        """A missing events file counts as zero stalls."""
        assert count_stall_timeouts(Path("runs/bot/missing.jsonl")) == 0

    def test_counts_only_action_outcome_stall_timeout(self) -> None:
        """Filter is strict on both ``diagnostic_kind`` and ``outcome``."""
        from platform_core.json_utils import dump_json_str

        # Two qualifying records, one near-miss (wrong kind), one
        # near-miss (wrong outcome), and a blank line.
        lines = [
            dump_json_str(
                {
                    "timestamp": "2026-06-20T15:00:00",
                    "level": "INFO",
                    "logger": "tankpit_bot.runtime.events",
                    "mode": "bot",
                    "channel": "DIAGNOSTIC",
                    "message": "move stalled",
                    "diagnostic_kind": "action_outcome",
                    "action_kind": "move",
                    "duration_ms": 10000,
                    "outcome": "stall_timeout",
                }
            ),
            dump_json_str(
                {
                    "timestamp": "2026-06-20T15:00:01",
                    "level": "INFO",
                    "logger": "tankpit_bot.runtime.events",
                    "mode": "bot",
                    "channel": "AI",  # wrong kind (no diagnostic_kind)
                    "message": "stall_timeout",
                    "outcome": "stall_timeout",
                }
            ),
            dump_json_str(
                {
                    "timestamp": "2026-06-20T15:00:02",
                    "level": "INFO",
                    "logger": "tankpit_bot.runtime.events",
                    "mode": "bot",
                    "channel": "DIAGNOSTIC",
                    "message": "map_open resolved",
                    "diagnostic_kind": "action_outcome",
                    "action_kind": "map_open",
                    "duration_ms": 250,
                    "outcome": "map_data_processed",  # wrong outcome
                }
            ),
            "",  # blank line should be skipped
            dump_json_str(
                {
                    "timestamp": "2026-06-20T15:00:03",
                    "level": "INFO",
                    "logger": "tankpit_bot.runtime.events",
                    "mode": "bot",
                    "channel": "DIAGNOSTIC",
                    "message": "teleport stalled",
                    "diagnostic_kind": "action_outcome",
                    "action_kind": "teleport",
                    "duration_ms": 10000,
                    "outcome": "stall_timeout",
                }
            ),
        ]
        path = Path("runs/bot/events.jsonl")
        self._fake.write(path, "\n".join(lines))
        assert count_stall_timeouts(path) == 2


def test_default_index_path_is_runs_bot_index_tsv() -> None:
    """The CLI defaults to ``runs/bot/_index.tsv``."""
    assert Path("runs/bot/_index.tsv") == DEFAULT_INDEX_PATH
