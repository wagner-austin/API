"""Monitoring: cgroup file reading and parsing."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pytest
from platform_core.logging import JsonFormatter, get_logger

import handwriting_ai.monitoring as mon

# Helper classes for mocking


class _DummyMemInfo:
    def __init__(self, rss: int) -> None:
        self._rss = rss

    @property
    def rss(self) -> int:
        return self._rss


class _DummyVM:
    def __init__(self, total: int, available: int, used: int, percent: float) -> None:
        self.total = total
        self.available = available
        self.used = used
        self.percent = percent


class _DummyProcess:
    def __init__(self, pid: int, rss: int, children: list[_DummyProcess] | None = None) -> None:
        self._pid = pid
        self._rss = rss
        self._children = children or []

    @property
    def pid(self) -> int:
        return self._pid

    def memory_info(self) -> _DummyMemInfo:
        return _DummyMemInfo(self._rss)

    def children(self, recursive: bool = False) -> Sequence[_DummyProcess]:
        return self._children


# Test cgroup file I/O functions


def test_read_cgroup_file_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("12345\n", encoding="utf-8")
    result = mon._read_cgroup_file(test_file)
    assert result == "12345"


def test_read_cgroup_file_failure() -> None:
    nonexistent = Path("/nonexistent/file.txt")
    with pytest.raises(FileNotFoundError):
        mon._read_cgroup_file(nonexistent)


def test_read_cgroup_int_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("67890", encoding="utf-8")
    result = mon._read_cgroup_int(test_file)
    assert result == 67890


def test_read_cgroup_int_invalid(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("not_a_number", encoding="utf-8")
    with pytest.raises(ValueError):
        mon._read_cgroup_int(test_file)


def test_parse_cgroup_stat_valid() -> None:
    content = """anon 123456
file 789012
kernel 345678
slab 901234
some_other_field 111222
"""
    result = mon._parse_cgroup_stat(content)
    assert result["anon"] == 123456
    assert result["file"] == 789012
    assert result["kernel"] == 345678
    assert result["slab"] == 901234
    assert result["some_other_field"] == 111222


def test_parse_cgroup_stat_empty() -> None:
    content = ""
    result = mon._parse_cgroup_stat(content)
    assert result == {}


def test_parse_cgroup_stat_with_empty_lines() -> None:
    """Test continue on empty lines"""
    content = """anon 100

file 200

kernel 300
"""
    result = mon._parse_cgroup_stat(content)
    assert result == {"anon": 100, "file": 200, "kernel": 300}


def test_parse_cgroup_stat_malformed_format() -> None:
    """Test parsing skips lines with invalid format (not 2 parts)"""
    content = """anon 100
file 200 extra_field
kernel
slab 300
"""
    result = mon._parse_cgroup_stat(content)
    # Should only include valid lines (anon and slab)
    assert result == {"anon": 100, "slab": 300}


def test_parse_cgroup_stat_invalid_integer_raises_after_logging() -> None:
    """Test parsing raises on lines with non-integer values after logging"""
    content = """anon 100
file not_a_number
kernel 300
slab -invalid
total 400
"""
    import pytest

    # Should raise ValueError when encountering non-integer value
    with pytest.raises(ValueError):
        mon._parse_cgroup_stat(content)


# Test cgroup reading functions


def test_read_cgroup_usage(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("524288000", encoding="utf-8")
    max_file.write_text("1048576000", encoding="utf-8")

    _test_hooks.cgroup_mem_current = current_file
    _test_hooks.cgroup_mem_max = max_file

    usage = mon._read_cgroup_usage()
    assert usage["usage_bytes"] == 524288000
    assert usage["limit_bytes"] == 1048576000
    assert abs(usage["percent"] - 50.0) < 0.01


def test_read_cgroup_usage_unlimited(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("524288000", encoding="utf-8")
    max_file.write_text("max", encoding="utf-8")

    _test_hooks.cgroup_mem_current = current_file
    _test_hooks.cgroup_mem_max = max_file

    with pytest.raises(RuntimeError, match="unlimited"):
        mon._read_cgroup_usage()


def test_read_cgroup_usage_no_files(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    current = tmp_path / "current"

    _test_hooks.cgroup_mem_current = current

    with pytest.raises(RuntimeError, match="no cgroup memory files found"):
        mon._read_cgroup_usage()


def test_read_cgroup_breakdown(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    stat_file = tmp_path / "memory.stat"
    stat_content = """anon 100000000
file 200000000
kernel 50000000
slab 25000000
other_field 12345
"""
    stat_file.write_text(stat_content, encoding="utf-8")
    _test_hooks.cgroup_mem_stat = stat_file

    breakdown = mon._read_cgroup_breakdown()
    assert breakdown["anon_bytes"] == 100000000
    assert breakdown["file_bytes"] == 200000000
    assert breakdown["kernel_bytes"] == 50000000
    assert breakdown["slab_bytes"] == 25000000


def test_read_cgroup_breakdown_missing_fields(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    stat_file = tmp_path / "memory.stat"
    stat_content = "other_field 12345\n"
    stat_file.write_text(stat_content, encoding="utf-8")
    _test_hooks.cgroup_mem_stat = stat_file

    breakdown = mon._read_cgroup_breakdown()
    assert breakdown["anon_bytes"] == 0
    assert breakdown["file_bytes"] == 0
    assert breakdown["kernel_bytes"] == 0
    assert breakdown["slab_bytes"] == 0


def test_read_cgroup_breakdown_no_files(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    stat = tmp_path / "stat"

    _test_hooks.cgroup_mem_stat = stat

    with pytest.raises(RuntimeError, match=r"no cgroup memory\.stat file found"):
        mon._read_cgroup_breakdown()


def test_read_cgroup_breakdown_empty_file(tmp_path: Path) -> None:
    """Test that empty stat file raises RuntimeError"""
    from handwriting_ai import _test_hooks

    stat_file = tmp_path / "memory.stat"
    stat_file.write_text("", encoding="utf-8")
    _test_hooks.cgroup_mem_stat = stat_file

    with pytest.raises(RuntimeError, match="parsing produced no valid entries"):
        mon._read_cgroup_breakdown()


def test_read_cgroup_breakdown_all_invalid_lines_raises_after_logging(tmp_path: Path) -> None:
    """Test that file with all invalid lines raises after logging"""
    from handwriting_ai import _test_hooks

    stat_file = tmp_path / "memory.stat"
    stat_content = """invalid line format
field1 not_a_number
single_field
"""
    stat_file.write_text(stat_content, encoding="utf-8")
    _test_hooks.cgroup_mem_stat = stat_file

    # Should raise ValueError from _parse_cgroup_stat after logging
    with pytest.raises(ValueError):
        mon._read_cgroup_breakdown()


def test_read_cgroup_breakdown_missing_core_metrics(tmp_path: Path) -> None:
    """Test that missing core metrics (anon=0, file=0) logs warning but continues"""

    from handwriting_ai import _test_hooks

    stat_file = tmp_path / "memory.stat"
    stat_content = """kernel 50000000
slab 25000000
"""
    stat_file.write_text(stat_content, encoding="utf-8")
    _test_hooks.cgroup_mem_stat = stat_file

    logger = get_logger("handwriting_ai")
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
    logger.addHandler(handler)
    try:
        breakdown = mon._read_cgroup_breakdown()
    finally:
        logger.removeHandler(handler)

    # Should return breakdown with zeros for anon and file
    assert breakdown["anon_bytes"] == 0
    assert breakdown["file_bytes"] == 0
    assert breakdown["kernel_bytes"] == 50000000
    assert breakdown["slab_bytes"] == 25000000

    # Should have logged warning
    out = buf.getvalue()
    assert "cgroup_breakdown_missing_core_metrics" in out


# Test worker process detection
