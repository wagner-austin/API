"""Integration tests for resources.py to achieve 100% coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import handwriting_ai.training.resources as res
from handwriting_ai import _test_hooks


def test_read_text_file_with_real_file() -> None:
    """Test read_text_file hook with actual file I/O."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("  test content  \n")
        temp_path = Path(f.name)

    try:
        # Use the hook directly (which calls the default implementation)
        result = _test_hooks.read_text_file(temp_path)
        assert result == "test content"
    finally:
        temp_path.unlink()


def test_detect_memory_limit_invalid_returns_none(tmp_path: Path) -> None:
    """Test return None when cgroup value is invalid."""
    mem_max_file = tmp_path / "memory.max"
    mem_max_file.write_text("max")  # Non-digit to skip cgroup

    _test_hooks.cgroup_mem_max = mem_max_file

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read

    # Return None when cgroup fails isdigit() check
    assert res._detect_memory_limit_bytes() is None


def test_detect_resource_limits_with_cpu_cgroup(tmp_path: Path) -> None:
    """Test cpu_cores = _detect_cpu_cores() when cgroup files exist."""
    # Create fake cgroup files
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000", encoding="utf-8")

    # Point hooks to our temp files
    _test_hooks.cgroup_cpu_max = cpu_max
    # Make memory cgroup path not exist
    _test_hooks.cgroup_mem_max = tmp_path / "nonexistent"

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read

    # Call detect_resource_limits
    limits = res.detect_resource_limits()
    # Should detect 2 cores from our fake file
    assert limits["cpu_cores"] == 2
