"""Monitoring: worker processes, monitors, system info."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pytest
from platform_core.logging import JsonFormatter, get_logger

import handwriting_ai.monitoring as mon
from handwriting_ai._hook_protocols_system import MemoryInfoProtocol, PsutilProcessProtocol

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


def test_get_worker_processes_no_workers() -> None:
    from handwriting_ai import _test_hooks

    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    _test_hooks.psutil_process = _proc_factory

    workers = mon._get_worker_processes(100)
    assert workers == ()


def test_get_worker_processes_with_workers() -> None:
    from handwriting_ai import _test_hooks

    child1 = _DummyProcess(pid=101, rss=50 * 1024 * 1024)
    child2 = _DummyProcess(pid=102, rss=75 * 1024 * 1024)
    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[child1, child2])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    _test_hooks.psutil_process = _proc_factory

    workers = mon._get_worker_processes(100)
    assert len(workers) == 2
    assert workers[0]["pid"] == 101
    assert workers[0]["rss_bytes"] == 50 * 1024 * 1024
    assert workers[1]["pid"] == 102
    assert workers[1]["rss_bytes"] == 75 * 1024 * 1024


def test_get_worker_processes_lookup_failure_raises_after_logging() -> None:
    """Test that psutil.Process raises OSError after logging"""
    import psutil

    from handwriting_ai import _test_hooks

    def _proc_factory(pid: int | None = None) -> psutil.Process:
        raise OSError("Process not found")

    _test_hooks.psutil_process = _proc_factory
    # Should raise after logging
    with pytest.raises(OSError, match="Process not found"):
        mon._get_worker_processes(100)


def test_get_worker_processes_child_memory_failure_raises_after_logging() -> None:
    """Test that child.memory_info() raises exception after logging"""
    from handwriting_ai import _test_hooks

    class _FailingProcess:
        @property
        def pid(self) -> int:
            return 101

        def memory_info(self) -> MemoryInfoProtocol:
            raise OSError("Access denied")

        def children(self, recursive: bool = False) -> Sequence[PsutilProcessProtocol]:
            return [self]

    def _proc_factory(pid: int | None = None) -> PsutilProcessProtocol:
        return _FailingProcess()

    _test_hooks.psutil_process = _proc_factory
    # Should raise after logging
    with pytest.raises(OSError, match="Access denied"):
        mon._get_worker_processes(100)


def test_get_worker_processes_invalid_types() -> None:
    """Test line 185->179: isinstance check fails for non-int pid/rss.

    This test deliberately creates a fake that returns wrong types at runtime
    to verify the production code's isinstance defensive checks work correctly.
    """
    from handwriting_ai import _test_hooks

    class _BadMemInfo:
        @property
        def rss(self) -> str:
            return "not_an_int"

    class _BadProcess:
        @property
        def pid(self) -> int:
            return 101

        def memory_info(self) -> MemoryInfoProtocol:
            return _BadMemInfo()

        def children(self, recursive: bool = False) -> Sequence[PsutilProcessProtocol]:
            return [self]

    def _proc_factory(pid: int | None = None) -> PsutilProcessProtocol:
        return _BadProcess()

    _test_hooks.psutil_process = _proc_factory
    workers = mon._get_worker_processes(100)
    # Should skip the bad process and return empty tuple
    assert workers == ()


# Test CgroupMemoryMonitor


def test_cgroup_monitor_get_snapshot(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    # Set up cgroup files
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    stat_file = tmp_path / "memory.stat"

    current_file.write_text("524288000", encoding="utf-8")
    max_file.write_text("1048576000", encoding="utf-8")
    stat_file.write_text(
        "anon 100000000\nfile 200000000\nkernel 50000000\nslab 25000000\n",
        encoding="utf-8",
    )

    _test_hooks.cgroup_mem_current = current_file
    _test_hooks.cgroup_mem_max = max_file
    _test_hooks.cgroup_mem_stat = stat_file

    # Mock psutil for main process and workers
    child1 = _DummyProcess(pid=101, rss=50 * 1024 * 1024)
    child2 = _DummyProcess(pid=102, rss=75 * 1024 * 1024)
    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[child1, child2])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    _test_hooks.psutil_process = _proc_factory
    _test_hooks.os_getpid = lambda: 100

    monitor = mon.CgroupMemoryMonitor()
    snap = monitor.get_snapshot()

    assert snap["main_process"]["pid"] == 100
    assert snap["main_process"]["rss_bytes"] == 100 * 1024 * 1024
    assert len(snap["workers"]) == 2
    assert snap["cgroup_usage"]["usage_bytes"] == 524288000
    assert snap["cgroup_usage"]["limit_bytes"] == 1048576000
    assert abs(snap["cgroup_usage"]["percent"] - 50.0) < 0.01
    assert snap["cgroup_breakdown"]["anon_bytes"] == 100000000
    assert snap["cgroup_breakdown"]["file_bytes"] == 200000000


def test_cgroup_monitor_check_pressure(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("950000000", encoding="utf-8")
    max_file.write_text("1000000000", encoding="utf-8")

    _test_hooks.cgroup_mem_current = current_file
    _test_hooks.cgroup_mem_max = max_file

    monitor = mon.CgroupMemoryMonitor()
    assert monitor.check_pressure(threshold_percent=90.0) is True
    assert monitor.check_pressure(threshold_percent=96.0) is False


def test_cgroup_monitor_log_snapshot(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    # Set up minimal cgroup files
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    stat_file = tmp_path / "memory.stat"

    current_file.write_text("100000000", encoding="utf-8")
    max_file.write_text("200000000", encoding="utf-8")
    stat_file.write_text("anon 10000000\nfile 20000000\nkernel 5000000\nslab 2000000\n")

    _test_hooks.cgroup_mem_current = current_file
    _test_hooks.cgroup_mem_max = max_file
    _test_hooks.cgroup_mem_stat = stat_file

    parent = _DummyProcess(pid=100, rss=50 * 1024 * 1024, children=[])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    _test_hooks.psutil_process = _proc_factory
    _test_hooks.os_getpid = lambda: 100

    monitor = mon.CgroupMemoryMonitor()
    logger = get_logger("handwriting_ai")
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
    logger.addHandler(handler)
    try:
        monitor.log_snapshot(context="test_context")
    finally:
        logger.removeHandler(handler)

    out = buf.getvalue()
    assert "test_context memory" in out
    assert "main_rss_mb=" in out
    assert "cgroup_usage_mb=" in out
    assert "cgroup_pct=" in out


# Test SystemMemoryMonitor


def test_system_monitor_get_snapshot() -> None:
    from handwriting_ai import _test_hooks

    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    def _vm() -> _DummyVM:
        return _DummyVM(
            total=16 * 1024 * 1024 * 1024,
            available=8 * 1024 * 1024 * 1024,
            used=8 * 1024 * 1024 * 1024,
            percent=50.0,
        )

    _test_hooks.psutil_process = _proc_factory
    _test_hooks.psutil_virtual_memory = _vm
    _test_hooks.os_getpid = lambda: 100

    monitor = mon.SystemMemoryMonitor()
    snap = monitor.get_snapshot()

    assert snap["main_process"]["pid"] == 100
    assert snap["main_process"]["rss_bytes"] == 100 * 1024 * 1024
    assert len(snap["workers"]) == 0
    assert snap["cgroup_usage"]["usage_bytes"] == 8 * 1024 * 1024 * 1024
    assert snap["cgroup_usage"]["limit_bytes"] == 16 * 1024 * 1024 * 1024
    assert abs(snap["cgroup_usage"]["percent"] - 50.0) < 0.01
    # Breakdown: anon approximated by main RSS, file from buffers+cached (0 if not present)
    assert snap["cgroup_breakdown"]["anon_bytes"] == 100 * 1024 * 1024
    assert snap["cgroup_breakdown"]["file_bytes"] == 0  # No buffers/cached in _DummyVM
    assert snap["cgroup_breakdown"]["kernel_bytes"] == 0
    assert snap["cgroup_breakdown"]["slab_bytes"] == 0


def test_system_monitor_check_pressure() -> None:
    from handwriting_ai import _test_hooks

    def _vm() -> _DummyVM:
        return _DummyVM(
            total=100 * 1024 * 1024,
            available=5 * 1024 * 1024,
            used=95 * 1024 * 1024,
            percent=95.0,
        )

    _test_hooks.psutil_virtual_memory = _vm

    monitor = mon.SystemMemoryMonitor()
    assert monitor.check_pressure(threshold_percent=90.0) is True
    assert monitor.check_pressure(threshold_percent=96.0) is False


# Test module-level functions and detection


def test_detect_cgroups_available(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    current_file = tmp_path / "memory.current"
    current_file.write_text("1000", encoding="utf-8")

    _test_hooks.cgroup_mem_current = current_file

    assert mon._detect_cgroups_available() is True


def test_detect_cgroups_not_available(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    current = tmp_path / "current"

    _test_hooks.cgroup_mem_current = current

    assert mon._detect_cgroups_available() is False


def test_log_system_info_container(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    # Container mode: cgroup files exist
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("500000000", encoding="utf-8")
    max_file.write_text("1000000000", encoding="utf-8")

    _test_hooks.cgroup_mem_current = current_file
    _test_hooks.cgroup_mem_max = max_file

    def _cpu_count(*, logical: bool = True) -> int:
        return 8 if logical else 4

    _test_hooks.psutil_cpu_count = _cpu_count

    logger = get_logger("handwriting_ai")
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
    logger.addHandler(handler)
    try:
        mon.log_system_info()
    finally:
        logger.removeHandler(handler)

    out = buf.getvalue()
    assert "system_info" in out
    assert "cpu_logical=8" in out
    assert "cpu_physical=4" in out
    assert "cgroup_mem_limit_mb=" in out


def test_log_system_info_non_container_raises_when_vm_fails(tmp_path: Path) -> None:
    from handwriting_ai import _test_hooks

    # Non-container mode: no cgroup files
    current = tmp_path / "current"

    _test_hooks.cgroup_mem_current = current

    def _cpu_count(*, logical: bool = True) -> int:
        return 8 if logical else 4

    def _vm() -> _DummyVM:
        raise RuntimeError("VM info failed")

    _test_hooks.psutil_cpu_count = _cpu_count
    _test_hooks.psutil_virtual_memory = _vm

    logger = get_logger("handwriting_ai")
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter(static_fields={}, extra_field_names=[]))
    logger.addHandler(handler)
    try:
        # Should raise when virtual_memory fails
        with pytest.raises(RuntimeError, match="VM info failed"):
            mon.log_system_info()
    finally:
        logger.removeHandler(handler)


def test_create_monitor_cgroup(tmp_path: Path) -> None:
    """Test _create_monitor returns CgroupMemoryMonitor"""
    from handwriting_ai import _test_hooks

    current_file = tmp_path / "memory.current"
    current_file.write_text("1000", encoding="utf-8")

    _test_hooks.cgroup_mem_current = current_file

    monitor = mon._create_monitor()
    assert type(monitor) is mon.CgroupMemoryMonitor


def test_create_monitor_system(tmp_path: Path) -> None:
    """Test _create_monitor returns SystemMemoryMonitor"""
    from handwriting_ai import _test_hooks

    current = tmp_path / "current"

    _test_hooks.cgroup_mem_current = current

    def _vm() -> _DummyVM:
        return _DummyVM(
            total=16 * 1024 * 1024 * 1024,
            available=8 * 1024 * 1024 * 1024,
            used=8 * 1024 * 1024 * 1024,
            percent=50.0,
        )

    _test_hooks.psutil_virtual_memory = _vm

    monitor = mon._create_monitor()
    assert type(monitor) is mon.SystemMemoryMonitor


def test_get_monitor() -> None:
    """Test line 336: get_monitor returns module-level singleton"""
    monitor = mon.get_monitor()
    assert monitor is mon._monitor
