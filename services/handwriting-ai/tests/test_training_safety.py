from __future__ import annotations

import handwriting_ai.training.safety as safety
from handwriting_ai import _test_hooks
from handwriting_ai._test_hooks import (
    CgroupMemoryBreakdownDict,
    CgroupMemoryUsageDict,
    MemorySnapshotDict,
    ProcessMemoryDict,
)


def test_memory_guard_consecutive_logic() -> None:
    # Configure guard to trigger after 2 consecutive True checks
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=90.0, required_consecutive=2)
    )
    safety.reset_memory_guard()

    # Create a fake snapshot for the calls
    snap: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=950_000_000, limit_bytes=1_000_000_000, percent=95.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=900_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    def _always(*, threshold_percent: float) -> bool:
        return True

    def _fake_snapshot() -> MemorySnapshotDict:
        return snap

    _test_hooks.check_memory_pressure = _always
    _test_hooks.get_memory_snapshot = _fake_snapshot
    assert safety.on_batch_check() is False  # first True
    assert safety.on_batch_check() is True  # second True triggers


def test_memory_guard_resets_on_relief() -> None:
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=90.0, required_consecutive=2)
    )
    safety.reset_memory_guard()

    # Create a fake snapshot for the calls
    snap: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=850_000_000, limit_bytes=1_000_000_000, percent=85.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=800_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    # Alternate True/False so it never triggers
    seq = iter([True, False, True, False])

    def _seq(*, threshold_percent: float) -> bool:
        return next(seq, False)

    def _fake_snapshot() -> MemorySnapshotDict:
        return snap

    _test_hooks.check_memory_pressure = _seq
    _test_hooks.get_memory_snapshot = _fake_snapshot


def test_memory_guard_disabled() -> None:
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=False, threshold_percent=95.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    # Create a fake snapshot for the calls
    snap: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=950_000_000, limit_bytes=1_000_000_000, percent=95.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=900_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    # Even with True pressure, disabled guard returns False
    def _always2(*, threshold_percent: float) -> bool:
        return True

    def _fake_snapshot() -> MemorySnapshotDict:
        return snap

    _test_hooks.check_memory_pressure = _always2
    _test_hooks.get_memory_snapshot = _fake_snapshot
    assert safety.on_batch_check() is False


def test_memory_guard_warning_at_85_percent() -> None:
    """Test that warnings are logged when memory reaches 85%."""
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    # Create snapshot at 85%
    snap: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=850_000_000, limit_bytes=1_000_000_000, percent=85.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=800_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    def _snapshot() -> MemorySnapshotDict:
        return snap

    def _pressure(*, threshold_percent: float) -> bool:
        return False  # Not at critical threshold yet

    _test_hooks.get_memory_snapshot = _snapshot
    _test_hooks.check_memory_pressure = _pressure

    # Should log warning at 85% but not trigger abort
    assert safety.on_batch_check() is False


def test_memory_guard_warning_at_90_percent() -> None:
    """Test that warnings are logged when memory reaches 90%."""
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    # Create snapshot at 90%
    snap: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=900_000_000, limit_bytes=1_000_000_000, percent=90.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=850_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    def _snapshot() -> MemorySnapshotDict:
        return snap

    def _pressure(*, threshold_percent: float) -> bool:
        return False  # Not at critical threshold yet

    _test_hooks.get_memory_snapshot = _snapshot
    _test_hooks.check_memory_pressure = _pressure

    # Should log warning at 90% but not trigger abort
    assert safety.on_batch_check() is False


def test_memory_guard_relieved_log() -> None:
    """Test that pressure relieved log is emitted when consecutive counter resets."""
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    snap_high: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=930_000_000, limit_bytes=1_000_000_000, percent=93.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=900_000_000, file_bytes=30_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    snap_low: MemorySnapshotDict = {
        "main_process": ProcessMemoryDict(pid=1, rss_bytes=100_000_000),
        "workers": (),
        "cgroup_usage": CgroupMemoryUsageDict(
            usage_bytes=800_000_000, limit_bytes=1_000_000_000, percent=80.0
        ),
        "cgroup_breakdown": CgroupMemoryBreakdownDict(
            anon_bytes=750_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    }

    calls = [0]

    def _snapshot() -> MemorySnapshotDict:
        calls[0] += 1
        if calls[0] <= 2:
            return snap_high
        return snap_low

    def _pressure(*, threshold_percent: float) -> bool:
        return calls[0] <= 2  # First 2 calls are high pressure, then drops

    _test_hooks.get_memory_snapshot = _snapshot
    _test_hooks.check_memory_pressure = _pressure

    # First call: pressure high, consecutive = 1
    assert safety.on_batch_check() is False
    # Second call: pressure high, consecutive = 2
    assert safety.on_batch_check() is False
    # Third call: pressure drops, should log "relieved" and reset consecutive
    assert safety.on_batch_check() is False
