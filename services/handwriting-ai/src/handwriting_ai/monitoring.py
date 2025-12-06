from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, TypedDict

import psutil
from platform_core.logging import get_logger

# Cgroup v2 paths
_CGROUP_MEM_CURRENT: Path = Path("/sys/fs/cgroup/memory.current")
_CGROUP_MEM_MAX: Path = Path("/sys/fs/cgroup/memory.max")
_CGROUP_MEM_STAT: Path = Path("/sys/fs/cgroup/memory.stat")


# Minimal protocols for psutil return types
class _MemInfoProto(Protocol):
    """Protocol for psutil memory_info() result - read-only rss attribute."""

    @property
    def rss(self) -> int: ...


class _ProcessProto(Protocol):
    """Protocol for psutil.Process - read-only pid and memory_info()."""

    @property
    def pid(self) -> int: ...

    def memory_info(self) -> _MemInfoProto: ...


class _VirtualMemoryProto(Protocol):
    """Protocol for psutil.virtual_memory() - read-only total and used."""

    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...


class CgroupMemoryUsage(TypedDict):
    """Cgroup-level memory usage (what the kernel OOM killer sees)."""

    usage_bytes: int
    limit_bytes: int
    percent: float


class CgroupMemoryBreakdown(TypedDict):
    """Detailed memory breakdown from cgroup memory.stat."""

    anon_bytes: int
    file_bytes: int
    kernel_bytes: int
    slab_bytes: int


class ProcessMemory(TypedDict):
    """Per-process memory information."""

    pid: int
    rss_bytes: int


class MemorySnapshot(TypedDict):
    """Complete memory snapshot including process, cgroup, and worker data."""

    main_process: ProcessMemory
    workers: tuple[ProcessMemory, ...]
    cgroup_usage: CgroupMemoryUsage
    cgroup_breakdown: CgroupMemoryBreakdown


class MemoryMonitor(Protocol):
    """Protocol for memory monitoring implementations."""

    def get_snapshot(self) -> MemorySnapshot:
        """Capture current memory snapshot."""
        ...

    def check_pressure(self, threshold_percent: float) -> bool:
        """Return True if memory usage exceeds threshold."""
        ...

    def log_snapshot(self, context: str) -> None:
        """Log memory snapshot with optional context."""
        ...


def _read_cgroup_file(path: Path) -> str:
    """Read cgroup file contents, raising on failure."""
    return path.read_text(encoding="utf-8").strip()


def _read_cgroup_int(path: Path) -> int:
    """Read cgroup file as integer, raising on failure."""
    content = _read_cgroup_file(path)
    return int(content)


def _parse_cgroup_stat(content: str) -> dict[str, int]:
    """Parse cgroup memory.stat format into key-value pairs.

    Kernel format: each line is "<key> <value>" where value is an integer.
    This parser targets the documented cgroup v2 memory.stat format.

    Skips malformed lines with logging to handle edge cases gracefully
    while maintaining visibility into parsing issues.
    """
    from platform_core.logging import get_logger

    logger = get_logger("handwriting_ai")
    result: dict[str, int] = {}

    for line_num, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            logger.debug(
                "cgroup_stat_parse_skip line=%d reason=invalid_format expected=2_parts got=%d",
                line_num,
                len(parts),
            )
            continue

        key, value_str = parts
        try:
            value = int(value_str)
        except ValueError as exc:
            logger.error(
                "cgroup_stat_parse_skip line=%d key=%s reason=invalid_int value=%r error=%s",
                line_num,
                key,
                value_str,
                exc,
            )
            raise

        result[key] = value

    return result


def _read_cgroup_usage() -> CgroupMemoryUsage:
    """Read cgroup v2 memory usage and limit."""
    if not _CGROUP_MEM_CURRENT.exists():
        msg = "no cgroup memory files found (not in container?)"
        raise RuntimeError(msg)

    usage_bytes = _read_cgroup_int(_CGROUP_MEM_CURRENT)
    limit_content = _read_cgroup_file(_CGROUP_MEM_MAX)
    if limit_content == "max":
        msg = "cgroup memory.max is 'max' (unlimited)"
        raise RuntimeError(msg)
    limit_bytes = int(limit_content)

    percent = (float(usage_bytes) / float(limit_bytes)) * 100.0
    return {
        "usage_bytes": usage_bytes,
        "limit_bytes": limit_bytes,
        "percent": percent,
    }


def _read_cgroup_breakdown() -> CgroupMemoryBreakdown:
    """Read cgroup v2 memory breakdown from memory.stat.

    Validates that at least one core metric (anon or file) is present
    to ensure we got valid cgroup data.
    """

    if not _CGROUP_MEM_STAT.exists():
        msg = "no cgroup memory.stat file found"
        raise RuntimeError(msg)

    stat_content = _read_cgroup_file(_CGROUP_MEM_STAT)
    stats = _parse_cgroup_stat(stat_content)

    # Validate we got at least some expected fields
    if not stats:
        msg = "cgroup memory.stat parsing produced no valid entries"
        raise RuntimeError(msg)

    # Extract required fields
    anon = stats.get("anon", 0)
    file_cache = stats.get("file", 0)
    kernel = stats.get("kernel", 0)
    slab = stats.get("slab", 0)

    # Validate at least one core metric is present
    if anon == 0 and file_cache == 0:
        logger = get_logger("handwriting_ai")
        logger.warning(
            "cgroup_breakdown_missing_core_metrics anon=0 file=0 kernel=%d slab=%d fields=%s",
            kernel,
            slab,
            sorted(stats.keys()),
        )

    return {
        "anon_bytes": anon,
        "file_bytes": file_cache,
        "kernel_bytes": kernel,
        "slab_bytes": slab,
    }


def _get_worker_processes(parent_pid: int) -> tuple[ProcessMemory, ...]:
    """Find all child processes of parent_pid and return their memory usage."""
    from platform_core.logging import get_logger

    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except (OSError, ValueError, RuntimeError) as exc:
        get_logger("handwriting_ai").error(
            "worker_process_lookup_failed pid=%s %s", parent_pid, exc
        )
        raise

    workers: list[ProcessMemory] = []
    for child in children:
        try:
            # Type-annotate to use Protocol - overrides Any from psutil
            child_typed: _ProcessProto = child
            mem_info: _MemInfoProto = child_typed.memory_info()
            rss_val: int = mem_info.rss
            pid_val: int = child_typed.pid
            if not isinstance(pid_val, int) or not isinstance(rss_val, int):
                get_logger("handwriting_ai").error(
                    "worker_memory_invalid_types pid_type=%s rss_type=%s",
                    type(pid_val).__name__,
                    type(rss_val).__name__,
                )
                continue
            workers.append({"pid": pid_val, "rss_bytes": rss_val})
        except (OSError, ValueError, RuntimeError) as exc:
            get_logger("handwriting_ai").error("worker_memory_read_failed %s", exc)
            raise

    return tuple(workers)


class CgroupMemoryMonitor:
    """Memory monitor using cgroup metrics (container environments)."""

    def get_snapshot(self) -> MemorySnapshot:
        """Capture complete memory snapshot including cgroup and worker data."""
        pid = os.getpid()
        proc: _ProcessProto = psutil.Process(pid)
        mem_info: _MemInfoProto = proc.memory_info()
        main_rss: int = mem_info.rss

        main_process: ProcessMemory = {"pid": pid, "rss_bytes": main_rss}
        workers = _get_worker_processes(pid)
        cgroup_usage = _read_cgroup_usage()
        cgroup_breakdown = _read_cgroup_breakdown()

        return {
            "main_process": main_process,
            "workers": workers,
            "cgroup_usage": cgroup_usage,
            "cgroup_breakdown": cgroup_breakdown,
        }

    def check_pressure(self, threshold_percent: float) -> bool:
        """Return True if cgroup memory usage exceeds threshold."""
        cgroup = _read_cgroup_usage()
        return cgroup["percent"] >= threshold_percent

    def log_snapshot(self, context: str = "") -> None:
        """Log comprehensive memory snapshot with optional context prefix."""
        log = get_logger("handwriting_ai")
        log.setLevel(20)
        snap = self.get_snapshot()
        ctx = f"{context} " if context else ""

        main_mb = snap["main_process"]["rss_bytes"] // (1024 * 1024)
        workers_mb = sum(w["rss_bytes"] for w in snap["workers"]) // (1024 * 1024)
        cgroup_usage_mb = snap["cgroup_usage"]["usage_bytes"] // (1024 * 1024)
        cgroup_limit_mb = snap["cgroup_usage"]["limit_bytes"] // (1024 * 1024)

        anon_mb = snap["cgroup_breakdown"]["anon_bytes"] // (1024 * 1024)
        file_mb = snap["cgroup_breakdown"]["file_bytes"] // (1024 * 1024)
        kernel_mb = snap["cgroup_breakdown"]["kernel_bytes"] // (1024 * 1024)
        slab_mb = snap["cgroup_breakdown"]["slab_bytes"] // (1024 * 1024)

        worker_count = len(snap["workers"])
        log.info(
            f"{ctx}memory "
            f"main_rss_mb={main_mb} workers_rss_mb={workers_mb} worker_count={worker_count} "
            f"cgroup_usage_mb={cgroup_usage_mb} cgroup_limit_mb={cgroup_limit_mb} "
            f"cgroup_pct={snap['cgroup_usage']['percent']:.1f} "
            f"anon_mb={anon_mb} file_mb={file_mb} kernel_mb={kernel_mb} slab_mb={slab_mb}"
        )


class SystemMemoryMonitor:
    """Memory monitor using system metrics (test/dev environments without cgroups)."""

    def __init__(self) -> None:
        # Get system memory total once for consistent "limit"
        self._system_total = int(psutil.virtual_memory().total)

    def get_snapshot(self) -> MemorySnapshot:
        """Capture basic memory snapshot using system metrics."""
        pid = os.getpid()
        proc: _ProcessProto = psutil.Process(pid)
        mem_info: _MemInfoProto = proc.memory_info()
        main_rss: int = mem_info.rss

        main_process: ProcessMemory = {"pid": pid, "rss_bytes": main_rss}
        workers = _get_worker_processes(pid)

        # Create synthetic cgroup usage from system memory
        vm: _VirtualMemoryProto = psutil.virtual_memory()
        usage_bytes = int(vm.used)
        limit_bytes = self._system_total
        percent = (float(usage_bytes) / float(limit_bytes)) * 100.0

        cgroup_usage: CgroupMemoryUsage = {
            "usage_bytes": usage_bytes,
            "limit_bytes": limit_bytes,
            "percent": percent,
        }

        # Emulate cgroup breakdown using psutil.virtual_memory()
        # In non-cgroup environments, only main process RSS is reliably available across platforms.
        # Buffers/cached are Linux-specific and not exposed via psutil's cross-platform API.
        # Set file/kernel/slab to 0 since they're not available outside cgroup environments.
        anon_approx = main_rss

        cgroup_breakdown: CgroupMemoryBreakdown = {
            "anon_bytes": anon_approx,
            "file_bytes": 0,  # Not available via psutil cross-platform API
            "kernel_bytes": 0,  # Not available via psutil
            "slab_bytes": 0,  # Not available via psutil
        }

        return {
            "main_process": main_process,
            "workers": workers,
            "cgroup_usage": cgroup_usage,
            "cgroup_breakdown": cgroup_breakdown,
        }

    def check_pressure(self, threshold_percent: float) -> bool:
        """Return True if system memory usage exceeds threshold."""
        vm = psutil.virtual_memory()
        percent = (float(vm.used) / float(self._system_total)) * 100.0
        return percent >= threshold_percent

    def log_snapshot(self, context: str = "") -> None:
        """Log basic memory snapshot with system metrics."""
        log = get_logger("handwriting_ai")
        log.setLevel(20)
        snap = self.get_snapshot()
        ctx = f"{context} " if context else ""

        main_mb = snap["main_process"]["rss_bytes"] // (1024 * 1024)
        workers_mb = sum(w["rss_bytes"] for w in snap["workers"]) // (1024 * 1024)
        usage_mb = snap["cgroup_usage"]["usage_bytes"] // (1024 * 1024)
        limit_mb = snap["cgroup_usage"]["limit_bytes"] // (1024 * 1024)

        worker_count = len(snap["workers"])
        log.info(
            f"{ctx}memory "
            f"main_rss_mb={main_mb} workers_rss_mb={workers_mb} worker_count={worker_count} "
            f"system_usage_mb={usage_mb} system_total_mb={limit_mb} "
            f"system_pct={snap['cgroup_usage']['percent']:.1f}"
        )


def _detect_cgroups_available() -> bool:
    """Detect if cgroup v2 memory files are available."""
    return _CGROUP_MEM_CURRENT.exists()


def is_cgroup_available() -> bool:
    """Public helper to indicate whether cgroup memory metrics are available.

    Used by components that need to adapt behavior (e.g., calibration preflight)
    when running outside containerized environments.
    """
    return _detect_cgroups_available()


def _create_monitor() -> MemoryMonitor:
    """Create appropriate memory monitor based on environment."""
    if _detect_cgroups_available():
        return CgroupMemoryMonitor()
    return SystemMemoryMonitor()


# Module-level singleton monitor instance
_monitor: MemoryMonitor = _create_monitor()


def get_monitor() -> MemoryMonitor:
    """Get the module-level memory monitor instance."""
    return _monitor


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory snapshot using the active monitor."""
    return _monitor.get_snapshot()


def check_memory_pressure(threshold_percent: float = 90.0) -> bool:
    """Return True if memory usage exceeds threshold using the active monitor."""
    return _monitor.check_pressure(threshold_percent)


def log_memory_snapshot(*, context: str = "") -> None:
    """Log memory snapshot using the active monitor."""
    _monitor.log_snapshot(context)


def log_system_info() -> None:
    """Log system CPU and memory information at startup."""
    log = get_logger("handwriting_ai")
    log.setLevel(20)
    cpu_logical = int(psutil.cpu_count(logical=True) or 0)
    cpu_physical_val = psutil.cpu_count(logical=False)
    cpu_physical = int(cpu_physical_val) if cpu_physical_val is not None else 0

    if _detect_cgroups_available():
        cgroup = _read_cgroup_usage()
        limit_mb = cgroup["limit_bytes"] // (1024 * 1024)
        log.info(
            "system_info "
            f"cpu_logical={cpu_logical} cpu_physical={cpu_physical} "
            f"cgroup_mem_limit_mb={limit_mb}"
        )
        return
    # Non-container path: use system memory metrics; propagate failures
    vm = psutil.virtual_memory()
    limit_mb = int(vm.total // (1024 * 1024))
    log.info(
        "system_info "
        f"cpu_logical={cpu_logical} cpu_physical={cpu_physical} "
        f"system_total_mb={limit_mb}"
    )


__all__ = [
    "CgroupMemoryBreakdown",
    "CgroupMemoryMonitor",
    "CgroupMemoryUsage",
    "MemoryMonitor",
    "MemorySnapshot",
    "ProcessMemory",
    "SystemMemoryMonitor",
    "check_memory_pressure",
    "get_memory_snapshot",
    "get_monitor",
    "is_cgroup_available",
    "log_memory_snapshot",
    "log_system_info",
]
