from __future__ import annotations

from pathlib import Path

import handwriting_ai.training.resources as res
from handwriting_ai import _test_hooks


def test_detect_cpu_cores_cgroup(tmp_path: Path) -> None:
    cpu_max_file = tmp_path / "cpu.max"
    cpu_max_file.write_text("200000 100000")  # 2 cores

    _test_hooks.cgroup_cpu_max = cpu_max_file

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read
    assert res._detect_cpu_cores() == 2


def test_detect_memory_limit(tmp_path: Path) -> None:
    mem_max_file = tmp_path / "memory.max"
    mem_max_file.write_text("1048576")

    _test_hooks.cgroup_mem_max = mem_max_file

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read
    assert res._detect_memory_limit_bytes() == 1048576
