from __future__ import annotations

from pathlib import Path

import handwriting_ai.training.resources as res
from handwriting_ai import _test_hooks


def test_detect_cpu_cores_invalid_falls_to_os(tmp_path: Path) -> None:
    cpu_max_file = tmp_path / "cpu.max"
    cpu_max_file.write_text("200000 0")  # period=0 defaults to 100_000

    _test_hooks.cgroup_cpu_max = cpu_max_file

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read

    # When period is 0, it defaults to 100_000, so we get 200000/100000 = 2 cores
    assert res._detect_cpu_cores() == 2


def test_detect_cpu_cores_zero_quota_falls_to_os(tmp_path: Path) -> None:
    cpu_max_file = tmp_path / "cpu.max"
    cpu_max_file.write_text("0 100000")  # quota <= 0 -> fall through

    _test_hooks.cgroup_cpu_max = cpu_max_file

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read
    _test_hooks.os_cpu_count = lambda: 3

    assert res._detect_cpu_cores() == 3


def test_detect_cpu_cores_max_falls_to_os(tmp_path: Path) -> None:
    cpu_max_file = tmp_path / "cpu.max"
    cpu_max_file.write_text("max")  # skip cgroup

    _test_hooks.cgroup_cpu_max = cpu_max_file

    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    _test_hooks.read_text_file = _read
    _test_hooks.os_cpu_count = lambda: 2

    assert res._detect_cpu_cores() == 2
