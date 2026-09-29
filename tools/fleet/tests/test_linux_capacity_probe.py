"""The Linux capacity probe nets out what capped containers are promised (MCPs board task a282400d).

The text is pinned everywhere; the script is RUN under ``sh`` against a fake
``meminfo`` and cgroup tree shaped like diphtheria's on 2026-09-29 22:1xZ:
four containers capped at 4 GiB, the rest uncapped, and one that exited
between the listing and the read.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core.dialect_linux import CAPACITY_PROBE_SCRIPT, PROLOGUE, SH_INVOCATION, LinuxDialect
from fleet.core.linux_capacity_probe import CAPACITY_PROBE_BODY, CGROUPS_LINE, MEMINFO_LINE
from tests.test_dialect_linux import fields_of

GIB = 1073741824
CAP = 4 * GIB
#: diphtheria's MemAvailable at 22:1xZ, in kB as /proc/meminfo prints it.
AVAILABLE_KB = 16014192


def test_the_dialect_runs_the_body_after_its_prologue() -> None:
    assert CAPACITY_PROBE_SCRIPT == PROLOGUE + CAPACITY_PROBE_BODY
    assert LinuxDialect().capacity_probe_script() == CAPACITY_PROBE_SCRIPT
    assert CAPACITY_PROBE_BODY.startswith(MEMINFO_LINE + CGROUPS_LINE)
    assert MEMINFO_LINE == "meminfo=/proc/meminfo\n"
    assert CGROUPS_LINE == "cgroups=/sys/fs/cgroup/system.slice\n"


def _scope(root: pathlib.Path, name: str, limit: str, anon: int | None) -> None:
    """Write one fake container cgroup.

    Args:
        root: The fake ``system.slice``.
        name: The scope's directory name.
        limit: What its ``memory.max`` reads.
        anon: Its ``memory.stat`` ``anon`` line, or None for a scope whose
            ``memory.stat`` is already gone.
    """
    scope = root / name
    scope.mkdir()
    (scope / "memory.max").write_text(f"{limit}\n", encoding="utf-8")
    if anon is not None:
        (scope / "memory.stat").write_text(
            f"anon {anon}\nfile 734003200\nkernel 1048576\n", encoding="utf-8"
        )


def _run(tmp_path: pathlib.Path, cgroups: pathlib.Path) -> dict[str, str]:
    """Run the probe with its roots pointed at fakes.

    Args:
        tmp_path: Where the fake meminfo and the script go.
        cgroups: The fake ``system.slice``.

    Returns:
        The probe's key=value fields.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal:       32745488 kB\nMemFree:          362496 kB\n"
        f"MemAvailable:   {AVAILABLE_KB} kB\n",
        encoding="utf-8",
    )
    script = tmp_path / "probe.sh"
    script.write_text(
        CAPACITY_PROBE_SCRIPT.replace(MEMINFO_LINE, f"meminfo='{meminfo.as_posix()}'\n", 1).replace(
            CGROUPS_LINE, f"cgroups='{cgroups.as_posix()}'\n", 1
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)


@pytest.mark.host_linux
class TestTheProbeUnderSh:
    def test_capped_containers_are_charged_their_cap_less_their_anonymous_memory(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria, 2026-09-29: four capped workers, uncapped services beside them."""
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()
        anons = (1_300_000_000, 2_000_000_000, 2_800_000_000, 20_000_000)
        for index, anon in enumerate(anons):
            _scope(cgroups, f"docker-capped{index}.scope", str(CAP), anon)
        _scope(cgroups, "docker-uncapped.scope", "max", 2_500_000_000)
        _scope(cgroups, "docker-exited.scope", str(CAP), None)
        _scope(cgroups, "user-1002.slice", str(CAP), 0)

        fields = _run(tmp_path, cgroups)

        promised = 4 * CAP - sum(anons)
        assert promised == 11_059_869_184
        assert fields["promised_ram_gb"] == f"{promised / GIB:.3f}"
        assert fields["free_ram_gb"] == f"{(AVAILABLE_KB * 1024 - promised) / GIB:.3f}"
        assert fields["free_ram_gb"] == "4.972"
        assert list(fields) == ["free_ram_gb", "promised_ram_gb", "free_disk_gb", "logical_cores"]

    def test_a_node_with_no_capped_container_reads_its_whole_memavailable(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Every node but the stack host: nothing promised, the same reading as before."""
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()

        fields = _run(tmp_path, cgroups)

        assert fields["promised_ram_gb"] == "0.000"
        assert fields["free_ram_gb"] == f"{AVAILABLE_KB / 1048576:.3f}"
        assert fields["free_ram_gb"] == "15.272"
