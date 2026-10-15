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
from fleet.core.linux_capacity_probe import (
    CAPACITY_PROBE_BODY,
    CGROUPS_LINE,
    MEMINFO_LINE,
    RESERVATIONS_LINE,
    RESERVATIONS_PART,
)
from tests.test_dialect_linux import fields_of

GIB = 1073741824
CAP = 4 * GIB
#: diphtheria's MemAvailable at 22:1xZ, in kB as /proc/meminfo prints it.
AVAILABLE_KB = 16014192


def test_the_dialect_runs_the_body_after_its_prologue() -> None:
    assert CAPACITY_PROBE_SCRIPT.startswith(PROLOGUE + CAPACITY_PROBE_BODY)
    assert LinuxDialect().capacity_probe_script() == CAPACITY_PROBE_SCRIPT
    assert CAPACITY_PROBE_BODY.startswith(MEMINFO_LINE + CGROUPS_LINE)
    assert MEMINFO_LINE == "meminfo=/proc/meminfo\n"
    assert CGROUPS_LINE == "cgroups=/sys/fs/cgroup/system.slice\n"
    assert CAPACITY_PROBE_BODY.endswith(RESERVATIONS_PART)
    assert RESERVATIONS_PART.startswith(RESERVATIONS_LINE)
    assert RESERVATIONS_LINE == 'reservations="$HOME/.fleet/reservations"\n'


def _scope(root: pathlib.Path, name: str, limit: str, anon: int | None, shmem: int = 0) -> None:
    """Write one fake container cgroup.

    Args:
        root: The fake ``system.slice``.
        name: The scope's directory name.
        limit: What its ``memory.max`` reads.
        anon: Its ``memory.stat`` ``anon`` line, or None for a scope whose
            ``memory.stat`` is already gone.
        shmem: Its ``memory.stat`` ``shmem`` line, which the kernel also
            counts in ``file``.
    """
    scope = root / name
    scope.mkdir()
    (scope / "memory.max").write_text(f"{limit}\n", encoding="utf-8")
    if anon is not None:
        (scope / "memory.stat").write_text(
            f"anon {anon}\nfile {734003200 + shmem}\nkernel 1048576\nshmem {shmem}\n",
            encoding="utf-8",
        )


def _run(
    tmp_path: pathlib.Path,
    cgroups: pathlib.Path,
    held: pathlib.Path | None = None,
) -> dict[str, str]:
    """Run the probe with its roots pointed at fakes.

    Args:
        tmp_path: Where the fake meminfo and the script go.
        cgroups: The fake ``system.slice``.
        held: The fake reservations directory, or None for one that does
            not exist, as on every node nothing has reserved.

    Returns:
        The probe's key=value fields.
    """
    reservations = held if held is not None else tmp_path / "no-reservations"
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal:       32745488 kB\nMemFree:          362496 kB\n"
        f"MemAvailable:   {AVAILABLE_KB} kB\n",
        encoding="utf-8",
    )
    script = tmp_path / "probe.sh"
    script.write_text(
        CAPACITY_PROBE_SCRIPT.replace(MEMINFO_LINE, f"meminfo='{meminfo.as_posix()}'\n", 1)
        .replace(CGROUPS_LINE, f"cgroups='{cgroups.as_posix()}'\n", 1)
        .replace(RESERVATIONS_LINE, f"reservations='{reservations.as_posix()}'\n", 1),
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
        first = ["free_ram_gb", "promised_ram_gb", "free_disk_gb", "logical_cores"]
        assert list(fields)[: len(first)] == first

    def test_shared_memory_is_resident_and_is_not_charged_again(
        self, tmp_path: pathlib.Path
    ) -> None:
        """colossus, 2026-10-14 04:33Z: the search copy's shared buffers under its 8 GiB cap."""
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()
        anon, shmem = 96_636_928, 2_190_381_056
        _scope(cgroups, "docker-postgres-search.scope", str(8 * GIB), anon, shmem)
        _scope(cgroups, "docker-exited.scope", str(CAP), None)

        fields = _run(tmp_path, cgroups)

        promised = 8 * GIB - anon - shmem
        assert promised == 6_302_916_608
        assert fields["promised_ram_gb"] == f"{promised / GIB:.3f}"
        assert fields["promised_ram_gb"] == "5.870"

    def test_a_stat_with_no_anonymous_line_promises_nothing(self, tmp_path: pathlib.Path) -> None:
        """A cap whose ``memory.stat`` carries no ``anon`` line is not charged."""
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()
        scope = cgroups / "docker-tearing.scope"
        scope.mkdir()
        (scope / "memory.max").write_text(f"{CAP}\n", encoding="utf-8")
        (scope / "memory.stat").write_text("shmem 1048576\n", encoding="utf-8")

        fields = _run(tmp_path, cgroups)

        assert fields["promised_ram_gb"] == "0.000"

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
        assert fields["reserved_cores"] == "0"


def _reserve(held: pathlib.Path, name: str, text: str) -> None:
    """Write one fake reservation file.

    Args:
        held: The fake reservations directory.
        name: The file's name.
        text: Its whole content.
    """
    held.mkdir(exist_ok=True)
    (held / name).write_bytes(text.encode("utf-8"))


@pytest.mark.host_linux
class TestTheReservationsUnderSh:
    """MCPs board task 84a2b690: work the fleet does not run declares the cores it holds."""

    def test_a_live_reservation_is_charged_and_an_expired_one_is_not(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria, 2026-10-14: a 4-vCPU clean-VM smoke, beside one that ended."""
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()
        held = tmp_path / "reservations"
        _reserve(held, "vm.reservation", "holder=vm smoke\ncores=4\nuntil_unix=99999999999\n")
        _reserve(held, "old.reservation", "holder=old\r\ncores=2\r\nuntil_unix=1\r\n")
        _reserve(held, "build.reservation", "holder=build\r\ncores=3\r\nuntil_unix=99999999999\r\n")
        _reserve(held, "notes.txt", "cores=9\nuntil_unix=99999999999\n")

        fields = _run(tmp_path, cgroups, held)

        assert fields["reserved_cores"] == "7"
        # Read after the body's other fields; the dialect's CI-slice lines,
        # printed only on a node with a runners.slice, follow it.
        keys = list(fields)
        assert keys.index("reserved_cores") == keys.index("logical_cores") + 1

    def test_an_empty_directory_holds_nothing(self, tmp_path: pathlib.Path) -> None:
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()
        held = tmp_path / "reservations"
        held.mkdir()

        assert _run(tmp_path, cgroups, held)["reserved_cores"] == "0"

    @pytest.mark.parametrize(
        "text",
        [
            "holder=vm\ncores=four\nuntil_unix=99999999999\n",
            "holder=vm\nuntil_unix=99999999999\n",
            "holder=vm\ncores=4\n",
            "holder=vm\ncores=4\nuntil_unix=-1\n",
        ],
    )
    def test_a_file_it_cannot_read_is_named_and_ends_the_probe(
        self, tmp_path: pathlib.Path, text: str
    ) -> None:
        cgroups = tmp_path / "system.slice"
        cgroups.mkdir()
        held = tmp_path / "reservations"
        _reserve(held, "bad.reservation", text)

        fields = _run(tmp_path, cgroups, held)

        assert fields["reservation_unreadable"] == (held / "bad.reservation").as_posix()
        assert "reserved_cores" not in fields
        assert fields["logical_cores"].isdigit()
