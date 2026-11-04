"""The Linux capacity probe reads free memory as ``MemAvailable`` (MCPs board task d111690a).

The text is pinned everywhere; the script is RUN under ``sh`` against a fake
``meminfo``, pressure files and reservations, shaped like diphtheria's. No
capped container's headroom is charged: a check yields memory to the stack
by its ``OOMScoreAdjust`` instead (:mod:`fleet.core.linux_capacity_probe`).
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from fleet.contracts.action_cache import PROBE_FIELDS
from fleet.core import probe
from fleet.core.dialect import CAPACITY_PROBE_SCRIPT, PROLOGUE, SH_INVOCATION
from fleet.core.linux_capacity_probe import (
    BATTERY_PART,
    CAPACITY_PROBE_BODY,
    CHECK_QUOTA_LINE,
    CHECK_QUOTA_PART,
    IO_PRESSURE_LINE,
    MEMINFO_LINE,
    POWER_SUPPLY_LINE,
    PRESSURE_LINE,
    PRESSURE_PART,
    RESERVATIONS_LINE,
    RESERVATIONS_PART,
)
from fleet.core.linux_gpu_probe import GPU_PROBE_PART, read_gpus
from tests.test_dialect_linux import fields_of

#: diphtheria's MemAvailable at 22:1xZ on 2026-09-29, in kB as /proc/meminfo prints it.
AVAILABLE_KB = 16014192

#: diphtheria's /proc/pressure/cpu at 14:4xZ on 2026-10-16.
PRESSURE = (
    "some avg10=11.17 avg60=9.33 avg300=9.33 total=343634208497\n"
    "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
)

#: diphtheria's /proc/pressure/io at 17:43Z on 2026-10-18, a database dump
#: and a VM install sharing its disk (MCPs board task 814c4555).
IO_PRESSURE = (
    "some avg10=73.86 avg60=68.81 avg300=52.24 total=289679902560\n"
    "full avg10=54.57 avg60=39.17 avg300=25.84 total=170774218710\n"
)

#: colossus's check slice and its parent at 14:2xZ on 2026-11-01, as
#: ``systemctl show ... --property=CPUQuotaPerSecUSec --value`` prints them.
QUOTA = "infinity\n\n8s"


def test_the_dialect_runs_the_body_after_its_prologue() -> None:
    assert CAPACITY_PROBE_SCRIPT.startswith(PROLOGUE + CAPACITY_PROBE_BODY)
    assert CAPACITY_PROBE_BODY.startswith(MEMINFO_LINE)
    assert MEMINFO_LINE == "meminfo=/proc/meminfo\n"
    assert CAPACITY_PROBE_BODY.endswith(
        PRESSURE_PART + CHECK_QUOTA_PART + BATTERY_PART + RESERVATIONS_PART + GPU_PROBE_PART
    )
    assert BATTERY_PART.startswith(POWER_SUPPLY_LINE)
    assert POWER_SUPPLY_LINE == "power_supply=/sys/class/power_supply\n"
    assert CHECK_QUOTA_PART.startswith(CHECK_QUOTA_LINE)
    assert CHECK_QUOTA_LINE == (
        "check_quota=$(systemctl show system-dispatch-checks.slice system-dispatch.slice"
        " --property=CPUQuotaPerSecUSec --value)\n"
    )
    assert PRESSURE_PART.startswith(PRESSURE_LINE)
    assert PRESSURE_LINE == "pressure=/proc/pressure/cpu\n"
    assert IO_PRESSURE_LINE in PRESSURE_PART
    assert IO_PRESSURE_LINE == "io_pressure=/proc/pressure/io\n"
    assert RESERVATIONS_PART.startswith(RESERVATIONS_LINE)
    assert RESERVATIONS_LINE == 'reservations="$HOME/.fleet/reservations"\n'


def test_the_body_reads_no_container_s_memory_cap() -> None:
    """The charge of every capped container's headroom is gone, not moved."""
    assert "memory.max" not in CAPACITY_PROBE_SCRIPT
    assert "memory.stat" not in CAPACITY_PROBE_SCRIPT
    assert "promised" not in CAPACITY_PROBE_SCRIPT


@pytest.mark.host_linux
def test_the_unaltered_probe_reports_this_host_s_numbers(tmp_path: pathlib.Path) -> None:
    """The script as the fleet sends it, against this machine's own roots."""
    script = tmp_path / "probe.sh"
    script.write_text(CAPACITY_PROBE_SCRIPT, encoding="utf-8")
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)], capture_output=True, text=True, check=False, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    fields = fields_of(completed.stdout)
    sliced = {"ci_slice_current_gb", "ci_slice_high_gb"} <= set(fields)
    # A runner host whose root timer counted its runners in the last five
    # minutes prints all four counts (MCPs board task ce2b92e0); every other
    # node prints none.
    counted = set(PROBE_FIELDS) & set(fields)
    assert counted in (set(), set(PROBE_FIELDS))

    # Every display device on this host, then their count (MCPs board task
    # b4d5ca0f); each line reads as a GPU, so none is a zero.
    assert read_gpus("this host", fields)[1] == ""
    devices = {f"gpu_{index}" for index in range(int(fields["gpu_count"]))}
    reported = {
        *probe.REQUIRED_FIELDS,
        "logical_cores",
        probe.CHECK_QUOTA_FIELD,
        probe.BATTERY_FIELD,
        "gpu_count",
    }
    assert set(fields) == reported | counted | devices | (
        {"ci_slice_current_gb", "ci_slice_high_gb"} if sliced else set()
    )
    # MemAvailable, which the kernel never reports below zero.
    assert re.fullmatch(r"\d+\.\d{3}", fields["free_ram_gb"])
    assert float(fields["free_disk_gb"]) > 0.0
    assert int(fields["logical_cores"]) >= 1
    # None on a CI runner, a VM smoke's cores on diphtheria (84a2b690).
    assert fields["reserved_cores"].isdigit()
    assert re.fullmatch(r"\d+\.\d{2}", fields["cpu_pressure_pct"])
    assert re.fullmatch(r"\d+\.\d{2}", fields["io_pressure_pct"])
    assert re.fullmatch(r"max|\d+\.\d{3}", fields[probe.CHECK_QUOTA_FIELD])
    assert fields[probe.BATTERY_FIELD] in probe.BATTERY_READINGS


def _probe(
    tmp_path: pathlib.Path,
    held: pathlib.Path | None,
    pressure: str | None,
    io_pressure: str | None,
    quota: str = QUOTA,
    supplies: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the probe with its roots pointed at fakes.

    Args:
        tmp_path: Where the fake meminfo, the fake pressure files and the
            script go.
        held: The fake reservations directory, or None for one that does
            not exist, as on every node nothing has reserved.
        pressure: What the fake ``/proc/pressure/cpu`` holds, or None for
            a kernel without one.
        io_pressure: What the fake ``/proc/pressure/io`` holds, or None for
            a kernel without one.
        quota: What ``systemctl show`` prints for the two check slices'
            ``CPUQuotaPerSecUSec``.
        supplies: The fake ``/sys/class/power_supply``, or None for one that
            does not exist, so the reading never depends on whether the node
            running the test is itself on its battery.

    Returns:
        The finished probe.
    """
    reservations = held if held is not None else tmp_path / "no-reservations"
    power_supply = supplies if supplies is not None else tmp_path / "no-power-supply"
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal:       32745488 kB\nMemFree:          362496 kB\n"
        f"MemAvailable:   {AVAILABLE_KB} kB\n",
        encoding="utf-8",
    )
    cpu = tmp_path / "pressure-cpu"
    if pressure is not None:
        cpu.write_text(pressure, encoding="utf-8")
    io = tmp_path / "pressure-io"
    if io_pressure is not None:
        io.write_text(io_pressure, encoding="utf-8")
    script = tmp_path / "probe.sh"
    script.write_text(
        CAPACITY_PROBE_SCRIPT.replace(MEMINFO_LINE, f"meminfo='{meminfo.as_posix()}'\n", 1)
        .replace(PRESSURE_LINE, f"pressure='{cpu.as_posix()}'\n", 1)
        .replace(IO_PRESSURE_LINE, f"io_pressure='{io.as_posix()}'\n", 1)
        .replace(CHECK_QUOTA_LINE, f"check_quota='{quota}'\n", 1)
        .replace(POWER_SUPPLY_LINE, f"power_supply='{power_supply.as_posix()}'\n", 1)
        .replace(RESERVATIONS_LINE, f"reservations='{reservations.as_posix()}'\n", 1),
        encoding="utf-8",
    )
    return subprocess.run(
        [*SH_INVOCATION, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _run(
    tmp_path: pathlib.Path,
    held: pathlib.Path | None = None,
    pressure: str = PRESSURE,
    io_pressure: str = IO_PRESSURE,
    quota: str = QUOTA,
    supplies: pathlib.Path | None = None,
) -> dict[str, str]:
    """Run the probe with its roots pointed at fakes, and read what it printed.

    Args:
        tmp_path: Where the fakes and the script go.
        held: The fake reservations directory, or None for none.
        pressure: What the fake ``/proc/pressure/cpu`` holds.
        io_pressure: What the fake ``/proc/pressure/io`` holds.
        quota: What ``systemctl show`` prints for the check slices' quotas.
        supplies: The fake ``/sys/class/power_supply``, or None for none.

    Returns:
        The probe's key=value fields.
    """
    completed = _probe(tmp_path, held, pressure, io_pressure, quota, supplies)
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)


@pytest.mark.host_linux
class TestTheProbeUnderSh:
    def test_free_memory_is_the_whole_memavailable_in_gibibytes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria, 2026-09-29: whatever its capped workers could still grow into."""
        fields = _run(tmp_path)

        assert fields["free_ram_gb"] == f"{AVAILABLE_KB / 1048576:.3f}"
        assert fields["free_ram_gb"] == "15.272"
        first = ["free_ram_gb", "free_disk_gb", "logical_cores"]
        assert list(fields)[: len(first)] == first
        assert fields["reserved_cores"] == "0"

    def test_a_meminfo_without_memavailable_prints_no_free_memory(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Which :mod:`fleet.core.probe` refuses as a missing field, never as an empty node."""
        # Write the script and its fakes, then take MemAvailable out of meminfo.
        assert _probe(tmp_path, None, PRESSURE, IO_PRESSURE).returncode == 0
        (tmp_path / "meminfo").write_text("MemTotal:       32745488 kB\n", encoding="utf-8")
        completed = subprocess.run(
            [*SH_INVOCATION, str(tmp_path / "probe.sh")],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        assert completed.returncode == 0, completed.stderr
        assert "free_ram_gb" not in fields_of(completed.stdout)

    def test_cpu_pressure_is_the_some_line_s_one_minute_average(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria, 2026-10-16 (MCPs board task 2ce5dc13): avg60, never avg10 or full."""
        fields = _run(tmp_path)

        assert fields["cpu_pressure_pct"] == "9.33"
        keys = list(fields)
        assert keys.index("cpu_pressure_pct") == keys.index("logical_cores") + 1

    def test_io_pressure_is_the_some_line_s_one_minute_average_after_the_cpu_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        """diphtheria, 2026-10-18 (MCPs board task 814c4555): avg60 of some, not full."""
        fields = _run(tmp_path)

        assert fields["io_pressure_pct"] == "68.81"
        keys = list(fields)
        assert keys.index("io_pressure_pct") == keys.index("cpu_pressure_pct") + 1

    def test_an_io_pressure_file_with_no_some_line_prints_the_field_empty(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Which the parser refuses as not a number, as it does the CPU reading."""
        fields = _run(tmp_path, io_pressure="full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")

        assert (fields["cpu_pressure_pct"], fields["io_pressure_pct"]) == ("9.33", "")

    def test_a_kernel_without_the_io_pressure_file_ends_the_probe(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``awk`` cannot open it, and ``set -e`` stops the script on that line."""
        completed = _probe(tmp_path, None, PRESSURE, None)

        assert completed.returncode != 0
        assert list(fields_of(completed.stdout)) == [
            "free_ram_gb",
            "free_disk_gb",
            "logical_cores",
            "cpu_pressure_pct",
        ]

    def test_a_pressure_file_with_no_some_line_prints_the_field_empty(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Which the parser refuses as not a number, so it never reads as calm."""
        fields = _run(tmp_path, pressure="full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")

        assert fields["cpu_pressure_pct"] == ""
        assert fields["reserved_cores"] == "0"

    @pytest.mark.parametrize(
        ("quota", "cores"),
        [
            # loki at 14:27Z on 2026-11-01: its check slice capped at runtime.
            ("1s\n\n14s", "1.000"),
            ("infinity\n\n8s", "8.000"),
            ("1.500000s\n\ninfinity", "1.500"),
            ("500ms\n\n4s", "0.500"),
            ("250000us\n\ninfinity", "0.250"),
            ("infinity\n\ninfinity", "max"),
            ("unlimited\n\n4s", ""),
            ("", ""),
        ],
    )
    def test_the_check_quota_is_the_smaller_slice_s_in_cores_after_the_pressures(
        self, tmp_path: pathlib.Path, quota: str, cores: str
    ) -> None:
        """MCPs board task d77b8c73: a value in no unit it reads prints the field
        empty, which the parser refuses, so it never reads as no quota."""
        fields = _run(tmp_path, quota=quota)

        assert fields["check_cpu_quota"] == cores
        keys = list(fields)
        assert keys.index("check_cpu_quota") == keys.index("io_pressure_pct") + 1

    def test_a_kernel_without_the_pressure_file_ends_the_probe(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``awk`` cannot open it, and ``set -e`` stops the script on that line."""
        completed = _probe(tmp_path, None, None, IO_PRESSURE)

        assert completed.returncode != 0
        assert list(fields_of(completed.stdout)) == [
            "free_ram_gb",
            "free_disk_gb",
            "logical_cores",
        ]


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
        held = tmp_path / "reservations"
        _reserve(held, "vm.reservation", "holder=vm smoke\ncores=4\nuntil_unix=99999999999\n")
        _reserve(held, "old.reservation", "holder=old\ncores=2\nuntil_unix=1\n")
        _reserve(held, "build.reservation", "holder=build\ncores=3\nuntil_unix=99999999999\n")
        _reserve(held, "notes.txt", "cores=9\nuntil_unix=99999999999\n")

        fields = _run(tmp_path, held)

        assert fields["reserved_cores"] == "7"
        # Read after the body's other fields; the dialect's CI-slice lines,
        # printed only on a node with a runners.slice, follow it.
        keys = list(fields)
        assert keys.index("reserved_cores") == keys.index("on_battery") + 1

    def test_an_empty_directory_holds_nothing(self, tmp_path: pathlib.Path) -> None:
        held = tmp_path / "reservations"
        held.mkdir()

        assert _run(tmp_path, held)["reserved_cores"] == "0"

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
        held = tmp_path / "reservations"
        _reserve(held, "bad.reservation", text)

        fields = _run(tmp_path, held)

        assert fields["reservation_unreadable"] == (held / "bad.reservation").as_posix()
        assert "reserved_cores" not in fields
        assert fields["logical_cores"].isdigit()


def _supply(
    root: pathlib.Path,
    name: str,
    kind: str | None,
    status: str | None,
    *,
    online: str | None = None,
    scope: str | None = None,
) -> None:
    """Write one fake ``/sys/class/power_supply`` entry.

    Args:
        root: The fake power-supply directory.
        name: The entry's name, as the kernel names it (``BAT0``, ``AC0``).
        kind: What its ``type`` file holds, or None for an entry without one.
        status: What its ``status`` file holds, or None for an entry without one.
        online: What its ``online`` file holds, or None for an entry without one.
        scope: What its ``scope`` file holds, or None for an entry without one.
    """
    entry = root / name
    entry.mkdir(parents=True)
    for field, value in (("type", kind), ("status", status), ("online", online), ("scope", scope)):
        if value is not None:
            (entry / field).write_text(f"{value}\n", encoding="utf-8")


@pytest.mark.host_linux
class TestTheBatteryUnderSh:
    """MCPs board task 3268afdf: a node running on its battery says so."""

    def test_loki_off_its_charger_reads_on_battery_after_the_quota(
        self, tmp_path: pathlib.Path
    ) -> None:
        """loki at 08:53Z on 2026-11-03: AC0 offline, a USB-C source idle, BAT0 discharging."""
        supplies = tmp_path / "power_supply"
        _supply(supplies, "AC0", "Mains", None, online="0")
        _supply(supplies, "ucsi-source-psy-USBC000:001", "USB", "Not charging", online="0")
        _supply(supplies, "BAT0", "Battery", "Discharging")

        fields = _run(tmp_path, supplies=supplies)

        assert fields["on_battery"] == "yes"
        keys = list(fields)
        assert keys.index("on_battery") == keys.index("check_cpu_quota") + 1

    @pytest.mark.parametrize("status", ["Charging", "Full", "Not charging", "Unknown"])
    def test_a_battery_that_is_not_discharging_reads_mains(
        self, tmp_path: pathlib.Path, status: str
    ) -> None:
        supplies = tmp_path / "power_supply"
        _supply(supplies, "AC0", "Mains", None)
        _supply(supplies, "BAT0", "Battery", status)

        assert _run(tmp_path, supplies=supplies)["on_battery"] == "no"

    def test_no_supply_an_empty_directory_and_mains_alone_all_read_mains(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A desktop lists no battery; a missing directory and an empty one agree."""
        empty = tmp_path / "empty"
        empty.mkdir()
        mains = tmp_path / "mains"
        _supply(mains, "AC", "Mains", None)

        assert _run(tmp_path)["on_battery"] == "no"
        assert _run(tmp_path, supplies=empty)["on_battery"] == "no"
        assert _run(tmp_path, supplies=mains)["on_battery"] == "no"

    def test_an_entry_without_a_type_or_a_battery_without_a_status_is_not_discharging(
        self, tmp_path: pathlib.Path
    ) -> None:
        supplies = tmp_path / "power_supply"
        _supply(supplies, "hid-0", None, "Discharging")
        _supply(supplies, "BAT1", "Battery", None)

        assert _run(tmp_path, supplies=supplies)["on_battery"] == "no"

    def test_one_discharging_battery_among_several_reads_on_battery(
        self, tmp_path: pathlib.Path
    ) -> None:
        supplies = tmp_path / "power_supply"
        _supply(supplies, "BAT0", "Battery", "Full")
        _supply(supplies, "BAT1", "Battery", "Discharging")

        assert _run(tmp_path, supplies=supplies)["on_battery"] == "yes"

    def test_a_battery_discharging_beside_an_online_adapter_reads_mains(
        self, tmp_path: pathlib.Path
    ) -> None:
        """lenovold at 2026-11-04T03:06Z: its firmware's conservation mode held BAT0
        at 93 percent reading Discharging while ADP0 read online 1, on its charger."""
        supplies = tmp_path / "power_supply"
        _supply(supplies, "ADP0", "Mains", None, online="1")
        _supply(supplies, "BAT0", "Battery", "Discharging")

        assert _run(tmp_path, supplies=supplies)["on_battery"] == "no"

    def test_a_usb_c_charger_the_machine_sinks_from_reads_mains(
        self, tmp_path: pathlib.Path
    ) -> None:
        """loki on its USB-C charger with AC0 offline: the port's supply reads online 1."""
        supplies = tmp_path / "power_supply"
        _supply(supplies, "AC0", "Mains", None, online="0")
        _supply(supplies, "ucsi-source-psy-USBC000:002", "USB", "Charging", online="1")
        _supply(supplies, "BAT0", "Battery", "Discharging")

        assert _run(tmp_path, supplies=supplies)["on_battery"] == "no"

    def test_a_peripheral_s_discharging_battery_is_not_the_machine_s(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A desktop with a wireless mouse: its battery's scope is Device, and a
        device-scoped supply reading online does not stand for the machine's mains."""
        mouse = tmp_path / "mouse"
        _supply(mouse, "hidpp_battery_0", "Battery", "Discharging", scope="Device")
        laptop = tmp_path / "laptop"
        _supply(laptop, "hid-dock", "USB", None, online="1", scope="Device")
        _supply(laptop, "BAT0", "Battery", "Discharging", scope="System")

        assert _run(tmp_path, supplies=mouse)["on_battery"] == "no"
        assert _run(tmp_path, supplies=laptop)["on_battery"] == "yes"
