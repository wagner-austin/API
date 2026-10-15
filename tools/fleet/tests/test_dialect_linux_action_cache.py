"""The Linux capacity probe forwards a runner host's action cache counts (MCPs board task ce2b92e0).

A runner host's root timer writes four counts to a status file every
account can read, and the probe passes them on only while the file is fresh
and only as lines of their exact shape. The text is pinned here, and the
script is RUN under ``sh`` (``host_linux``, so on a Linux node by
``tools/fleet-execution-linux``) against a fake ``meminfo``, an empty
cgroup tree and a status file that is fresh, stale, absent, or holding a
line that would stand in for the probe's own memory reading.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

import pytest

from fleet.contracts.action_cache import STATUS_FRESH_MINUTES, STATUS_PATH
from fleet.core.dialect_linux import CAPACITY_PROBE_SCRIPT, SH_INVOCATION
from fleet.core.linux_capacity_probe import CGROUPS_LINE, MEMINFO_LINE, RESERVATIONS_LINE
from tests.test_dialect_linux import fields_of

#: What the fake meminfo's 16014192 kB MemAvailable reads as, with nothing capped.
FREE_RAM_GB = "15.272"

#: lavender's eight runners, all reading and served, as the timer writes them.
COUNTS = (
    "action_cache_runners=8\naction_cache_reading=8\naction_cache_served=8\naction_cache_missed=0\n"
)


def test_the_probe_passes_on_only_fresh_counts_of_the_exact_shape() -> None:
    lines = CAPACITY_PROBE_SCRIPT.splitlines()
    start = lines.index(f"c={STATUS_PATH}")

    assert lines[start:] == [
        "c=/var/lib/fleet-action-cache/status",
        f'if [ -n "$(find "$c" -mmin -{STATUS_FRESH_MINUTES} 2>/dev/null)" ]; then',
        "  grep -E '^action_cache_(runners|reading|served|missed)=[0-9]+$' \"$c\" || true",
        "fi",
    ]
    assert STATUS_FRESH_MINUTES == 5


def _run(tmp_path: pathlib.Path, status: str | None, age_minutes: int = 0) -> dict[str, str]:
    """Run the probe with its roots and the status file pointed at fakes.

    Args:
        tmp_path: Where the fakes and the script go.
        status: The status file's text, or None for no file.
        age_minutes: How long ago the file was last written.

    Returns:
        The probe's key=value fields.
    """
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   16014192 kB\n", encoding="utf-8")
    cgroups = tmp_path / "system.slice"
    cgroups.mkdir()
    path = tmp_path / "status"
    if status is not None:
        path.write_text(status, encoding="utf-8")
        written = time.time() - age_minutes * 60
        os.utime(path, (written, written))
    script = tmp_path / "probe.sh"
    script.write_text(
        CAPACITY_PROBE_SCRIPT.replace(MEMINFO_LINE, f"meminfo='{meminfo.as_posix()}'\n", 1)
        .replace(CGROUPS_LINE, f"cgroups='{cgroups.as_posix()}'\n", 1)
        .replace(RESERVATIONS_LINE, f"reservations='{(tmp_path / 'none').as_posix()}'\n", 1)
        .replace(f"c={STATUS_PATH}\n", f"c='{path.as_posix()}'\n", 1),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [*SH_INVOCATION, str(script)], capture_output=True, text=True, check=False, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    return fields_of(completed.stdout)


def _counts(fields: dict[str, str]) -> dict[str, str]:
    """The action cache fields among a probe's.

    Args:
        fields: Every field the probe printed.

    Returns:
        Those whose key begins ``action_cache_``.
    """
    return {key: value for key, value in fields.items() if key.startswith("action_cache_")}


@pytest.mark.host_linux
class TestTheProbeUnderSh:
    def test_a_fresh_file_s_counts_are_passed_on(self, tmp_path: pathlib.Path) -> None:
        fields = _run(tmp_path, COUNTS, age_minutes=1)

        assert _counts(fields) == {
            "action_cache_runners": "8",
            "action_cache_reading": "8",
            "action_cache_served": "8",
            "action_cache_missed": "0",
        }

    def test_a_stale_file_says_the_timer_stopped_and_passes_on_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        assert _counts(_run(tmp_path, COUNTS, age_minutes=STATUS_FRESH_MINUTES + 5)) == {}

    def test_a_host_with_no_file_passes_on_nothing_and_still_answers(
        self, tmp_path: pathlib.Path
    ) -> None:
        fields = _run(tmp_path, None)

        assert _counts(fields) == {}
        assert fields["free_ram_gb"] == FREE_RAM_GB

    def test_a_line_of_another_shape_cannot_stand_in_for_a_reading(
        self, tmp_path: pathlib.Path
    ) -> None:
        fields = _run(tmp_path, "free_ram_gb=999\naction_cache_runners=eight\n" + COUNTS)

        assert fields["free_ram_gb"] == FREE_RAM_GB
        assert _counts(fields)["action_cache_runners"] == "8"

    def test_a_file_with_no_count_line_passes_on_nothing_and_still_answers(
        self, tmp_path: pathlib.Path
    ) -> None:
        fields = _run(tmp_path, "garbage\n")

        assert _counts(fields) == {}
        assert fields["free_ram_gb"] == FREE_RAM_GB
