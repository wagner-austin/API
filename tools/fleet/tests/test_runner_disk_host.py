"""A runner host's root within the ceiling its roster declares, read on it (board task 621902c3).

The audit's disk row (:mod:`fleet.core.runner_audit_disk`) reads ``df -BG``
of ``/`` against ``base.disk.ceiling_gb``, but fleet-runners runs that audit
from a session, and nothing the board can run again recorded it. On
2026-10-30 that row failed lavender at 190 GB against 150 while every
bounded area was inside its own bound. The ceiling stays the operator's 150
('Get it under 150'), under the 193 GB
:func:`~fleet.contracts.runner_base.disk_bounds_gb` gives lavender's seven
installs, so this case fails a root over it whatever its areas hold. It
takes the audit's reading on the host: ``tools/fleet-execution-runners``
runs it there as the fleet's own user, and its ``host_reading`` names the
figure, the ceiling and the bounds.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from fleet.contracts.runner_base import disk_bounds_gb
from fleet.contracts.runners import HostRunnerSpec
from tests._host import READING_PROPERTY, command_output


@pytest.mark.host_runner_units
def test_the_root_uses_no_more_than_the_roster_s_ceiling(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    header, figure = command_output(["df", "-BG", "--output=used", "/"]).splitlines()
    assert header.strip() == "Used"
    used = int(figure.strip().removesuffix("G"))
    disk = runner_host["base"]["disk"]
    bounds = disk_bounds_gb(disk, len(runner_host["installs"]))
    assert used <= disk["ceiling_gb"]
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: / uses {used} GB of its {disk['ceiling_gb']} GB ceiling; its "
        f"{len(runner_host['installs'])} install(s)' bounds allow {bounds} GB over an idle "
        f"{disk['baseline_gb']} GB",
    )
