"""Restart-on-failure for the runner services, run for real.

The lines run under a real bash from a script file (as provision.sh delivers
them) with ``systemctl`` defined as a recording function and the one absolute
directory, ``/etc/systemd/system``, rewritten to the test's own directory:
the comparison, the write, the reload and the idempotence all run as
rendered.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.core import runner_recovery
from tests._host_bash import host_bash
from tests._runner_fixtures import an_install

#: The unit every case gives the policy.
_INSTALL = an_install("wagner-austin/MCPs")


class TestRendered:
    """What renders without running."""

    def test_an_unscriptable_unit_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            runner_recovery.render_recovery_lines({**_INSTALL, "service": "bad'unit"})
        assert str(refused.value) == (
            'service "bad\'unit" contains "\'", which cannot be embedded in a rendered '
            "script verbatim; rename the item rather than escaping it"
        )


class TestTheSystemdLinesRunForReal:
    """The drop-in lands once, byte for byte, and reloads systemd once."""

    def test_the_drop_in_is_written_once_and_reloaded_once(self, tmp_path: pathlib.Path) -> None:
        bash = host_bash()
        unit = _INSTALL["service"]
        lines = runner_recovery.render_recovery_lines(_INSTALL)
        body = "\n".join(lines).replace("/etc/systemd/system", "units")
        script = 'systemctl() { echo "$*" >> reloads.log; }\n' + body + "\n"
        (tmp_path / "recovery.sh").write_bytes(script.encode())
        outputs = []
        for _ in range(2):
            ran = subprocess.run(
                [bash, "recovery.sh"], cwd=tmp_path, capture_output=True, text=True, check=False
            )
            assert ran.returncode == 0, ran.stderr
            outputs.append(ran.stdout)
        assert outputs == [f"restart policy set for {unit}\n", ""]
        drop_in = tmp_path / "units" / f"{unit}.d" / runner_recovery.SYSTEMD_DROP_IN_NAME
        assert drop_in.read_bytes() == runner_recovery.SYSTEMD_DROP_IN.encode()
        assert (tmp_path / "reloads.log").read_text(encoding="utf-8") == "daemon-reload\n"
