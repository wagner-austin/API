"""Where a runner unit's drop-ins live on a runner host."""

from __future__ import annotations

import pytest

from fleet.core.runner_unit_files import drop_in_path
from tests._runner_fixtures import an_install


def test_a_drop_in_lives_in_its_unit_s_directory() -> None:
    install = an_install("wagner-austin/MCPs")
    assert drop_in_path(install, "fleet-kill.conf") == (
        "/etc/systemd/system/actions.runner.wagner-austin-MCPs.lavender.service.d/fleet-kill.conf"
    )


def test_a_unit_name_that_cannot_be_embedded_is_refused() -> None:
    install = an_install()
    install["service"] = "bad'unit.service"
    with pytest.raises(ValueError, match="service"):
        drop_in_path(install, "fleet-kill.conf")
