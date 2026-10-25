"""Which manager installs what on which node, and the script that does it.

Split from ``test_toolchain.py`` by role when that file crossed the 600-line
ceiling: this half is about INSTALLING (the manager table, its preference
order, the rendered script and the explicit install command), the other about
probing and readiness. The fixtures they share are :mod:`tests._toolchain_fixtures`.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.tagged_tools import TAGGED_TOOLS
from fleet.contracts.toolchain import (
    PACKAGE_MANAGERS,
    REQUIRED_TOOLS,
    ToolReport,
    available_managers,
    install_command,
    missing,
    python_is_right,
    uninstall_command,
)
from fleet.core import _test_hooks, toolchain, toolchain_install
from tests._toolchain_fixtures import (
    DIPHTHERIA_2026_09_23,
    NO_MAKE,
    NO_PYTHON,
    node,
)
from tests.conftest import FakeRun, failed, ok


class TestManagerSelection:
    def test_it_reports_only_the_managers_a_node_has(self) -> None:
        """diphtheria, 2026-09-23: apt-get and pipx, and no pip line."""
        reports = toolchain.parse_probe(DIPHTHERIA_2026_09_23)

        assert available_managers(reports) == ("pipx", "apt-get")

    def test_the_interpreter_is_not_a_manager(self) -> None:
        """A node reports its interpreter under ``python``, and the pip route
        is refused there (PEP 668); only a ``pip`` line that answered makes
        that manager available."""
        reports = toolchain.parse_probe("python=yes=Python 3.11.15\napt-get=yes=apt 2.8.3\n")

        assert available_managers(reports) == ("apt-get",)

    def test_preference_order_is_the_declared_one_not_report_order(self) -> None:
        reports = toolchain.parse_probe("apt-get=yes=apt 2.8.3\npipx=yes=1.4.3\npip=yes=pip 24.0\n")

        assert available_managers(reports) == PACKAGE_MANAGERS == ("pip", "pipx", "apt-get")

    def test_the_managers_that_left_with_windows_are_not_managers(self) -> None:
        """A probe line for winget or choco names nothing this package
        installs through any more (board task e705f08c)."""
        reports = (
            ToolReport(name="winget", present=True, version="v1.2"),
            ToolReport(name="choco", present=True, version="2.7.4"),
        )

        assert available_managers(reports) == ()

    def test_a_node_with_no_manager_reports_none(self) -> None:
        """Not an error: it means nothing can be installed automatically."""
        reports = toolchain.parse_probe("git=no=\nmake=no=\n")

        assert available_managers(reports) == ()

    def test_the_first_available_manager_wins(self) -> None:
        assert install_command("git", ("pipx", "apt-get")) == "sudo apt-get install -y git"
        assert install_command("poetry", ("pipx", "apt-get")) == "pipx install poetry==2.4.2"
        assert install_command("poetry", ("pip", "pipx")) == (
            "python -m pip install --user poetry==2.4.2"
        )

    def test_a_tool_this_package_never_installs_has_no_command(self) -> None:
        """tar ships with the system; apt-get's python and node are not the
        versions the fleet runs, so a node's are installed by hand."""
        assert install_command("tar", ("pipx", "apt-get")) == ""
        assert install_command("python", ("pip", "pipx", "apt-get")) == ""
        assert install_command("node", ("pip", "pipx", "apt-get")) == ""

    def test_ffmpeg_installs_from_apt_get(self) -> None:
        """2026-10-01, MCPs board task 512e7bf8: grandma-api's check converts
        real audio through ffmpeg."""
        assert install_command("ffmpeg", ("pipx", "apt-get")) == "sudo apt-get install -y ffmpeg"
        assert install_command("ffmpeg", ("pip", "pipx")) == ""

    def test_an_unknown_tool_has_no_command(self) -> None:
        assert install_command("kubectl", ("pip", "pipx", "apt-get")) == ""


class TestInstall:
    def test_only_tools_with_a_command_are_installable(self) -> None:
        """tar and python carry none."""
        reports = (
            ToolReport(name="python", present=False, version=""),
            ToolReport(name="make", present=False, version=""),
            ToolReport(name="tar", present=False, version=""),
            ToolReport(name="apt-get", present=True, version="apt 2.8.3"),
        )

        assert toolchain_install.installable(reports) == ("make",)

    def test_a_node_without_its_interpreter_is_offered_nothing_for_it(self) -> None:
        """python carries no install command, and with it gone pipx answers
        absent too, so poetry has no manager to come from: the gap is closed
        by hand, which :func:`installable` reports by naming nothing."""
        reports = toolchain.parse_probe(NO_PYTHON)

        assert set(missing(reports)) >= {"python", "poetry"}
        assert toolchain_install.installable(reports) == ()

    def test_todays_node_needs_nothing(self) -> None:
        reports = toolchain.parse_probe(DIPHTHERIA_2026_09_23)

        assert toolchain_install.installable(reports) == ()
        assert missing(reports) == ()
        assert python_is_right(reports)

    def test_a_node_without_the_needed_manager_cannot_install_it(self) -> None:
        """A gap to report, not a failure to raise.

        make has an apt-get command and no pip one, so a node with only pip
        cannot have it installed automatically.
        """
        reports = (
            ToolReport(name="make", present=False, version=""),
            ToolReport(name="pip", present=True, version="pip 24.0"),
            ToolReport(name="apt-get", present=False, version=""),
        )

        assert toolchain_install.installable(reports) == ()

    def test_the_install_script_echoes_each_step_in_sh(self) -> None:
        """So a transcript says which command produced which failure."""
        body = toolchain_install.install_script(("poetry", "make"), ("pipx", "apt-get"))

        assert body == (
            "printf '%s\\n' 'installing poetry'\npipx install poetry==2.4.2\n"
            "printf '%s\\n' 'installing make'\nsudo apt-get install -y make\n"
        )

    def test_the_same_tool_renders_the_command_of_the_node_s_own_manager(self) -> None:
        """THE DEFECT THE MAPPING REPLACED: the first version hardcoded one
        manager's install, inferred from one node, and would have failed
        with 'not found' on a node that lacked it."""
        through_pip = toolchain_install.install_script(("poetry",), ("pip", "apt-get"))
        through_pipx = toolchain_install.install_script(("poetry",), ("pipx", "apt-get"))

        assert "python -m pip install --user poetry==2.4.2" in through_pip
        assert "pipx" not in through_pip
        assert "pipx install poetry==2.4.2" in through_pipx
        assert "python -m pip" not in through_pipx

    def test_a_tool_with_no_command_for_these_managers_is_refused(self) -> None:
        """Silently omitting it would report an install covering less than
        it claimed, and the caller would re-probe to find the tool absent
        with no explanation."""
        with pytest.raises(ValueError, match="no install command"):
            toolchain_install.install_script(("make",), ("pip",))

    def test_installing_runs_the_command_and_re_probes_to_verify_it(self) -> None:
        """An install that ran is not an install that worked."""
        runner = FakeRun([ok(""), ok("installing make"), ok(""), ok(DIPHTHERIA_2026_09_23)])
        _test_hooks.run = runner

        after = toolchain_install.install_missing(
            node("diphtheria"), toolchain.parse_probe(NO_MAKE), writer=WRITER
        )

        assert after == toolchain.parse_probe(DIPHTHERIA_2026_09_23)
        assert runner.stdin[0] == (
            b"printf '%s\\n' 'installing make'\nsudo apt-get install -y make\n"
        )
        assert runner.calls[1][-2:] == ("/bin/sh", "/home/corvis/fleet/stage/fleet-install.sh")
        assert runner.calls[3][-1] == (
            "/home/corvis/fleet/stage/fleet-toolchain-fleet-bootstrap.sh"
        )
        assert len(runner.calls) == 4

    def test_a_node_with_nothing_installable_is_left_alone(self) -> None:
        """No ssh call at all, which is what the empty reply list asserts."""
        _test_hooks.run = FakeRun([])
        reports = toolchain.parse_probe(DIPHTHERIA_2026_09_23)

        assert toolchain_install.install_missing(node(), reports, writer=WRITER) is reports

    def test_a_failing_install_that_landed_nothing_rolls_back_nothing(self) -> None:
        """Not softened, and the rollback probe finds nothing to remove."""
        runner = FakeRun(
            [ok(""), failed(100, "E: Unable to locate package make"), ok(""), ok(NO_MAKE)]
        )
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            toolchain_install.install_missing(node(), toolchain.parse_probe(NO_MAKE), writer=WRITER)

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "E: Unable to locate package make" in excinfo.value.message
        assert excinfo.value.message.endswith("; rolled back nothing: none of them had landed")
        assert len(runner.calls) == 4

    def test_an_install_that_left_a_tool_absent_removes_what_it_landed(self) -> None:
        """A half-installed node looks closer to ready than it is."""
        runner = FakeRun(
            [
                *(ok(""), ok("installing poetry")),
                *(ok(""), ok(_half(poetry=True))),
                *(ok(""), ok(_half(poetry=True))),
                *(ok(""), ok("uninstalling poetry")),
                *(ok(""), ok(_half(poetry=False))),
            ]
        )
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            toolchain_install.install_missing(
                node(), toolchain.parse_probe(_half(poetry=False)), writer=WRITER
            )

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert excinfo.value.message == (
            "lavender: installing poetry, make exited 0 but the probe still finds make absent; "
            "rolled back poetry"
        )
        assert runner.stdin[6] == (b"printf '%s\\n' 'uninstalling poetry'\npipx uninstall poetry\n")
        assert runner.calls[7][-2:] == ("/bin/sh", "/home/corvis/fleet/stage/fleet-uninstall.sh")

    def test_a_rollback_that_leaves_a_tool_names_the_command_to_remove_it(self) -> None:
        _test_hooks.run = FakeRun(
            [
                *(ok(""), ok("installing poetry")),
                *(ok(""), ok(_half(poetry=True))),
                *(ok(""), ok(_half(poetry=True))),
                *(ok(""), ok("uninstalling poetry")),
                *(ok(""), ok(_half(poetry=True))),
            ]
        )

        with pytest.raises(AppError) as excinfo:
            toolchain_install.install_missing(
                node(), toolchain.parse_probe(_half(poetry=False)), writer=WRITER
            )

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert excinfo.value.message == (
            "lavender: rolling back an unfinished install left poetry installed; "
            "remove by hand: pipx uninstall poetry"
        )


class TestUninstall:
    def test_each_tool_is_removed_by_the_manager_that_installed_it(self) -> None:
        assert uninstall_command("git", ("pipx", "apt-get")) == "sudo apt-get remove -y git"
        assert uninstall_command("poetry", ("pipx", "apt-get")) == "pipx uninstall poetry"
        assert uninstall_command("poetry", ("pip",)) == "python -m pip uninstall -y poetry"
        assert uninstall_command("tar", ("pipx", "apt-get")) == ""

    def test_every_install_has_an_uninstall_on_the_same_managers(self) -> None:
        """A manager with an install and no removal would leave a rollback
        nothing to run; the key sets must be the same for every row."""
        for tool in REQUIRED_TOOLS + TAGGED_TOOLS:
            assert sorted(tool["uninstall"]) == sorted(tool["install"]), tool["name"]

    def test_every_install_names_a_manager_the_fleet_has(self) -> None:
        """No row keeps a command for a manager no node carries any more."""
        for tool in REQUIRED_TOOLS + TAGGED_TOOLS:
            assert set(tool["install"]) <= set(PACKAGE_MANAGERS), tool["name"]

    def test_a_tool_with_no_removal_for_these_managers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no uninstall command"):
            toolchain_install.uninstall_script(("make",), ("pip",))


#: The probe writer these tests install as, which names the probe scripts.
WRITER = "fleet-bootstrap"


def _half(*, poetry: bool) -> str:
    """DERIVED from :data:`DIPHTHERIA_2026_09_23`: make absent and, unless
    ``poetry``, poetry too.

    pipx installs poetry and apt-get make, so an install of both can land one.

    Args:
        poetry: Whether poetry answers present.

    Returns:
        The probe's output.
    """
    answer = NO_MAKE
    if poetry:
        return answer
    return answer.replace("poetry=yes=Poetry (version 2.5.1)\n", "poetry=no=\n")
