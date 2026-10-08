"""Which manager installs what on which node, and the script that does it.

Split from ``test_toolchain.py`` by role when that file crossed the 600-line
ceiling: this half is about INSTALLING (the manager table, its preference
order, the rendered script and the explicit install command), the other about
probing and readiness. The fixtures they share are :mod:`tests._toolchain_fixtures`.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.contracts.tagged_tools import TAGGED_TOOLS
from fleet.contracts.toolchain import (
    PACKAGE_MANAGERS,
    PINNED_PYTHON,
    PYTHON_REGISTERED_GUARD,
    REQUIRED_PYTHON,
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
    LAVENDER_2026_09_23,
    LAVENDER_STORE_STUB,
    LOKI,
    SEDONA,
    SEDONA_2026_09_23,
    node,
)
from tests.conftest import FakeRun, failed, ok


class TestManagerSelection:
    def test_it_reports_only_the_managers_a_node_has(self) -> None:
        """MEASURED 2026-09-04: lavender had winget and no choco."""
        reports = toolchain.parse_probe("pip=yes=pip 24.0\nwinget=yes=v1.2\nchoco=no=\n")

        assert available_managers(reports) == ("pip", "winget")

    def test_the_interpreter_is_not_a_manager(self) -> None:
        """A Linux node reports its interpreter under ``python`` too, and the
        pip route is refused there (PEP 668); only a ``pip`` line that answered
        makes that manager available."""
        reports = toolchain.parse_probe("python=yes=Python 3.11.9\nwinget=yes=v1.2\n")

        assert available_managers(reports) == ("winget",)

    def test_preference_order_is_the_declared_one_not_report_order(self) -> None:
        reports = toolchain.parse_probe(
            "apt-get=yes=apt 2.8.3\nchoco=yes=2.7.4\nwinget=yes=v1.2\npipx=yes=1.4.3\n"
            "pip=yes=pip 24.0\n"
        )

        assert available_managers(reports) == PACKAGE_MANAGERS

    def test_a_node_with_no_manager_reports_none(self) -> None:
        """Not an error: it means nothing can be installed automatically."""
        reports = toolchain.parse_probe("git=no=\nmake=no=\n")

        assert available_managers(reports) == ()

    def test_the_first_available_manager_wins(self) -> None:
        assert install_command("git", ("pip", "winget", "choco")).startswith("winget")
        assert install_command("git", ("pip", "choco")).startswith("choco")
        assert install_command("git", ("pipx", "apt-get")) == "sudo apt-get install -y git"
        assert install_command("poetry", ("pipx", "apt-get")) == "pipx install poetry"
        assert install_command("poetry", ("pip", "winget")).startswith("python -m pip")

    def test_a_tool_this_package_never_installs_has_no_command(self) -> None:
        """tar ships with the platform; apt-get's python and node are not the
        versions the fleet runs, so a Linux node's are installed by hand."""
        assert install_command("tar", ("winget", "choco")) == ""
        assert install_command("python", ("pipx", "apt-get")) == ""
        assert install_command("node", ("pipx", "apt-get")) == ""

    def test_python_installs_the_pinned_python_org_build_at_user_scope(self) -> None:
        """THE DECISION MADE 2026-09-23: the version every node carried, at
        the path every node carried it, with no elevation."""
        by_winget = install_command("python", ("pip", "winget", "choco"))

        assert by_winget == (
            f"{PYTHON_REGISTERED_GUARD}winget install --id Python.Python.3.11 "
            f"--version {PINNED_PYTHON} -e --scope user --source winget --silent "
            "--accept-package-agreements --accept-source-agreements --disable-interactivity"
        )
        assert install_command("python", ("choco",)) == (
            f"{PYTHON_REGISTERED_GUARD}choco install python311 --version {PINNED_PYTHON} -y"
        )
        assert PINNED_PYTHON.startswith(REQUIRED_PYTHON)

    def test_both_python_installs_stop_where_the_version_is_registered(self) -> None:
        """LAVENDER, 2026-09-23: a second install of the same version moved the
        Actions runner's copy out of its tool cache."""
        assert PYTHON_REGISTERED_GUARD.startswith("if (Get-ItemProperty 'HKLM:")
        assert f"-like 'Python {PINNED_PYTHON} Core Interpreter*'" in PYTHON_REGISTERED_GUARD
        assert PYTHON_REGISTERED_GUARD.endswith("exit 1 }; ")

    def test_node_installs_the_managers_lts(self) -> None:
        assert install_command("node", ("winget", "choco")).startswith(
            "winget install --id OpenJS.NodeJS.LTS -e --source winget"
        )
        assert install_command("node", ("choco",)) == "choco install nodejs-lts -y"

    def test_ffmpeg_installs_from_each_platforms_manager(self) -> None:
        """2026-10-01, MCPs board task 512e7bf8: grandma-api's check converts
        real audio through ffmpeg, which sedona and serendipity lacked; winget
        offered Gyan.FFmpeg.Essentials 9.0.1 on both."""
        assert install_command("ffmpeg", ("pip", "winget", "choco")) == (
            "winget install --id Gyan.FFmpeg.Essentials -e --source winget --silent "
            "--accept-package-agreements --accept-source-agreements --disable-interactivity"
        )
        assert install_command("ffmpeg", ("choco",)) == "choco install ffmpeg -y"
        assert install_command("ffmpeg", ("pipx", "apt-get")) == "sudo apt-get install -y ffmpeg"

    def test_an_unknown_tool_has_no_command(self) -> None:
        assert install_command("kubectl", ("winget", "choco")) == ""


class TestInstall:
    def test_only_tools_with_a_command_are_installable(self) -> None:
        """tar carries none: it ships with the platform."""
        reports = (
            ToolReport(name="python", present=False, version=""),
            ToolReport(name="make", present=False, version=""),
            ToolReport(name="tar", present=False, version=""),
            ToolReport(name="choco", present=True, version="2.7.4"),
        )

        assert toolchain_install.installable(reports) == ("python", "make")

    def test_a_node_behind_the_store_alias_is_offered_the_real_python(self) -> None:
        """LAVENDER'S RUNNER, 2026-09-22/23: python resolved to the WindowsApps
        alias, the probe reads that as absent, and the command offered is the
        guarded winget install of the pinned python.org build. On lavender
        itself the guard would have stopped it, since the Actions tool cache
        had registered 3.11.9 there, which is the outcome that protects the
        runner. poetry follows through pip only once there is a python to run
        it, which is why a second bootstrap pass is the one that finishes a
        node."""
        reports = toolchain.parse_probe(LAVENDER_STORE_STUB)

        assert toolchain_install.installable(reports) == ("python",)
        body = toolchain_install.install_script(
            ("python",), available_managers(reports), platform=NodePlatform.WINDOWS
        )
        assert "Write-Output 'installing python'" in body
        assert f"--version {PINNED_PYTHON} -e --scope user" in body
        assert "choco" not in body

    def test_todays_nodes_need_nothing(self) -> None:
        for answer in (LAVENDER_2026_09_23, SEDONA_2026_09_23, DIPHTHERIA_2026_09_23):
            reports = toolchain.parse_probe(answer)
            assert toolchain_install.installable(reports) == ()
            assert missing(reports) == ()
            assert python_is_right(reports)

    def test_a_node_without_the_needed_manager_cannot_install_it(self) -> None:
        """A gap to report, not a failure to raise.

        make has winget, choco and apt-get commands and no pip one, so a node
        with only pip cannot have it installed automatically.
        """
        reports = (
            ToolReport(name="make", present=False, version=""),
            ToolReport(name="pip", present=True, version="pip 24.0"),
            ToolReport(name="winget", present=False, version=""),
            ToolReport(name="choco", present=False, version=""),
        )

        assert toolchain_install.installable(reports) == ()

    def test_the_install_script_echoes_each_step(self) -> None:
        """So a transcript says which command produced which failure."""
        body = toolchain_install.install_script(
            ("poetry", "make"), ("pip", "choco"), platform=NodePlatform.WINDOWS
        )

        assert "Write-Output 'installing poetry'" in body
        assert "python -m pip install --user poetry" in body
        assert "choco install make -y" in body

    def test_a_linux_node_gets_sh_echoes_and_its_own_managers(self) -> None:
        body = toolchain_install.install_script(
            ("poetry", "make"), ("pipx", "apt-get"), platform=NodePlatform.LINUX
        )

        assert body == (
            "printf '%s\\n' 'installing poetry'\npipx install poetry\n"
            "printf '%s\\n' 'installing make'\nsudo apt-get install -y make\n"
        )

    def test_the_same_tool_renders_a_different_command_per_node(self) -> None:
        """THE DEFECT THE MAPPING REPLACED.

        The first version hardcoded `choco install`, inferred from loki's
        make living under the chocolatey lib directory -- one node
        generalised to three. lavender has no choco at all, so that command
        would have failed there with choco's own 'not recognized'.
        """
        on_lavender = toolchain_install.install_script(
            ("make",), ("pip", "winget"), platform=NodePlatform.WINDOWS
        )
        on_loki = toolchain_install.install_script(
            ("make",), ("pip", "choco"), platform=NodePlatform.WINDOWS
        )

        assert "winget install --id GnuWin32.Make" in on_lavender
        assert "choco" not in on_lavender
        assert "choco install make -y" in on_loki
        assert "winget" not in on_loki

    def test_a_tool_with_no_command_for_these_managers_is_refused(self) -> None:
        """Silently omitting it would report an install covering less than
        it claimed, and the caller would re-probe to find the tool absent
        with no explanation."""
        with pytest.raises(ValueError, match="no install command"):
            toolchain_install.install_script(("make",), ("pip",), platform=NodePlatform.WINDOWS)

    def test_installing_runs_the_command_and_re_probes_to_verify_it(self) -> None:
        """An install that ran is not an install that worked."""
        runner = FakeRun([ok(""), ok("installing make"), ok(""), ok(SEDONA_2026_09_23)])
        _test_hooks.run = runner

        reports = toolchain.parse_probe(SEDONA)

        after = toolchain_install.install_missing(node(), reports, writer=WRITER)

        assert after == toolchain.parse_probe(SEDONA_2026_09_23)
        # sedona has BOTH managers, and winget is the declared preference.
        assert b"winget install --id GnuWin32.Make" in (runner.stdin[0] or b"")
        assert runner.calls[0][-1].endswith("C:/fleet/stage/fleet-install.ps1' -Encoding utf8\"")
        assert runner.calls[2][-1].endswith(
            "C:/fleet/stage/fleet-toolchain-fleet-bootstrap.ps1' -Encoding utf8\""
        )
        assert len(runner.calls) == 4

    def test_a_linux_node_is_installed_through_sh(self) -> None:
        runner = FakeRun([ok(""), ok("installing make"), ok(""), ok(_linux_probe(make=True))])
        _test_hooks.run = runner

        after = toolchain_install.install_missing(
            _linux_node(), toolchain.parse_probe(_linux_probe(make=False)), writer=WRITER
        )

        assert "make" in {report["name"] for report in after if report["present"]}
        assert runner.stdin[0] == (
            b"printf '%s\\n' 'installing make'\nsudo apt-get install -y make\n"
        )
        assert runner.calls[1][-2:] == ("/bin/sh", "/home/corvis/fleet/stage/fleet-install.sh")

    def test_a_node_with_nothing_installable_is_left_alone(self) -> None:
        """No ssh call at all, which is what the empty reply list asserts."""
        _test_hooks.run = FakeRun([])
        reports = toolchain.parse_probe(LOKI)

        assert toolchain_install.install_missing(node("loki"), reports, writer=WRITER) is reports

    def test_a_failing_install_that_landed_nothing_rolls_back_nothing(self) -> None:
        """Not softened, and the rollback probe finds nothing to remove."""
        runner = FakeRun([ok(""), failed(1, "choco: not found"), ok(""), ok(SEDONA)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            toolchain_install.install_missing(node(), toolchain.parse_probe(SEDONA), writer=WRITER)

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "choco: not found" in excinfo.value.message
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
        assert runner.stdin[6] == (
            b"Write-Output 'uninstalling poetry'\npython -m pip uninstall -y poetry\n"
        )
        assert runner.calls[6][-1].endswith("C:/fleet/stage/fleet-uninstall.ps1' -Encoding utf8\"")

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
            "remove by hand: python -m pip uninstall -y poetry"
        )


class TestUninstall:
    def test_each_tool_is_removed_by_the_manager_that_installed_it(self) -> None:
        assert uninstall_command("git", ("pip", "winget", "choco")).startswith("winget uninstall")
        assert uninstall_command("git", ("pip", "choco")) == "choco uninstall git -y"
        assert uninstall_command("poetry", ("pipx", "apt-get")) == "pipx uninstall poetry"
        assert uninstall_command("tar", ("winget", "choco")) == ""

    def test_every_install_has_an_uninstall_on_the_same_managers(self) -> None:
        """A manager with an install and no removal would leave a rollback
        nothing to run; the key sets must be the same for every row."""
        for tool in REQUIRED_TOOLS + TAGGED_TOOLS:
            assert sorted(tool["uninstall"]) == sorted(tool["install"]), tool["name"]

    def test_a_tool_with_no_removal_for_these_managers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no uninstall command"):
            toolchain_install.uninstall_script(("make",), ("pip",), platform=NodePlatform.WINDOWS)


#: The probe writer these tests install as, which names the probe scripts.
WRITER = "fleet-bootstrap"


def _half(*, poetry: bool) -> str:
    """A Windows node missing make and, unless ``poetry``, poetry.

    Pip installs poetry and winget make, so an install of both can land one.

    Args:
        poetry: Whether poetry answers present.

    Returns:
        The probe's output.
    """
    line = "poetry=yes=Poetry (version 2.4.2)\n" if poetry else "poetry=no=\n"
    return (
        f"python=yes=Python 3.11.9\n{line}git=yes=git version 2.55.0\nmake=no=\n"
        "tar=yes=bsdtar 3.8.8\nwinget=yes=v1.29.380\nchoco=no=\npip=yes=pip 24.0\n"
    )


def _linux_probe(*, make: bool) -> str:
    """A Linux node that has everything but, unless ``make``, make.

    Args:
        make: Whether make answers present.

    Returns:
        The probe's output.
    """
    line = "make=yes=GNU Make 4.3\n" if make else "make=no=\n"
    return (
        "python=yes=Python 3.11.9\npoetry=yes=Poetry (version 2.5.1)\n"
        f"git=yes=git version 2.43.0\n{line}tar=yes=tar (GNU tar) 1.35\n"
        "apt-get=yes=apt 2.8.3\npipx=yes=1.4.3\n"
    )


def _linux_node() -> NodeConfig:
    """Diphtheria's declaration, staging under its home.

    Returns:
        The node.
    """
    linux = node("diphtheria")
    linux["platform"] = NodePlatform.LINUX
    linux["stage_root"] = "/home/corvis/fleet/stage"
    return linux
