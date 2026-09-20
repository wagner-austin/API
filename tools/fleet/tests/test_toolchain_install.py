"""Which manager installs what on which node, and the script that does it.

Split from ``test_toolchain.py`` by role when that file crossed the 600-line
ceiling: this half is about INSTALLING (the manager table, its preference
order, the rendered script and the explicit install command), the other about
probing and readiness. The fixtures they share are :mod:`tests._toolchain_fixtures`.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.toolchain import (
    PACKAGE_MANAGERS,
    ToolReport,
    available_managers,
    install_command,
)
from fleet.core import _test_hooks, toolchain
from tests._toolchain_fixtures import LOKI, SEDONA, node
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
        """python and tar are decisions about a machine, not packages."""
        assert install_command("python", ("winget", "choco")) == ""
        assert install_command("tar", ("winget", "choco")) == ""

    def test_an_unknown_tool_has_no_command(self) -> None:
        assert install_command("kubectl", ("winget", "choco")) == ""


class TestInstall:
    def test_only_tools_with_a_command_are_installable(self) -> None:
        """Python and tar carry none: which interpreter a machine should have
        is a decision, not a package."""
        reports = (
            ToolReport(name="python", present=False, version=""),
            ToolReport(name="make", present=False, version=""),
            ToolReport(name="tar", present=False, version=""),
            ToolReport(name="choco", present=True, version="2.7.4"),
        )

        assert toolchain.installable(reports) == ("make",)

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

        assert toolchain.installable(reports) == ()

    def test_the_install_script_echoes_each_step(self) -> None:
        """So a transcript says which command produced which failure."""
        body = toolchain.install_script(("poetry", "make"), ("pip", "choco"), platform="windows")

        assert "Write-Output 'installing poetry'" in body
        assert "python -m pip install --user poetry" in body
        assert "choco install make -y" in body

    def test_a_linux_node_gets_sh_echoes_and_its_own_managers(self) -> None:
        body = toolchain.install_script(("poetry", "make"), ("pipx", "apt-get"), platform="linux")

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
        on_lavender = toolchain.install_script(("make",), ("pip", "winget"), platform="windows")
        on_loki = toolchain.install_script(("make",), ("pip", "choco"), platform="windows")

        assert "winget install --id GnuWin32.Make" in on_lavender
        assert "choco" not in on_lavender
        assert "choco install make -y" in on_loki
        assert "winget" not in on_loki

    def test_a_tool_with_no_command_for_these_managers_is_refused(self) -> None:
        """Silently omitting it would report an install covering less than
        it claimed, and the caller would re-probe to find the tool absent
        with no explanation."""
        with pytest.raises(ValueError, match="no install command"):
            toolchain.install_script(("make",), ("pip",), platform="windows")

    def test_installing_runs_the_command_and_names_what_it_did(self) -> None:
        runner = FakeRun([ok(""), ok("installing make")])
        _test_hooks.run = runner

        installed = toolchain.install_missing(node(), toolchain.parse_probe(SEDONA))

        assert installed == ("make",)
        # sedona has BOTH managers, and winget is the declared preference.
        assert b"winget install --id GnuWin32.Make" in (runner.stdin[0] or b"")
        assert runner.calls[0][-1].endswith("C:/fleet/stage/fleet-install.ps1' -Encoding utf8\"")

    def test_a_linux_node_is_installed_through_sh(self) -> None:
        runner = FakeRun([ok(""), ok("installing make")])
        _test_hooks.run = runner
        linux = node("diphtheria")
        linux["platform"] = "linux"
        linux["stage_root"] = "/home/corvis/fleet/stage"
        reports = toolchain.parse_probe(
            "python=yes=Python 3.11.9\npoetry=yes=Poetry (version 2.5.1)\n"
            "git=yes=git version 2.43.0\nmake=no=\ntar=yes=tar (GNU tar) 1.35\n"
            "apt-get=yes=apt 2.8.3\npipx=yes=1.4.3\n"
        )

        assert toolchain.install_missing(linux, reports) == ("make",)
        assert runner.stdin[0] == (
            b"printf '%s\\n' 'installing make'\nsudo apt-get install -y make\n"
        )
        assert runner.calls[1][-2:] == ("/bin/sh", "/home/corvis/fleet/stage/fleet-install.sh")

    def test_a_node_with_nothing_installable_is_left_alone(self) -> None:
        """No ssh call at all, which is what the empty reply list asserts."""
        _test_hooks.run = FakeRun([])

        assert toolchain.install_missing(node("loki"), toolchain.parse_probe(LOKI)) == ()

    def test_a_failing_install_is_not_softened(self) -> None:
        """A half-installed node is worse than an untouched one: it looks ready."""
        _test_hooks.run = FakeRun([ok(""), failed(1, "choco: not found")])

        with pytest.raises(AppError) as excinfo:
            toolchain.install_missing(node(), toolchain.parse_probe(SEDONA))

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "choco: not found" in excinfo.value.message
