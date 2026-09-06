"""Whether a node can run a build, and the command that says so.

THE FIXTURES ARE THE REAL MEASUREMENT. ``sedona``, ``lavender`` and ``loki``
answered these exact shapes on 2026-09-04, and one node of the three could
have run a ``make check``. A test written against an invented "node with
everything" would have proved the happy path and nothing about the fleet that
actually exists.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str, load_json_str

from fleet.cli import _config, bootstrap
from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig
from fleet.contracts.toolchain import (
    PACKAGE_MANAGERS,
    REQUIRED_TOOLS,
    ToolReport,
    available_managers,
    decode_tool_report,
    describe_gap,
    encode_tool_report,
    install_command,
    missing,
    python_is_right,
    version_number,
)
from fleet.core import _test_hooks, toolchain
from tests.conftest import FakeRun, failed, ok

#: What loki answered: everything present, poetry on the wrong Python, and
#: CHOCO BUT NO WINGET -- which is why the same missing tool renders a
#: different install command here than on lavender.
_LOKI = (
    "python=yes=Python 3.11.9\n"
    "poetry=yes=Poetry (version 2.1.3)\n"
    "git=yes=git version 2.50.1.windows.1\n"
    "make=yes=GNU Make 4.4.1\n"
    "tar=yes=bsdtar 3.7.2\n"
    "winget=no=\n"
    "choco=yes=2.5.0\n"
)

#: What lavender answered: no poetry, no git, no make, WINGET BUT NO CHOCO.
_LAVENDER = (
    "python=yes=Python 3.11.9\n"
    "poetry=no=\n"
    "git=no=\n"
    "make=no=\n"
    "tar=yes=bsdtar 3.7.2\n"
    "winget=yes=v1.11.400\n"
    "choco=no=\n"
)

#: What sedona answered: only make absent, and BOTH managers present.
_SEDONA = (
    "python=yes=Python 3.11.9\n"
    "poetry=yes=Poetry (version 2.2.1)\n"
    "git=yes=git version 2.43.0.windows.1\n"
    "make=no=\n"
    "tar=yes=bsdtar 3.7.2\n"
    "winget=yes=v1.11.400\n"
    "choco=yes=2.6.0\n"
)

#: A node carrying the wrong interpreter, which no probed node did.
_WRONG_PYTHON = _LOKI.replace("Python 3.11.9", "Python 3.12.4")


def _node(host: str = "lavender") -> NodeConfig:
    """Build a node declaration.

    Args:
        host: SSH alias.

    Returns:
        The node.
    """
    return NodeConfig(
        host=host,
        stage_root="C:/fleet/stage",
        logical_cores=16,
        ram_gb=32.0,
        gpu=None,
        enabled=True,
        budget=NodeBudget(
            reserved_cores=2,
            reserved_ram_gb=4.0,
            worker_ram_gb=1.1,
            max_concurrent_runs=2,
            max_disk_gb=20.0,
        ),
    )


def _workspace_document() -> JSONObject:
    """Build a two-node workspace as JSON.

    Returns:
        The document, ready to serialise.
    """
    budget: JSONObject = {
        "reserved_cores": 2,
        "reserved_ram_gb": 4.0,
        "worker_ram_gb": 1.1,
        "max_concurrent_runs": 2,
        "max_disk_gb": 20.0,
    }
    node: JSONObject = {
        "host": "lavender",
        "stage_root": "C:/fleet/stage",
        "logical_cores": 16,
        "ram_gb": 32.0,
        "gpu": None,
        "enabled": True,
        "budget": budget,
    }
    return {
        "nodes": {"lavender": node, "loki": {**node, "host": "loki"}},
        "projects": {
            "libs/demo": {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
            }
        },
        "ledger": "ledger.jsonl",
        "feed": "feed.jsonl",
        "leases": "leases.json",
    }


@pytest.fixture(name="config_path")
def _config_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a workspace document.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the written document.
    """
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(_workspace_document()), encoding="utf-8")
    return path


class TestParseProbe:
    def test_it_reads_every_tool_a_node_reports(self) -> None:
        reports = toolchain.parse_probe(_LOKI)

        assert [report["name"] for report in reports] == [
            "python",
            "poetry",
            "git",
            "make",
            "tar",
            "winget",
            "choco",
        ]
        assert reports[1]["version"] == "Poetry (version 2.1.3)"

    def test_an_absent_tool_carries_no_version(self) -> None:
        reports = toolchain.parse_probe(_LAVENDER)

        absent = [report for report in reports if report["name"] == "make"]
        assert absent == [ToolReport(name="make", present=False, version="")]

    def test_a_line_that_is_not_a_tool_is_ignored(self) -> None:
        """PowerShell writes warnings to the same stream."""
        reports = toolchain.parse_probe("WARNING: something\n" + _LOKI)

        assert len(reports) == 7

    def test_an_unrecognisable_answer_is_a_probe_that_did_not_run(self) -> None:
        """Reporting five absent tools would send the reader to install
        five things that are already there."""
        with pytest.raises(AppError) as excinfo:
            toolchain.parse_probe("The term 'foreach' is not recognized")

        assert excinfo.value.code is FleetErrorCode.NODE_TOOL_MISSING
        assert "never asked" in excinfo.value.message


class TestReadiness:
    def test_loki_is_ready(self) -> None:
        assert missing(toolchain.parse_probe(_LOKI)) == ()
        assert python_is_right(toolchain.parse_probe(_LOKI))

    def test_lavender_is_missing_three_tools(self) -> None:
        assert missing(toolchain.parse_probe(_LAVENDER)) == ("poetry", "git", "make")

    def test_sedona_is_missing_only_make(self) -> None:
        """MEASURED: make was on one node of three."""
        assert missing(toolchain.parse_probe(_SEDONA)) == ("make",)

    def test_the_wrong_python_is_not_ready_even_with_every_tool(self) -> None:
        reports = toolchain.parse_probe(_WRONG_PYTHON)

        assert missing(reports) == ()
        assert not python_is_right(reports)

    def test_a_node_reporting_no_python_at_all_is_not_ready(self) -> None:
        assert not python_is_right(())


class TestVersionNumber:
    def test_it_takes_the_number_out_of_what_a_tool_printed(self) -> None:
        """THE BUG THIS FUNCTION FIXES.

        Comparing the whole string against '3.11' reported every node as
        carrying the wrong interpreter, including the three that carry the
        right one, because `python --version` prints 'Python 3.11.9'.
        """
        assert version_number("Python 3.11.9") == "3.11.9"

    def test_it_unwraps_a_parenthesised_version(self) -> None:
        assert version_number("Poetry (version 2.1.3)") == "2.1.3"

    def test_a_trailing_platform_suffix_survives(self) -> None:
        assert version_number("git version 2.50.1.windows.1") == "2.50.1.windows.1"

    def test_a_bare_number_is_unchanged(self) -> None:
        assert version_number("3.11.9") == "3.11.9"

    def test_nothing_reported_is_nothing(self) -> None:
        assert version_number("") == ""


class TestDescribeGap:
    def test_a_ready_node_says_so(self) -> None:
        assert describe_gap("loki", toolchain.parse_probe(_LOKI)) == "loki: ready"

    def test_it_names_the_install_command(self) -> None:
        """The reader's next question is always how to fix it."""
        described = describe_gap("lavender", toolchain.parse_probe(_LAVENDER))

        assert "poetry (python -m pip install --user poetry)" in described
        # lavender has winget and no choco, so the command it is told to run
        # is the winget one. loki, missing the same tool, would be told choco.
        assert "make (winget install --id GnuWin32.Make" in described
        assert "choco" not in described

    def test_a_tool_with_no_automatic_install_says_so(self) -> None:
        described = describe_gap("odd", (ToolReport(name="tar", present=False, version=""),))

        assert "tar (install by hand)" in described

    def test_the_wrong_python_is_named_with_what_was_found(self) -> None:
        described = describe_gap("loki", toolchain.parse_probe(_WRONG_PYTHON))

        assert "python 3.11 (found Python 3.12.4)" in described

    def test_an_unreported_python_reads_as_unknown(self) -> None:
        described = describe_gap("odd", (ToolReport(name="git", present=True, version="x"),))

        assert "found unknown" in described


class TestRequireReady:
    def test_a_ready_node_passes(self) -> None:
        toolchain.require_ready("loki", _node("loki"), toolchain.parse_probe(_LOKI))

    def test_a_missing_tool_names_every_one_and_why(self) -> None:
        with pytest.raises(AppError) as excinfo:
            toolchain.require_ready("lavender", _node(), toolchain.parse_probe(_LAVENDER))

        assert excinfo.value.code is FleetErrorCode.NODE_TOOL_MISSING
        assert "make check is the entry point" in excinfo.value.message
        assert "winget install --id GnuWin32.Make" in excinfo.value.message

    def test_the_wrong_python_is_its_own_code(self) -> None:
        """The fixes differ: a package manager versus a decision."""
        with pytest.raises(AppError) as excinfo:
            toolchain.require_ready("loki", _node("loki"), toolchain.parse_probe(_WRONG_PYTHON))

        assert excinfo.value.code is FleetErrorCode.NODE_PYTHON_MISMATCH
        assert "3.12.4" in excinfo.value.message

    def test_a_node_that_reported_no_python_says_unknown(self) -> None:
        reports = tuple(
            report for report in toolchain.parse_probe(_LOKI) if report["name"] != "python"
        )

        with pytest.raises(AppError, match="unknown"):
            toolchain.require_ready("odd", _node(), reports)


class TestManagerSelection:
    def test_it_reports_only_the_managers_a_node_has(self) -> None:
        """MEASURED 2026-09-04: lavender had winget and no choco."""
        reports = toolchain.parse_probe("python=yes=Python 3.11.9\nwinget=yes=v1.2\nchoco=no=\n")

        assert available_managers(reports) == ("python", "winget")

    def test_preference_order_is_the_declared_one_not_report_order(self) -> None:
        reports = toolchain.parse_probe(
            "choco=yes=2.7.4\nwinget=yes=v1.2\npython=yes=Python 3.11.9\n"
        )

        assert available_managers(reports) == PACKAGE_MANAGERS

    def test_a_node_with_no_manager_reports_none(self) -> None:
        """Not an error: it means nothing can be installed automatically."""
        reports = toolchain.parse_probe("git=no=\nmake=no=\n")

        assert available_managers(reports) == ()

    def test_the_first_available_manager_wins(self) -> None:
        assert install_command("git", ("python", "winget", "choco")).startswith("winget")
        assert install_command("git", ("python", "choco")).startswith("choco")

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

        make has winget and choco commands and no python one, so a node with
        only python cannot have it installed automatically.
        """
        reports = (
            ToolReport(name="make", present=False, version=""),
            ToolReport(name="python", present=True, version="Python 3.11.9"),
            ToolReport(name="winget", present=False, version=""),
            ToolReport(name="choco", present=False, version=""),
        )

        assert toolchain.installable(reports) == ()

    def test_the_install_script_echoes_each_step(self) -> None:
        """So a transcript says which command produced which failure."""
        body = toolchain.install_script(("poetry", "make"), ("python", "choco"))

        assert "installing poetry" in body
        assert "python -m pip install --user poetry" in body
        assert "choco install make -y" in body

    def test_the_same_tool_renders_a_different_command_per_node(self) -> None:
        """THE DEFECT THE MAPPING REPLACED.

        The first version hardcoded `choco install`, inferred from loki's
        make living under the chocolatey lib directory -- one node
        generalised to three. lavender has no choco at all, so that command
        would have failed there with choco's own 'not recognized'.
        """
        on_lavender = toolchain.install_script(("make",), ("python", "winget"))
        on_loki = toolchain.install_script(("make",), ("python", "choco"))

        assert "winget install --id GnuWin32.Make" in on_lavender
        assert "choco" not in on_lavender
        assert "choco install make -y" in on_loki
        assert "winget" not in on_loki

    def test_a_tool_with_no_command_for_these_managers_is_refused(self) -> None:
        """Silently omitting it would report an install covering less than
        it claimed, and the caller would re-probe to find the tool absent
        with no explanation."""
        with pytest.raises(ValueError, match="no install command"):
            toolchain.install_script(("make",), ("python",))

    def test_installing_runs_the_command_and_names_what_it_did(self) -> None:
        runner = FakeRun([ok(""), ok("installing make")])
        _test_hooks.run = runner

        installed = toolchain.install_missing(_node(), toolchain.parse_probe(_SEDONA))

        assert installed == ("make",)
        # sedona has BOTH managers, and winget is the declared preference.
        assert b"winget install --id GnuWin32.Make" in (runner.stdin[0] or b"")

    def test_a_node_with_nothing_installable_is_left_alone(self) -> None:
        """No ssh call at all, which is what the empty reply list asserts."""
        _test_hooks.run = FakeRun([])

        assert toolchain.install_missing(_node("loki"), toolchain.parse_probe(_LOKI)) == ()

    def test_a_failing_install_is_not_softened(self) -> None:
        """A half-installed node is worse than an untouched one: it looks ready."""
        _test_hooks.run = FakeRun([ok(""), failed(1, "choco: not found")])

        with pytest.raises(AppError) as excinfo:
            toolchain.install_missing(_node(), toolchain.parse_probe(_SEDONA))

        assert excinfo.value.code is FleetErrorCode.DISPATCH_FAILED
        assert "choco: not found" in excinfo.value.message


class TestProbeToolchain:
    def test_it_sends_the_constant_script_and_parses_the_answer(self) -> None:
        runner = FakeRun([ok(""), ok(_LOKI)])
        _test_hooks.run = runner

        reports = toolchain.probe_toolchain(_node("loki"))

        assert len(reports) == 7
        assert runner.stdin[0] == toolchain.PROBE_SCRIPT.encode("utf-8")

    def test_an_unreachable_node_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "timed out")])

        with pytest.raises(AppError) as excinfo:
            toolchain.probe_toolchain(_node())

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE


class TestToolReportCodec:
    def test_a_report_survives_encoding(self) -> None:
        original = ToolReport(name="poetry", present=True, version="Poetry (version 2.2.1)")

        assert decode_tool_report(load_json_str(dump_json_str(encode_tool_report(original)))) == (
            original
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_tool_report("poetry")

    def test_absent_but_versioned_is_refused(self) -> None:
        """Only a present tool can have reported a version."""
        with pytest.raises(JSONTypeError, match="came from different reads"):
            decode_tool_report({"name": "make", "present": False, "version": "4.4.1"})

    def test_every_required_tool_carries_a_reason(self) -> None:
        """A refusal that names a binary and not what it was for is half a
        diagnostic."""
        assert all(tool["reason"] for tool in REQUIRED_TOOLS)


class TestBootstrapCommand:
    def test_a_ready_fleet_exits_zero(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_LOKI), ok(""), ok(_LOKI)])

        assert bootstrap.main([_config.CONFIG_FLAG, str(config_path)]) == 0

    def test_an_unready_node_exits_one(self, config_path: pathlib.Path) -> None:
        """Usable as a gate in front of a dispatch, not something to read."""
        _test_hooks.run = FakeRun([ok(""), ok(_LAVENDER), ok(""), ok(_LOKI)])

        assert bootstrap.main([_config.CONFIG_FLAG, str(config_path)]) == 1

    def test_a_named_node_is_asked_alone(self, config_path: pathlib.Path) -> None:
        runner = FakeRun([ok(""), ok(_LOKI)])
        _test_hooks.run = runner

        assert (
            bootstrap.main([_config.CONFIG_FLAG, str(config_path), bootstrap.NODE_FLAG, "loki"])
            == 0
        )
        assert len(runner.calls) == 2

    def test_an_unknown_node_is_refused(self, config_path: pathlib.Path) -> None:
        with pytest.raises(AppError) as excinfo:
            bootstrap.main([_config.CONFIG_FLAG, str(config_path), bootstrap.NODE_FLAG, "sedona"])

        assert excinfo.value.code is FleetErrorCode.WORKSPACE_NODE_UNKNOWN

    def test_without_the_install_flag_nothing_is_installed(self, config_path: pathlib.Path) -> None:
        """Reporting is the default because these are other people's machines."""
        runner = FakeRun([ok(""), ok(_SEDONA)])
        _test_hooks.run = runner

        assert (
            bootstrap.main([_config.CONFIG_FLAG, str(config_path), bootstrap.NODE_FLAG, "lavender"])
            == 1
        )
        # The probe script itself names both managers, so the absence to
        # assert is an INSTALL command, not the word.
        assert not any(b"installing " in (sent or b"") for sent in runner.stdin)

    def test_with_the_install_flag_the_gap_is_closed_and_re_probed(
        self, config_path: pathlib.Path
    ) -> None:
        """Re-probed rather than assumed: an install that ran is not an
        install that worked."""
        _test_hooks.run = FakeRun(
            [ok(""), ok(_SEDONA), ok(""), ok("installing make"), ok(""), ok(_LOKI)]
        )

        assert (
            bootstrap.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    bootstrap.NODE_FLAG,
                    "lavender",
                    bootstrap.INSTALL_FLAG,
                ]
            )
            == 0
        )

    def test_installing_nothing_installable_does_not_re_probe(
        self, config_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok(""), ok(_WRONG_PYTHON)])
        _test_hooks.run = runner

        assert (
            bootstrap.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    bootstrap.NODE_FLAG,
                    "loki",
                    bootstrap.INSTALL_FLAG,
                ]
            )
            == 1
        )
        assert len(runner.calls) == 2

    def test_the_entrypoint_exits_with_the_gate_s_status(self, config_path: pathlib.Path) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(_LAVENDER)])
        saved = sys.argv
        sys.argv = [
            "fleet-bootstrap",
            _config.CONFIG_FLAG,
            str(config_path),
            bootstrap.NODE_FLAG,
            "lavender",
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                bootstrap.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 1

    def test_running_as_a_module_actually_checks(self, config_path: pathlib.Path) -> None:
        """Without an `if __name__` block it exits 0 having asked nothing,
        which reads as "every node is ready" -- the worst false answer a gate
        can give."""
        _test_hooks.run = FakeRun([ok(""), ok(_LAVENDER)])
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.bootstrap", None)
        sys.argv = [
            "x",
            _config.CONFIG_FLAG,
            str(config_path),
            bootstrap.NODE_FLAG,
            "lavender",
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.bootstrap", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.bootstrap"] = saved_module

        assert raised.value.code == 1
