"""Whether a node can run a build, and the command that says so.

THE FIXTURES ARE THE REAL MEASUREMENT (:mod:`tests._toolchain_fixtures`).
``sedona``, ``lavender`` and ``loki`` answered those exact shapes on
2026-09-04, and one node of the three could have run a ``make check``. A test
written against an invented "node with everything" would have proved the happy
path and nothing about the fleet that actually exists. Installing is tested in
``test_toolchain_install.py``.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str, load_json_str

from fleet.cli import _config, bootstrap
from fleet.contracts.toolchain import (
    REQUIRED_TOOLS,
    ToolReport,
    available_managers,
    decode_tool_report,
    describe_gap,
    encode_tool_report,
    missing,
    python_is_right,
    version_number,
)
from fleet.core import _test_hooks, dialect_linux, dialect_windows, toolchain
from tests._toolchain_fixtures import DIPHTHERIA, LAVENDER, LOKI, SEDONA, WRONG_PYTHON, node
from tests.conftest import FakeRun, failed, ok


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
        "platform": "windows",
        "stage_root": "C:/fleet/stage",
        "logical_cores": 16,
        "ram_gb": 32.0,
        "gpu": None,
        "enabled": True,
        "budget": budget,
    }
    return {
        "nodes": {"lavender": node, "loki": {**node, "host": "loki"}},
        "not_dispatchable": {},
        "projects": {
            "libs/demo": {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
                "required_tags": [],
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
        reports = toolchain.parse_probe(LOKI)

        assert [report["name"] for report in reports] == [
            "python",
            "poetry",
            "git",
            "make",
            "tar",
            "winget",
            "choco",
            "pip",
        ]
        assert reports[1]["version"] == "Poetry (version 2.1.3)"

    def test_a_linux_answer_reads_the_same_way(self) -> None:
        reports = toolchain.parse_probe(DIPHTHERIA)

        assert [report["name"] for report in reports] == [
            "python",
            "poetry",
            "git",
            "make",
            "tar",
            "apt-get",
            "pipx",
        ]
        assert reports[0]["version"] == "Python 3.12.3"
        assert available_managers(reports) == ("pipx", "apt-get")
        assert missing(reports) == ()
        assert python_is_right(reports) is False

    def test_an_absent_tool_carries_no_version(self) -> None:
        reports = toolchain.parse_probe(LAVENDER)

        absent = [report for report in reports if report["name"] == "make"]
        assert absent == [ToolReport(name="make", present=False, version="")]

    def test_a_line_that_is_not_a_tool_is_ignored(self) -> None:
        """PowerShell writes warnings to the same stream."""
        reports = toolchain.parse_probe("WARNING: something\n" + LOKI)

        assert len(reports) == 8

    def test_an_unrecognisable_answer_is_a_probe_that_did_not_run(self) -> None:
        """Reporting five absent tools would send the reader to install
        five things that are already there."""
        with pytest.raises(AppError) as excinfo:
            toolchain.parse_probe("The term 'foreach' is not recognized")

        assert excinfo.value.code is FleetErrorCode.NODE_TOOL_MISSING
        assert "never asked" in excinfo.value.message


class TestReadiness:
    def test_loki_is_ready(self) -> None:
        assert missing(toolchain.parse_probe(LOKI)) == ()
        assert python_is_right(toolchain.parse_probe(LOKI))

    def test_lavender_is_missing_three_tools(self) -> None:
        assert missing(toolchain.parse_probe(LAVENDER)) == ("poetry", "git", "make")

    def test_sedona_is_missing_only_make(self) -> None:
        """MEASURED: make was on one node of three."""
        assert missing(toolchain.parse_probe(SEDONA)) == ("make",)

    def test_the_wrong_python_is_not_ready_even_with_every_tool(self) -> None:
        reports = toolchain.parse_probe(WRONG_PYTHON)

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
        assert describe_gap("loki", toolchain.parse_probe(LOKI)) == "loki: ready"

    def test_it_names_the_install_command(self) -> None:
        """The reader's next question is always how to fix it."""
        described = describe_gap("lavender", toolchain.parse_probe(LAVENDER))

        assert "poetry (python -m pip install --user poetry)" in described
        # lavender has winget and no choco, so the command it is told to run
        # is the winget one. loki, missing the same tool, would be told choco.
        assert "make (winget install --id GnuWin32.Make" in described
        assert "choco" not in described

    def test_a_tool_with_no_automatic_install_says_so(self) -> None:
        described = describe_gap("odd", (ToolReport(name="tar", present=False, version=""),))

        assert "tar (install by hand)" in described

    def test_the_wrong_python_is_named_with_what_was_found(self) -> None:
        described = describe_gap("loki", toolchain.parse_probe(WRONG_PYTHON))

        assert "python 3.11 (found Python 3.12.4)" in described

    def test_an_unreported_python_reads_as_unknown(self) -> None:
        described = describe_gap("odd", (ToolReport(name="git", present=True, version="x"),))

        assert "found unknown" in described


class TestRequireReady:
    def test_a_ready_node_passes(self) -> None:
        toolchain.require_ready("loki", node("loki"), toolchain.parse_probe(LOKI))

    def test_a_missing_tool_names_every_one_and_why(self) -> None:
        with pytest.raises(AppError) as excinfo:
            toolchain.require_ready("lavender", node(), toolchain.parse_probe(LAVENDER))

        assert excinfo.value.code is FleetErrorCode.NODE_TOOL_MISSING
        assert "make check is the entry point" in excinfo.value.message
        assert "winget install --id GnuWin32.Make" in excinfo.value.message

    def test_the_wrong_python_is_its_own_code(self) -> None:
        """The fixes differ: a package manager versus a decision."""
        with pytest.raises(AppError) as excinfo:
            toolchain.require_ready("loki", node("loki"), toolchain.parse_probe(WRONG_PYTHON))

        assert excinfo.value.code is FleetErrorCode.NODE_PYTHON_MISMATCH
        assert "3.12.4" in excinfo.value.message

    def test_a_node_that_reported_no_python_says_unknown(self) -> None:
        reports = tuple(
            report for report in toolchain.parse_probe(LOKI) if report["name"] != "python"
        )

        with pytest.raises(AppError, match="unknown"):
            toolchain.require_ready("odd", node(), reports)


class TestProbeToolchain:
    def test_it_sends_the_constant_script_and_parses_the_answer(self) -> None:
        runner = FakeRun([ok(""), ok(LOKI)])
        _test_hooks.run = runner

        reports = toolchain.probe_toolchain(node("loki"))

        assert len(reports) == 8
        assert runner.stdin[0] == dialect_windows.TOOLCHAIN_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[0][-1].endswith("C:/fleet/stage/fleet-toolchain.ps1' -Encoding utf8\"")

    def test_a_linux_node_is_asked_in_sh(self) -> None:
        runner = FakeRun([ok(""), ok(DIPHTHERIA)])
        _test_hooks.run = runner
        linux = node("diphtheria")
        linux["platform"] = "linux"
        linux["stage_root"] = "/home/corvis/fleet/stage"

        reports = toolchain.probe_toolchain(linux)

        assert len(reports) == 7
        assert runner.stdin[0] == dialect_linux.TOOLCHAIN_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[1][-2:] == ("/bin/sh", "/home/corvis/fleet/stage/fleet-toolchain.sh")

    def test_an_unreachable_node_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "timed out")])

        with pytest.raises(AppError) as excinfo:
            toolchain.probe_toolchain(node())

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
        _test_hooks.run = FakeRun([ok(""), ok(LOKI), ok(""), ok(LOKI)])

        assert bootstrap.main([_config.CONFIG_FLAG, str(config_path)]) == 0

    def test_an_unready_node_exits_one(self, config_path: pathlib.Path) -> None:
        """Usable as a gate in front of a dispatch, not something to read."""
        _test_hooks.run = FakeRun([ok(""), ok(LAVENDER), ok(""), ok(LOKI)])

        assert bootstrap.main([_config.CONFIG_FLAG, str(config_path)]) == 1

    def test_a_named_node_is_asked_alone(self, config_path: pathlib.Path) -> None:
        runner = FakeRun([ok(""), ok(LOKI)])
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
        runner = FakeRun([ok(""), ok(SEDONA)])
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
            [ok(""), ok(SEDONA), ok(""), ok("installing make"), ok(""), ok(LOKI)]
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
        runner = FakeRun([ok(""), ok(WRONG_PYTHON)])
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
        _test_hooks.run = FakeRun([ok(""), ok(LAVENDER)])
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
        _test_hooks.run = FakeRun([ok(""), ok(LAVENDER)])
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
