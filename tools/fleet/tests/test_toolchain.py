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
from fleet.contracts.node import NodePlatform
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
from fleet.core import _test_hooks, dialect_linux, toolchain, windows_toolchain_probe
from tests._toolchain_fixtures import (
    DIPHTHERIA,
    GO_NO_INSTALL,
    GO_REASON,
    GO_WINGET,
    HOOKS_INSTALL,
    HOOKS_REASON,
    LAVENDER,
    LOKI,
    SEDONA,
    SEDONA_2026_09_23,
    WRONG_PYTHON,
    node,
)
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
        "max_disk_gb": 20.0,
        "checks_at_once": None,
    }
    node: JSONObject = {
        "host": "lavender",
        "platform": "windows",
        "stage_root": "C:/fleet/stage",
        "logical_cores": 16,
        "ram_gb": 32.0,
        "gpu": None,
        "enabled": True,
        "test_database": False,
        "rust": None,
        "cxx": None,
        "docker": None,
        "stack": None,
        "elevated": False,
        "wsl_host": None,
        "budget": budget,
    }
    return {
        "nodes": {"lavender": node, "loki": {**node, "host": "loki"}},
        "not_dispatchable": {},
        "data_paths": {},
        "node_local_resources": [],
        "projects": {
            "libs/demo": {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
                "required_tags": [],
                "source": None,
            }
        },
        "node_serve_seconds": 0,
        "node_poll_seconds": 180,
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
    def test_loki_has_every_tool_it_was_asked_for_and_the_right_python(self) -> None:
        """Not READY: its 2026-09-04 probe predates the node question, and an
        unanswered floor is not a met one (test_toolchain_readiness)."""
        assert missing(toolchain.parse_probe(LOKI)) == ()
        assert python_is_right(toolchain.parse_probe(LOKI))

    def test_lavender_is_missing_three_tools(self) -> None:
        assert missing(toolchain.parse_probe(LAVENDER)) == ("poetry", "git", "make")

    def test_sedona_is_missing_only_make(self) -> None:
        """MEASURED: make was on one node of three."""
        assert missing(toolchain.parse_probe(SEDONA)) == ("make",)

    def test_a_node_without_ffmpeg_can_build_and_is_told_what_it_misses(self) -> None:
        """serendipity's answer of 2026-10-01 (MCPs board task 512e7bf8), every
        other tool and winget present and 'where ffmpeg' empty. ffmpeg is a
        tag now (MCPs board task 939ec5c7), so the node is ready, and the
        tagged gap names the tool, who needs it and the install here."""
        reports = toolchain.parse_probe(
            "python=yes=Python 3.11.9\npoetry=yes=Poetry (version 2.4.2)\n"
            "git=yes=git version 2.55.0.windows.5\nmake=yes=GNU Make 4.4.1\n"
            "node=yes=v24.21.0\nffmpeg=no=\ntar=yes=bsdtar 3.8.8\nwinget=yes=v1.12\n"
        )

        assert missing(reports) == ()
        assert describe_gap("serendipity", reports) == "serendipity: ready"
        assert toolchain.absent_tagged(reports) == ("ffmpeg", "hooks", "go")
        assert toolchain.tagged_gap("serendipity", reports) == (
            "serendipity claims without the tag of every tool it lacks: ffmpeg -- grandma-api's "
            "check converts real audio files through ffmpeg, so those jobs go to a node that has "
            "it -- winget install --id Gyan.FFmpeg.Essentials -e --source winget --silent "
            "--accept-package-agreements --accept-source-agreements --disable-interactivity; "
            f"hooks -- {HOOKS_REASON}, so those jobs go to a node that has it -- no automatic "
            f"install on this node; go -- {GO_REASON}, so those jobs go to a node that has it "
            f"-- {GO_WINGET}"
        )

    def test_a_node_with_every_tagged_tool_has_no_tagged_gap(self) -> None:
        reports = toolchain.parse_probe(
            "ffmpeg=yes=ffmpeg version 7.1.1-essentials\n"
            "hooks=yes=C:\\Users\\austi\\.claude\\corvis-hooks.json\n"
            "go=yes=go version go1.27.1 windows/amd64\n"
        )

        assert toolchain.absent_tagged(reports) == ()
        assert toolchain.tagged_gap("sedona", reports) is None

    def test_a_tagged_tool_with_no_install_here_says_so(self) -> None:
        """A node whose only manager is pip has no command for ffmpeg, and is
        offered the pinned pip line for the hooks check's tools (MCPs board
        task ec895824)."""
        reports = toolchain.parse_probe("ffmpeg=no=\nhooks=no=\ngo=no=\npip=yes=pip 25.2\n")

        assert toolchain.tagged_gap("lenovoold", reports) == (
            "lenovoold claims without the tag of every tool it lacks: ffmpeg -- grandma-api's "
            "check converts real audio files through ffmpeg, so those jobs go to a node that has "
            f"it -- no automatic install on this node; hooks -- {HOOKS_REASON}, so those jobs go "
            f"to a node that has it -- {HOOKS_INSTALL}{GO_NO_INSTALL}"
        )

    def test_a_linux_node_is_offered_go_from_apt_get(self) -> None:
        """diphtheria and lavender-wsl answered go=no= on 2026-10-05 (MCPs board
        task 1da15750); their manager is apt-get, whose golang-go fetches the
        go.mod's toolchain itself."""
        reports = toolchain.parse_probe(
            "ffmpeg=yes=ffmpeg version 6.1.1\nhooks=yes=/home/corvis/.claude/corvis-hooks.json\n"
            "go=no=\napt-get=yes=apt 2.7.14 (amd64)\n"
        )

        assert toolchain.absent_tagged(reports) == ("go",)
        assert toolchain.tagged_gap("diphtheria", reports) == (
            f"diphtheria claims without the tag of every tool it lacks: go -- {GO_REASON}, so "
            "those jobs go to a node that has it -- sudo apt-get install -y golang-go"
        )

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
        assert describe_gap("sedona", toolchain.parse_probe(SEDONA_2026_09_23)) == "sedona: ready"

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
        toolchain.require_ready("sedona", node("sedona"), toolchain.parse_probe(SEDONA_2026_09_23))

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

        reports = toolchain.probe_toolchain(node("loki"), writer="fleet-bootstrap")

        assert len(reports) == 8
        assert runner.stdin[0] == windows_toolchain_probe.TOOLCHAIN_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[0][-1].endswith(
            "C:/fleet/stage/fleet-toolchain-fleet-bootstrap.ps1' -Encoding utf8\""
        )

    def test_a_linux_node_is_asked_in_sh(self) -> None:
        runner = FakeRun([ok(""), ok(DIPHTHERIA)])
        _test_hooks.run = runner
        linux = node("diphtheria")
        linux["platform"] = NodePlatform.LINUX
        linux["stage_root"] = "/home/corvis/fleet/stage"

        reports = toolchain.probe_toolchain(linux, writer="diphtheria")

        assert len(reports) == 7
        assert runner.stdin[0] == dialect_linux.TOOLCHAIN_PROBE_SCRIPT.encode("utf-8")
        assert runner.calls[1][-2:] == (
            "/bin/sh",
            "/home/corvis/fleet/stage/fleet-toolchain-diphtheria.sh",
        )

    def test_an_unreachable_node_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "timed out")])

        with pytest.raises(AppError) as excinfo:
            toolchain.probe_toolchain(node(), writer="fleet-bootstrap")

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
        _test_hooks.run = FakeRun([ok(""), ok(SEDONA_2026_09_23), ok(""), ok(SEDONA_2026_09_23)])

        assert bootstrap.main([_config.CONFIG_FLAG, str(config_path)]) == 0

    def test_an_unready_node_exits_one(self, config_path: pathlib.Path) -> None:
        """Usable as a gate in front of a dispatch, not something to read."""
        _test_hooks.run = FakeRun([ok(""), ok(LAVENDER), ok(""), ok(SEDONA_2026_09_23)])

        assert bootstrap.main([_config.CONFIG_FLAG, str(config_path)]) == 1

    def test_a_named_node_is_asked_alone(self, config_path: pathlib.Path) -> None:
        runner = FakeRun([ok(""), ok(SEDONA_2026_09_23)])
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
            [ok(""), ok(SEDONA), ok(""), ok("installing make"), ok(""), ok(SEDONA_2026_09_23)]
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
