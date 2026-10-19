"""A node the workspace declares off is never asked, and says so.

"WAS NEVER ASKED" IS NOT "DID NOT ANSWER", and until 2026-09-05 this package
could not tell the difference. The fleet's identity registry
(``fleet-mcp/fleet-nodes.json``) had carried ``enabled`` since the fleet was
first written down; this workspace had no such field, so loki -- powered off
for a trip, and marked so in the other file -- was probed on every single
dispatch. Each one paid a ten-second ssh ConnectTimeout to rediscover it, and
one auto-select dispatch was refused outright because of it.

The three states are now distinct, and the distinction is the point:

  disabled     deliberately off. Not probed at all.
  unreachable  should have answered and did not. Probed, cost a timeout.
  full         answered, no room.

They call for three different actions -- edit the workspace, check the tailnet,
wait -- and a reader given one code for all three goes to the wrong one twice.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import _config, nodes, run
from fleet.core import _test_hooks, staging
from tests._registry import write_registry
from tests._roster_fixtures import machine_document, machine_of, machines_of
from tests.conftest import (
    DEMO_NOW,
    DEMO_PROJECT,
    PROBE_OK,
    FakeClock,
    FakeRun,
    FakeTempRoot,
    dispatch_replies,
    ok,
    prebuilt_archive,
    workspace_document,
)


@pytest.fixture(name="mixed_config")
def _mixed_config(tmp_path: pathlib.Path) -> pathlib.Path:
    """A workspace with one enabled node and one deliberately disabled.

    ``asleep`` sorts BEFORE ``lavender``, so a run that probed it would do so
    first -- which is how the old single-node fixtures hid this entirely.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the written document.
    """
    _test_hooks.now = FakeClock(DEMO_NOW)
    _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
    document = workspace_document()
    machines_of(document)["asleep"] = machine_document(enabled=False)
    return write_registry(tmp_path / "registry", document)


class TestAutoSelectSkipsWithoutProbing:
    def test_a_disabled_node_costs_no_ssh_at_all(
        self, mixed_config: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE MEASURED COST. The scripted runner holds ONLY the calls
        lavender's dispatch makes -- no probe pair for ``asleep``. If the
        disabled node were probed, this runner would raise "unscripted call"
        on its very first command."""
        payload = prebuilt_archive(mixed_config, repo)
        runner = FakeRun(dispatch_replies(staging.digest(payload), mixed_config))
        _test_hooks.run = runner

        assert (
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.AGENT_FLAG,
                    "opus-fleet-0905",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )
            == 0
        )

        assert not any("asleep" in " ".join(call) for call in runner.calls)
        ledger = (mixed_config.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert '"node":"lavender"' in ledger

    def test_when_every_node_is_disabled_the_refusal_names_them(
        self, tmp_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Nothing was tried, so this is not NODE_UNREACHABLE -- nothing
        failed to answer. It is a workspace that has switched itself off."""
        _test_hooks.now = FakeClock(DEMO_NOW)
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        document = workspace_document()
        machine_of(document, "lavender")["enabled"] = False
        path = write_registry(tmp_path / "registry", document)
        _test_hooks.run = FakeRun([])

        with pytest.raises(AppError) as excinfo:
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.AGENT_FLAG,
                    "opus-fleet-0905",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )

        assert excinfo.value.code is FleetErrorCode.NODE_DISABLED
        assert "lavender: declared disabled in this workspace" in excinfo.value.message


class TestNamingADisabledNodeIsAnError:
    def test_it_is_refused_by_its_own_code_and_never_rerouted(
        self, mixed_config: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Silently running somewhere else would answer a question nobody
        asked -- the same argument that keeps --node raising when the machine
        is merely unreachable."""
        _test_hooks.run = FakeRun([])

        with pytest.raises(AppError) as excinfo:
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.NODE_FLAG,
                    "asleep",
                    run.AGENT_FLAG,
                    "opus-fleet-0905",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )

        assert excinfo.value.code is FleetErrorCode.NODE_DISABLED
        assert "nothing was asked of it" in excinfo.value.message
        # The roster is one file per machine (MCPs board task 33831692).
        assert "Set enabled in registry/machines/asleep.json" in excinfo.value.message


class TestFleetNodesReports:
    def test_a_disabled_node_is_a_line_and_not_a_failure(
        self, mixed_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """It has not failed to answer, so it must not push the exit status
        non-zero -- otherwise a fleet with one machine deliberately off reads
        as a fleet with a fault, every run, forever."""
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        with caplog.at_level("INFO"):
            assert nodes.main([_config.CONFIG_FLAG, str(mixed_config)]) == 0

        assert "asleep: DISABLED -- declared off in this workspace, not probed" in caplog.text

    def test_the_identity_registry_is_no_longer_read_here(self, mixed_config: pathlib.Path) -> None:
        """The roster is this registry's own ``machines/`` (MCPs board task
        e447de72), so the flag that compared it with MCPs' old copy is
        refused as unknown rather than ignored."""
        _test_hooks.run = FakeRun([])

        with pytest.raises(ValueError, match=r"unknown argument '--registry'"):
            nodes.main([_config.CONFIG_FLAG, str(mixed_config), "--registry", "fleet-nodes.json"])
