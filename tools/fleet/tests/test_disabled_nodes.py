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
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, nodes, run
from fleet.core import _test_hooks, staging
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
    declared = document["nodes"]
    if not isinstance(declared, dict):
        raise AssertionError("the fixture workspace must declare nodes")
    template = declared["lavender"]
    if not isinstance(template, dict):
        raise AssertionError("the fixture workspace must declare lavender")
    asleep: JSONObject = dict(template)
    asleep["host"] = "asleep"
    asleep["enabled"] = False
    declared["asleep"] = asleep
    document["nodes"] = declared
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return path


class TestAutoSelectSkipsWithoutProbing:
    def test_a_disabled_node_costs_no_ssh_at_all(
        self, mixed_config: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE MEASURED COST. The scripted runner holds ONLY the calls
        lavender's dispatch makes -- no probe pair for ``asleep``. If the
        disabled node were probed, this runner would raise "unscripted call"
        on its very first command."""
        payload = prebuilt_archive(mixed_config, repo)
        runner = FakeRun(dispatch_replies(staging.digest(payload)))
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
        declared = document["nodes"]
        if not isinstance(declared, dict):
            raise AssertionError("the fixture workspace must declare nodes")
        lavender = declared["lavender"]
        if not isinstance(lavender, dict):
            raise AssertionError("the fixture workspace must declare lavender")
        lavender["enabled"] = False
        path = tmp_path / "fleet.json"
        path.write_text(dump_json_str(document), encoding="utf-8")
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


class TestFleetNodesReportsAndReconciles:
    def test_a_disabled_node_is_a_line_and_not_a_failure(self, mixed_config: pathlib.Path) -> None:
        """It has not failed to answer, so it must not push the exit status
        non-zero -- otherwise a fleet with one machine deliberately off reads
        as a fleet with a fault, every run, forever."""
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        assert nodes.main([_config.CONFIG_FLAG, str(mixed_config)]) == 0

    def test_the_registry_flag_reports_drift_and_exits_non_zero(
        self, mixed_config: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        registry_path = tmp_path / "fleet-nodes.json"
        registry_path.write_text(
            dump_json_str(
                {
                    "nodes": [
                        {"name": "lavender", "enabled": True},
                        {"name": "asleep", "enabled": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        assert (
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.REGISTRY_FLAG,
                    str(registry_path),
                ]
            )
            == 1
        )

    def test_agreement_leaves_the_status_alone(
        self, mixed_config: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        registry_path = tmp_path / "fleet-nodes.json"
        registry_path.write_text(
            dump_json_str(
                {
                    "nodes": [
                        {"name": "lavender", "enabled": True},
                        {"name": "asleep", "enabled": False},
                    ]
                }
            ),
            encoding="utf-8",
        )
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        assert (
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.REGISTRY_FLAG,
                    str(registry_path),
                ]
            )
            == 0
        )

    def test_without_the_flag_no_reconciliation_is_claimed(
        self, mixed_config: pathlib.Path
    ) -> None:
        """Omitting it must not read as "the registries agree". Nobody asked,
        so nothing was established."""
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        assert nodes.main([_config.CONFIG_FLAG, str(mixed_config)]) == 0


class TestReconcilingWithoutTouchingTheNetwork:
    """``--probe never``: the two-file comparison, and nothing else.

    THE CALLER IS A SCHEDULE. Reconciliation asks whether two files on one
    disk agree, so paying an ssh round trip per node every N minutes to
    answer it is the wrong shape -- and it puts two schedules on the same
    fleet, producing two answers about one node minutes apart. Asked for by
    the owner of ``fleet-nodes.json`` on board task df6f1dc8.
    """

    def _registry(self, tmp_path: pathlib.Path, **enabled: bool) -> pathlib.Path:
        """Write an identity registry declaring the given nodes.

        Args:
            tmp_path: pytest's per-test temporary directory.
            **enabled: Node name to whether the registry says it should
                answer.

        Returns:
            Path to the written registry.
        """
        path = tmp_path / "fleet-nodes.json"
        path.write_text(
            dump_json_str({"nodes": [{"name": name, "enabled": v} for name, v in enabled.items()]}),
            encoding="utf-8",
        )
        return path

    def test_not_one_ssh_call_is_made(
        self, mixed_config: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """THE MEASURED COST, and the only reason the flag exists. The
        scripted runner holds NOTHING, so any command sent to any node --
        including the enabled one -- raises "unscripted call" rather than
        passing quietly."""
        runner = FakeRun([])
        _test_hooks.run = runner

        assert (
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.REGISTRY_FLAG,
                    str(self._registry(tmp_path, lavender=True, asleep=False)),
                    nodes.PROBE_FLAG,
                    nodes.PROBE_NEVER,
                ]
            )
            == 0
        )

        assert runner.calls == []

    def test_drift_still_sets_the_status(
        self, mixed_config: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Skipping the probe must not also skip the finding. A schedule that
        exits 0 on drift is a schedule nobody would notice."""
        _test_hooks.run = FakeRun([])

        assert (
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.REGISTRY_FLAG,
                    str(self._registry(tmp_path, lavender=True, asleep=True)),
                    nodes.PROBE_FLAG,
                    nodes.PROBE_NEVER,
                ]
            )
            == 1
        )

    def test_asking_always_is_the_same_as_not_asking(self, mixed_config: pathlib.Path) -> None:
        """The default has to stay the default. Every existing caller and the
        20-minute fleet audit depend on the bare command probing."""
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        assert (
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.PROBE_FLAG,
                    nodes.PROBE_ALWAYS,
                ]
            )
            == 0
        )

    def test_a_mode_that_does_not_exist_is_refused(self, mixed_config: pathlib.Path) -> None:
        """``--probe no`` must not be read as ``never``. A schedule that
        quietly probed anyway would reintroduce the exact cost the flag was
        added to remove, while reporting success."""
        _test_hooks.run = FakeRun([])

        with pytest.raises(ValueError, match=r"--probe must be one of"):
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.PROBE_FLAG,
                    "no",
                ]
            )

    def test_probing_nothing_and_reconciling_nothing_is_refused(
        self, mixed_config: pathlib.Path
    ) -> None:
        """It would print no lines and exit 0 -- a silence indistinguishable
        from a fleet in perfect health, which is the one thing an exit status
        must never be able to mean by accident."""
        _test_hooks.run = FakeRun([])

        with pytest.raises(ValueError, match=r"needs --registry"):
            nodes.main(
                [
                    _config.CONFIG_FLAG,
                    str(mixed_config),
                    nodes.PROBE_FLAG,
                    nodes.PROBE_NEVER,
                ]
            )
