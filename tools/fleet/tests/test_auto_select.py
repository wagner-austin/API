"""One asleep laptop must not disable the fleet.

THE INCIDENT THIS FILE IS THE REGRESSION TEST FOR, measured 2026-09-05 on the
first real auto-select dispatch ever made through the corvis queue:

    dispatch_submit(project="tools/fleet")          # no node named
    -> refused: NODE_UNREACHABLE: ssh to loki failed ... bad handshake

loki was powered off for a trip. lavender had already answered and had room.
The dispatch was refused anyway, because ``run.choose`` probed nodes in sorted
order and the probe RAISED -- so the first unreachable node ended the search
before any node that could have taken the work was weighed.

Two of this fleet's three nodes are laptops. One being asleep is the ordinary
case, not a fault, and no test had ever put a working node behind a broken one
because every fixture in this suite declares a single node. That is why the
whole suite was green while auto-select could not work at all.

So every test here uses AT LEAST TWO nodes, and the order matters: the
unreachable one is probed FIRST, because probing it second would pass even
under the old code.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str, narrow_json_to_dict

from fleet.cli import _config, run
from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.core import _test_hooks, probe, staging
from fleet.core.capacity import Unassessed, first_fit
from tests.conftest import (
    DEMO_NOW,
    DEMO_PROJECT,
    PROBE_OK,
    FakeClock,
    FakeRun,
    FakeTempRoot,
    dispatch_replies,
    failed,
    ok,
    prebuilt_archive,
    workspace_document,
)


def _node(host: str) -> NodeConfig:
    """Build a node declaration with room for the demo project.

    Args:
        host: SSH alias, which is also its workspace name here.

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


def _state(host: str, *, free_ram_gb: float = 27.0) -> NodeState:
    """Build a node's live state.

    Args:
        host: The node it belongs to.
        free_ram_gb: What it reported free.

    Returns:
        The state.
    """
    return NodeState(host=host, free_ram_gb=free_ram_gb, free_disk_gb=860.0, live_runs=0)


def _project() -> ProjectConfig:
    """Build a project that fits comfortably on a healthy node.

    Returns:
        The project.
    """
    return ProjectConfig(
        worker_ram_gb=1.1,
        minimum_workers=2,
        expected_minutes=5,
        exclusive_resources=(),
        external_paths=(),
    )


def _silent(name: str, reason: str) -> Unassessed:
    """A node that was asked over ssh and did not answer.

    Args:
        name: Its workspace name.
        reason: What the ssh call reported.

    Returns:
        The entry to hand ``first_fit``.
    """
    return Unassessed(name=name, reason=reason, asked=True)


def _off(name: str) -> Unassessed:
    """A node the workspace declares disabled, so nothing was sent to it.

    Args:
        name: Its workspace name.

    Returns:
        The entry to hand ``first_fit``.
    """
    return Unassessed(name=name, reason="declared disabled in this workspace", asked=False)


class TestFirstFitWeighsWhatAnswered:
    def test_an_unreachable_node_does_not_stop_a_healthy_one_winning(self) -> None:
        """THE REGRESSION. loki asleep, lavender fine -> lavender takes it."""
        chosen, workers = first_fit(
            ((("lavender"), _node("lavender"), _state("lavender")),),
            _project(),
            unassessed=(_silent("loki", "ssh to loki failed: bad handshake"),),
        )

        assert chosen == "lavender"
        assert workers > 0

    def test_the_refusal_names_the_unreachable_node_too(self) -> None:
        """A caller must learn "loki is off AND sedona is full" in ONE answer.

        Learning about whichever node happened to be probed first is what
        sends somebody to the wrong machine.
        """
        with pytest.raises(AppError) as excinfo:
            first_fit(
                (("sedona", _node("sedona"), _state("sedona", free_ram_gb=0.2)),),
                _project(),
                unassessed=(_silent("loki", "ssh to loki failed: bad handshake"),),
            )

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "loki: ssh to loki failed" in excinfo.value.message
        assert "sedona:" in excinfo.value.message

    def test_when_nothing_answered_the_code_says_unreachable(self) -> None:
        """ "The fleet is off" and "the fleet is busy" send a reader to
        different places -- the tailnet, or the clock."""
        with pytest.raises(AppError) as excinfo:
            first_fit(
                (),
                _project(),
                unassessed=(_silent("loki", "ssh to loki failed"), _silent("sedona", "timed out")),
            )

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "loki:" in excinfo.value.message
        assert "sedona:" in excinfo.value.message

    def test_a_fleet_that_was_never_asked_is_not_called_unreachable(self) -> None:
        """NOTHING FAILED HERE. Every node is switched off in the workspace,
        so no ssh was sent and no machine declined to answer. Calling that
        NODE_UNREACHABLE sends the reader to the tailnet to debug a network
        that is fine, when the fix is one boolean in fleet.json."""
        with pytest.raises(AppError) as excinfo:
            first_fit(
                (),
                _project(),
                unassessed=(_off("loki"), _off("sedona")),
            )

        assert excinfo.value.code is FleetErrorCode.NODE_DISABLED
        assert "loki: declared disabled" in excinfo.value.message
        assert "sedona: declared disabled" in excinfo.value.message

    def test_one_silent_node_outranks_the_disabled_ones(self) -> None:
        """A machine that was asked and said nothing is the only one of the
        three with something to investigate, so it names the refusal."""
        with pytest.raises(AppError) as excinfo:
            first_fit(
                (),
                _project(),
                unassessed=(_off("loki"), _silent("sedona", "timed out")),
            )

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE

    def test_a_workspace_with_no_nodes_is_not_called_unreachable(self) -> None:
        """Declaring zero nodes is a config fault, not a fleet that is down.
        Nothing was tried, so nothing failed to answer."""
        with pytest.raises(AppError) as excinfo:
            first_fit((), _project())

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED


class TestAttemptProbeReportsRatherThanRaises:
    def test_an_unreachable_node_comes_back_as_a_reason(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "bad handshake")])

        outcome = probe.attempt_probe(_node("loki"), live_runs=0)

        assert outcome["state"] is None
        assert "ssh to loki failed" in outcome["reason"]
        assert "bad handshake" in outcome["reason"]

    def test_a_node_answering_gibberish_comes_back_as_a_reason(self) -> None:
        """It ANSWERED, so it is not unreachable in the ssh sense -- but
        nothing can be decided about it either, and auto-select must weigh
        that the same way rather than abandoning the fleet over it."""
        _test_hooks.run = FakeRun([ok(""), ok("Get-CimInstance : access denied")])

        outcome = probe.attempt_probe(_node("loki"), live_runs=0)

        assert outcome["state"] is None
        assert "free_ram_gb" in outcome["reason"]

    def test_a_healthy_node_comes_back_with_its_state(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        outcome = probe.attempt_probe(_node("lavender"), live_runs=3)

        state = outcome["state"]
        if state is None:
            raise AssertionError(f"expected a state, got: {outcome['reason']}")
        assert state["host"] == "lavender"
        assert state["free_ram_gb"] == 27.0
        assert state["live_runs"] == 3
        assert outcome["reason"] == ""

    def test_the_named_node_path_still_raises(self) -> None:
        """`fleet-run --node lavender` asked for one machine. If it is down
        that is an ERROR -- silently running somewhere else would answer a
        question nobody asked."""
        _test_hooks.run = FakeRun([failed(255, "bad handshake")])

        with pytest.raises(AppError) as excinfo:
            probe.probe_node(_node("lavender"), live_runs=0)

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE


class TestAnAsleepNodeDoesNotBlockADispatch:
    """End to end through `fleet-run`, with a real two-node workspace."""

    @pytest.fixture(name="two_node_config")
    def _two_node_config(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Write a workspace whose FIRST node sorts before a healthy one.

        ``asleep`` sorts before ``lavender``, so the unreachable node is
        probed first. Probing it second would pass even under the old code,
        which is exactly why the single-node fixtures never caught this.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Returns:
            Path to the written document.
        """
        _test_hooks.now = FakeClock(DEMO_NOW)
        _test_hooks.temp_root = FakeTempRoot(tmp_path / "scratch")
        document = workspace_document()
        nodes = narrow_json_to_dict(document["nodes"])
        asleep = dict(narrow_json_to_dict(nodes["lavender"]))
        asleep["host"] = "asleep"
        nodes["asleep"] = asleep
        document["nodes"] = nodes
        path = tmp_path / "fleet.json"
        path.write_text(dump_json_str(document), encoding="utf-8")
        return path

    def test_the_dispatch_lands_on_the_node_that_is_awake(
        self, two_node_config: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """THE MEASURED INCIDENT, AS A TEST. Before the fix this raised
        NODE_UNREACHABLE naming the asleep node and dispatched nothing."""
        payload = prebuilt_archive(two_node_config, repo)
        _test_hooks.run = FakeRun(
            [
                ok(""),  # asleep: send probe script
                failed(255, "bad handshake"),  # asleep: ssh dies
                *dispatch_replies(staging.digest(payload)),  # lavender takes it
            ]
        )

        assert (
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(two_node_config),
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

        ledger = (two_node_config.parent / "runs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert '"node":"lavender"' in ledger
        assert '"node":"asleep"' not in ledger

    def test_both_nodes_asleep_refuses_and_names_both(
        self, two_node_config: pathlib.Path, repo: pathlib.Path
    ) -> None:
        _test_hooks.run = FakeRun(
            [
                ok(""),
                failed(255, "bad handshake"),
                ok(""),
                failed(255, "timed out"),
            ]
        )

        with pytest.raises(AppError) as excinfo:
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(two_node_config),
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

        assert excinfo.value.code is FleetErrorCode.NODE_UNREACHABLE
        assert "asleep:" in excinfo.value.message
        assert "lavender:" in excinfo.value.message
