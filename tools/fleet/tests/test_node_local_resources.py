"""A resource each node runs its own copy of (MCPs board task c4fc4f3e).

``corvis-fleet-testdb`` is a container every testdb node runs for itself. It
was declared exclusive when diphtheria was its only host, and on 2026-09-29,
the day lavender-wsl became the second, every testdb job lavender-wsl claimed
while diphtheria ran one was closed ``RESOURCE_HELD ... held fleet-wide``. The
cases here hold the new distinction both ways: two nodes' copies never
contend, one node's copy still does, and a fleet-wide name is untouched.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from fleet.cli import _config, run
from fleet.contracts.lease import Lease
from fleet.contracts.resources import (
    decode_held_names,
    decode_names,
    encode_names,
    fleet_wide,
    scoped,
)
from fleet.contracts.workspace import (
    decode_fleet_workspace,
    encode_fleet_workspace,
    require_project,
)
from fleet.core import _test_hooks, leases, run_lease
from tests.conftest import DEMO_NOW, DEMO_PROJECT, FakeClock, workspace_document

#: The real node-local resource.
TESTDB = "corvis-fleet-testdb"

#: A resource that stays one thing in the fleet.
SHARED = "corvis_test"


def _lease(
    *, node: str, resources: tuple[str, ...], run_id: str, project: str = DEMO_PROJECT
) -> Lease:
    """Build a lease on one node holding these (already scoped) names.

    Args:
        node: The node it is on.
        resources: The names it holds, as a lease records them.
        run_id: Its run.
        project: Its project.

    Returns:
        The lease.
    """
    return Lease(
        node=node,
        project=project,
        run_id=run_id,
        agent="opus-fleet-0929",
        session_id="s",
        acquired_unix=DEMO_NOW,
        expires_unix=DEMO_NOW + 600,
        resources=resources,
    )


def _document(*, node_local: tuple[str, ...], exclusive: tuple[str, ...]) -> JSONObject:
    """The demo workspace declaring these resources.

    Args:
        node_local: Its ``node_local_resources``.
        exclusive: The demo project's ``exclusive_resources``.

    Returns:
        The document.
    """
    document = workspace_document()
    projects = document["projects"]
    assert isinstance(projects, dict)
    plan = projects[DEMO_PROJECT]
    assert isinstance(plan, dict)
    plan["exclusive_resources"] = encode_names(exclusive)
    document["node_local_resources"] = encode_names(node_local)
    return document


class TestScoping:
    def test_a_node_local_name_is_held_as_that_node_s_copy(self) -> None:
        assert scoped((TESTDB, SHARED), node="lavender-wsl", node_local=(TESTDB,)) == (
            "corvis-fleet-testdb@lavender-wsl",
            SHARED,
        )

    def test_the_early_check_asks_only_about_fleet_wide_names(self) -> None:
        assert fleet_wide((TESTDB, SHARED), node_local=(TESTDB,)) == (SHARED,)


class TestTheNames:
    def test_a_declared_name_may_not_carry_the_scope_separator(self) -> None:
        with pytest.raises(JSONTypeError, match=r"project\.x\[0\] is 'db@node'; '@' is reserved"):
            decode_names(["db@node"], field="project.x")

    def test_a_lease_record_reads_both_scopes(self) -> None:
        assert decode_held_names([SHARED, "corvis-fleet-testdb@diphtheria"], field="r") == (
            SHARED,
            "corvis-fleet-testdb@diphtheria",
        )

    @pytest.mark.parametrize("entry", ["a@b@c", "@diphtheria", "corvis-fleet-testdb@"])
    def test_a_lease_record_refuses_a_scope_no_lease_writes(self, entry: str) -> None:
        with pytest.raises(JSONTypeError, match="one separator and both sides named"):
            decode_held_names([entry], field="lease.resources")


class TestTheLeaseFile:
    def test_two_nodes_hold_their_own_copies_at_once(self, tmp_path: pathlib.Path) -> None:
        """THE FIX. diphtheria's run no longer turns lavender-wsl's away."""
        path = tmp_path / "leases.json"
        for node, run_id in (("diphtheria", "a"), ("lavender-wsl", "b")):
            held = scoped((TESTDB,), node=node, node_local=(TESTDB,))
            lease = _lease(node=node, resources=held, run_id=run_id)
            leases.acquire(path, lease, now_unix=DEMO_NOW)
        running = leases.held_leases(path, now_unix=DEMO_NOW)
        assert [lease["run_id"] for lease in running] == ["a", "b"]

    def test_one_node_s_copy_is_still_exclusive_and_the_refusal_names_the_way_out(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "leases.json"
        held = scoped((TESTDB,), node="diphtheria", node_local=(TESTDB,))
        first = _lease(node="diphtheria", resources=held, run_id="a")
        leases.acquire(path, first, now_unix=DEMO_NOW)
        second = _lease(node="diphtheria", resources=held, run_id="b", project="libs/other")
        with pytest.raises(AppError) as refusal:
            leases.acquire(path, second, now_unix=DEMO_NOW)
        assert refusal.value.code is FleetErrorCode.RESOURCE_HELD
        assert refusal.value.message == (
            "cannot dispatch: corvis-fleet-testdb@diphtheria is held on diphtheria by "
            f"opus-fleet-0929 (run a, {DEMO_PROJECT} on diphtheria, session s), 600s remaining; "
            "each node runs its own copy, so another node carrying one is an alternative"
        )

    def test_a_fleet_wide_name_held_beside_a_copy_keeps_the_fleet_wide_message(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "leases.json"
        held = scoped((TESTDB, SHARED), node="diphtheria", node_local=(TESTDB,))
        first = _lease(node="diphtheria", resources=held, run_id="a")
        leases.acquire(path, first, now_unix=DEMO_NOW)
        second = _lease(node="diphtheria", resources=held, run_id="b", project="libs/other")
        with pytest.raises(AppError) as refusal:
            leases.acquire(path, second, now_unix=DEMO_NOW)
        assert "is held fleet-wide by" in refusal.value.message
        assert refusal.value.message.endswith("so no other node is an alternative")

    def test_a_dispatch_s_lease_holds_its_node_s_copy(self, tmp_path: pathlib.Path) -> None:
        document = _document(node_local=(TESTDB,), exclusive=(TESTDB, SHARED))
        workspace = decode_fleet_workspace(document)
        lease = run_lease.open_lease(
            node="lavender-wsl",
            project=DEMO_PROJECT,
            run_id="r",
            agent="opus-fleet-0929",
            session_id="s",
            plan=require_project(workspace, DEMO_PROJECT),
            node_local=workspace["node_local_resources"],
            now_unix=DEMO_NOW,
        )
        assert lease["resources"] == ("corvis-fleet-testdb@lavender-wsl", SHARED)


class TestWhatARunnerLeavesOut:
    """:func:`fleet.core.run_lease.held_on_node`, which a runner asks before
    it claims (MCPs board task 939ec5c7), against a real lease file."""

    def test_the_node_s_own_copy_holds_and_an_expired_lease_holds_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "leases.json"
        workspace = decode_fleet_workspace(_document(node_local=(TESTDB,), exclusive=(TESTDB,)))
        held = scoped((TESTDB,), node="diphtheria", node_local=(TESTDB,))
        leases.acquire(
            path,
            _lease(node="diphtheria", resources=held, run_id="a", project="libs/db"),
            now_unix=DEMO_NOW,
        )

        def ask(node: str) -> tuple[str, ...]:
            return run_lease.held_on_node(
                path,
                node=node,
                names=(DEMO_PROJECT,),
                projects=workspace["projects"],
                node_local=workspace["node_local_resources"],
            )

        _test_hooks.now = FakeClock(DEMO_NOW)
        assert (ask("diphtheria"), ask("lavender-wsl")) == ((DEMO_PROJECT,), ())
        _test_hooks.now = FakeClock(DEMO_NOW + 601)
        assert ask("diphtheria") == ()


class TestTheEarlyCheck:
    def test_another_node_s_copy_held_does_not_refuse_before_a_node_is_chosen(
        self, tmp_path: pathlib.Path
    ) -> None:
        config = tmp_path / "fleet.json"
        document = _document(node_local=(TESTDB,), exclusive=(TESTDB,))
        config.write_text(dump_json_str(document), encoding="utf-8")
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config)})
        held = scoped((TESTDB,), node="diphtheria", node_local=(TESTDB,))
        holder = _lease(node="diphtheria", resources=held, run_id="a")
        leases.acquire(loaded.leases, holder, now_unix=DEMO_NOW)
        # Returns rather than raising RESOURCE_HELD: the choice of node, and
        # then that node's own copy, decide.
        run.require_resources_free(loaded, require_project(loaded.workspace, DEMO_PROJECT))
        assert leases.held_leases(loaded.leases, now_unix=DEMO_NOW) == (holder,)


class TestTheWorkspace:
    def test_it_reads_and_writes_the_names(self) -> None:
        workspace = decode_fleet_workspace(_document(node_local=(TESTDB,), exclusive=(TESTDB,)))
        assert workspace["node_local_resources"] == (TESTDB,)
        assert encode_fleet_workspace(workspace)["node_local_resources"] == [TESTDB]

    def test_an_older_workspace_without_the_key_is_refused(self) -> None:
        document = _document(node_local=(), exclusive=())
        del document["node_local_resources"]
        with pytest.raises(JSONTypeError, match="workspace must declare node_local_resources"):
            decode_fleet_workspace(document)

    def test_a_name_no_project_declares_is_refused_as_the_typo_it_is(self) -> None:
        misspelt = _document(node_local=("corvis-fleet-testbd",), exclusive=(TESTDB,))
        with pytest.raises(JSONTypeError, match="names 'corvis-fleet-testbd', which no project"):
            decode_fleet_workspace(misspelt)
