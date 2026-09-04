"""Serialising a thing there is exactly one of in the fleet.

THE DISTINCTION EVERY TEST HERE IS ABOUT. A held ENVIRONMENT is per node, so
the answer to a refusal is another node. A held RESOURCE has no second copy
anywhere, so the answer is to wait. If those two ever share a code path or a
message, half the readers go hunting for capacity that could not have helped
them -- which is why they have separate error codes and separate renderings,
and why the check happens before any node is probed rather than among them.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str

from fleet.cli import _config, preflight, run
from fleet.contracts.lease import Lease, contends, decode_lease, encode_lease
from fleet.contracts.project import decode_project_config
from fleet.contracts.resources import contended, decode_names
from fleet.core import _test_hooks, leases, records, staging
from tests.conftest import (
    DEMO_NOW,
    DEMO_PROJECT,
    DEMO_RUN_ID,
    PROBE_OK,
    FakeClock,
    FakeRun,
    dispatch_replies,
    ok,
    prebuilt_archive,
    workspace_document,
)

#: A second project sharing the demo project's exclusive resource.
OTHER_PROJECT = "libs/other"

#: The shared thing. Named after the real case: MCPs `packages/db` runs
#: `migrate-test` against one `corvis_test`, on whatever node it lands.
SHARED_DB = "corvis_test"


def _lease(*, resources: tuple[str, ...], node: str = "lavender") -> Lease:
    """Build a lease holding those resources.

    Args:
        resources: What it claims fleet-wide.
        node: The node its environment claim is on.

    Returns:
        The lease.
    """
    return Lease(
        node=node,
        project=DEMO_PROJECT,
        run_id=DEMO_RUN_ID,
        agent="opus-fleet-0904",
        session_id="s",
        acquired_unix=DEMO_NOW,
        expires_unix=DEMO_NOW + 600,
        resources=resources,
    )


class TestDecoding:
    def test_an_absent_list_is_no_resources(self) -> None:
        """The ordinary case. A self-contained suite should not have to say
        that it is self-contained."""
        assert decode_names(None, field="r") == ()

    def test_names_keep_their_declared_order(self) -> None:
        assert decode_names(["b", "a"], field="r") == ("b", "a")

    def test_a_repeated_name_is_carried_once(self) -> None:
        assert decode_names(["a", "a"], field="r") == ("a",)

    def test_a_non_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a list"):
            decode_names("corvis_test", field="r")

    def test_a_non_string_entry_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match=r"r\[1\] must be a string"):
            decode_names(["a", 7], field="r")

    def test_an_empty_name_is_refused(self) -> None:
        """A resource named "" would be contended by every other unnamed one,
        silently serialising projects that share nothing."""
        with pytest.raises(JSONTypeError, match="is empty"):
            decode_names(["a", "  "], field="r")

    def test_a_project_declaring_none_decodes_to_none(self) -> None:
        plan = decode_project_config(
            {"worker_ram_gb": 1.0, "minimum_workers": 2, "expected_minutes": 5}
        )

        assert plan["exclusive_resources"] == ()

    def test_a_project_can_declare_one(self) -> None:
        plan = decode_project_config(
            {
                "worker_ram_gb": 1.0,
                "minimum_workers": 2,
                "expected_minutes": 5,
                "exclusive_resources": [SHARED_DB],
            }
        )

        assert plan["exclusive_resources"] == (SHARED_DB,)

    def test_a_lease_round_trips_its_resources(self) -> None:
        original = _lease(resources=(SHARED_DB,))

        assert decode_lease(encode_lease(original)) == original

    def test_a_lease_written_before_resources_existed_still_decodes(self) -> None:
        """Not back-compat: the field is absent because the record predates
        it, and an absent list means no resources, which is what those runs
        actually held."""
        written = encode_lease(_lease(resources=()))
        del written["resources"]

        assert decode_lease(written)["resources"] == ()


class TestContention:
    def test_no_overlap_is_no_contention(self) -> None:
        assert contended(("a",), ("b",)) == ()

    def test_the_asker_s_order_is_reported(self) -> None:
        """The message is read by the asker, so it lists the asker's own
        resources in the order it named them."""
        assert contended(("x", "y"), ("y", "x")) == ("y", "x")

    def test_a_lease_holding_nothing_contends_with_nothing(self) -> None:
        assert contends(_lease(resources=()), wanted=(SHARED_DB,)) == ()

    def test_a_lease_holding_it_contends(self) -> None:
        assert contends(_lease(resources=(SHARED_DB,)), wanted=(SHARED_DB,)) == (SHARED_DB,)


class TestTheLeaseFile:
    def test_asking_for_nothing_is_never_contended(self, tmp_path: pathlib.Path) -> None:
        """The path every self-contained project takes, and it must not read
        the file to answer."""
        assert leases.contended_by(tmp_path / "absent.json", wanted=(), now_unix=DEMO_NOW) is None

    def test_a_held_resource_is_found_with_its_holder(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(resources=(SHARED_DB,)), now_unix=DEMO_NOW)

        found = leases.contended_by(path, wanted=(SHARED_DB,), now_unix=DEMO_NOW)

        assert found == (_lease(resources=(SHARED_DB,)), (SHARED_DB,))

    def test_an_expired_lease_does_not_hold_its_resource(self, tmp_path: pathlib.Path) -> None:
        """Expiry frees a fleet-wide resource exactly as it frees an
        environment -- a wedge holding one would otherwise stop every database
        suite in the fleet forever."""
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(resources=(SHARED_DB,)), now_unix=DEMO_NOW)

        assert leases.contended_by(path, wanted=(SHARED_DB,), now_unix=DEMO_NOW + 100_000) is None

    def test_a_second_dispatch_on_a_different_node_is_refused(self, tmp_path: pathlib.Path) -> None:
        """THE WHOLE POINT. Different node, different project environment,
        nothing a capacity check would object to -- and it must still be
        refused, because both would migrate one database."""
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(resources=(SHARED_DB,), node="lavender"), now_unix=DEMO_NOW)
        second = Lease(
            node="sedona",
            project=OTHER_PROJECT,
            run_id="other-1",
            agent="opus-other-0904",
            session_id="s2",
            acquired_unix=DEMO_NOW,
            expires_unix=DEMO_NOW + 600,
            resources=(SHARED_DB,),
        )

        with pytest.raises(AppError) as refusal:
            leases.acquire(path, second, now_unix=DEMO_NOW)

        assert refusal.value.code is FleetErrorCode.RESOURCE_HELD

    def test_the_refusal_says_another_node_will_not_help(self, tmp_path: pathlib.Path) -> None:
        """A reader told only "held by X" goes looking for a free node. For a
        fleet-wide resource that search cannot succeed, so the line says so."""
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(resources=(SHARED_DB,)), now_unix=DEMO_NOW)
        second = Lease(
            node="sedona",
            project=OTHER_PROJECT,
            run_id="other-1",
            agent="opus-other-0904",
            session_id="s2",
            acquired_unix=DEMO_NOW,
            expires_unix=DEMO_NOW + 600,
            resources=(SHARED_DB,),
        )

        with pytest.raises(AppError) as refusal:
            leases.acquire(path, second, now_unix=DEMO_NOW)

        assert "no other node is an alternative" in refusal.value.message
        assert SHARED_DB in refusal.value.message
        assert "opus-fleet-0904" in refusal.value.message

    def test_a_different_resource_is_admitted(self, tmp_path: pathlib.Path) -> None:
        """Two singletons are not one singleton."""
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(resources=("corvis_test",)), now_unix=DEMO_NOW)
        second = Lease(
            node="sedona",
            project=OTHER_PROJECT,
            run_id="other-1",
            agent="opus-other-0904",
            session_id="s2",
            acquired_unix=DEMO_NOW,
            expires_unix=DEMO_NOW + 600,
            resources=("some_other_db",),
        )

        leases.acquire(path, second, now_unix=DEMO_NOW)

        assert len(leases.held_leases(path, now_unix=DEMO_NOW)) == 2

    def test_releasing_frees_the_resource_too(self, tmp_path: pathlib.Path) -> None:
        """One lease, one release. There is no second claim to forget."""
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(resources=(SHARED_DB,)), now_unix=DEMO_NOW)

        leases.release(path, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW)

        assert leases.contended_by(path, wanted=(SHARED_DB,), now_unix=DEMO_NOW) is None


def _shared_workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a workspace whose demo project holds an exclusive resource.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the written document.
    """
    document = workspace_document()
    projects = document["projects"]
    assert isinstance(projects, dict)
    plan = projects[DEMO_PROJECT]
    assert isinstance(plan, dict)
    plan["exclusive_resources"] = [SHARED_DB]
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return path


class TestTheCommands:
    def test_a_dispatch_records_what_it_holds(
        self, tmp_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        config_path = _shared_workspace(tmp_path)
        _test_hooks.now = FakeClock(DEMO_NOW)
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})

        run.main(
            [
                _config.CONFIG_FLAG,
                str(config_path),
                run.PROJECT_FLAG,
                DEMO_PROJECT,
                run.AGENT_FLAG,
                "opus-fleet-0904",
                run.SESSION_FLAG,
                "s",
                run.ROOT_FLAG,
                str(repo),
            ]
        )

        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) == _lease(
            resources=(SHARED_DB,)
        )
        assert records.live_runs(loaded.ledger, node="lavender") == 1

    def test_a_second_dispatch_is_refused_without_probing_any_node(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Checked BEFORE the nodes. A fleet-wide resource makes every node
        refuse for the same reason, so probing three to collect three
        identical refusals costs three round trips and produces a message
        shaped like a capacity problem.

        The runner has nothing scripted: reaching it at all is the failure.
        """
        config_path = _shared_workspace(tmp_path)
        _test_hooks.now = FakeClock(DEMO_NOW)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        leases.acquire(loaded.leases, _lease(resources=(SHARED_DB,)), now_unix=DEMO_NOW)
        runner = FakeRun([])
        _test_hooks.run = runner

        with pytest.raises(AppError) as refusal:
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.AGENT_FLAG,
                    "opus-other-0904",
                    run.SESSION_FLAG,
                    "s2",
                    run.ROOT_FLAG,
                    str(tmp_path),
                ]
            )

        assert refusal.value.code is FleetErrorCode.RESOURCE_HELD
        assert runner.calls == []

    def test_preflight_does_not_promise_a_node_the_dispatch_would_refuse(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Without this, preflight answers "yes, lavender, 12 workers" about
        a project fleet-run will turn away for a reason no node can fix --
        worse than not asking, because the reader has a specific wrong
        answer."""
        config_path = _shared_workspace(tmp_path)
        _test_hooks.now = FakeClock(DEMO_NOW)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        leases.acquire(loaded.leases, _lease(resources=(SHARED_DB,)), now_unix=DEMO_NOW)
        _test_hooks.run = FakeRun([ok(""), ok(PROBE_OK)])

        with pytest.raises(AppError) as refusal:
            preflight.main(
                [_config.CONFIG_FLAG, str(config_path), preflight.PROJECT_FLAG, DEMO_PROJECT]
            )

        assert refusal.value.code is FleetErrorCode.RESOURCE_HELD

    def test_a_project_holding_nothing_is_unaffected(
        self, config_path: pathlib.Path, repo: pathlib.Path
    ) -> None:
        """Every project registered today declares no exclusive resource, and
        the new check must be invisible to them."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        payload = prebuilt_archive(config_path, repo)
        _test_hooks.run = FakeRun(dispatch_replies(staging.digest(payload)))

        assert (
            run.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    run.PROJECT_FLAG,
                    DEMO_PROJECT,
                    run.AGENT_FLAG,
                    "opus-fleet-0904",
                    run.SESSION_FLAG,
                    "s",
                    run.ROOT_FLAG,
                    str(repo),
                ]
            )
            == 0
        )
        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) == _lease(
            resources=()
        )
