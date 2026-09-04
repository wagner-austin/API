"""The claim that stops two dispatches sharing one environment.

THE REGRESSION TEST FOR THE INCIDENT is
:meth:`TestAcquire.test_a_second_dispatch_for_one_project_on_one_node_is_refused`.
Everything else here supports it: the key that decides what excludes what, the
expiry that stops a wedge holding a project forever, and the release that must
refuse rather than shrug.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.lease import (
    Lease,
    claims,
    decode_lease,
    describe_lease,
    encode_lease,
    is_expired,
)
from fleet.core import _test_hooks, leases
from tests.conftest import FakeClock

_NOW = 1_757_000_000


def _lease(
    *,
    node: str = "lavender",
    project: str = "services/Model-Trainer",
    run_id: str = "run-1",
    agent: str = "opus-fleet-0904",
    acquired: int = _NOW,
    expires: int = _NOW + 600,
) -> Lease:
    """Build a lease, letting a test name only the field it is about.

    Args:
        node: The node's workspace name.
        project: Repo-relative project path.
        run_id: The dispatch holding it.
        agent: Board label of the dispatching session.
        acquired: When it was taken.
        expires: When it lapses.

    Returns:
        The lease.
    """
    return Lease(
        node=node,
        project=project,
        run_id=run_id,
        agent=agent,
        session_id="acc774c0-3bc3-4cce-9dda-c7a12fb99519",
        acquired_unix=acquired,
        expires_unix=expires,
    )


def _write_leases(path: pathlib.Path, *held: Lease) -> None:
    """Put leases on disk the way the store writes them.

    Args:
        path: The lease file.
        held: The leases to record.
    """
    path.write_text(dump_json_str([encode_lease(entry) for entry in held]), encoding="utf-8")


class TestClaims:
    def test_a_lease_claims_its_own_pair(self) -> None:
        held = _lease(node="lavender", project="services/Model-Trainer")

        assert claims(held, node="lavender", project="services/Model-Trainer")

    def test_a_lease_on_another_node_claims_nothing_here(self) -> None:
        held = _lease(node="loki", project="services/Model-Trainer")

        assert not claims(held, node="lavender", project="services/Model-Trainer")

    def test_a_lease_on_another_project_claims_nothing_here(self) -> None:
        held = _lease(node="lavender", project="tools/hpc3")

        assert not claims(held, node="lavender", project="services/Model-Trainer")

    def test_two_pairs_that_a_string_key_would_alias_stay_distinct(self) -> None:
        """THE DEFECT THIS FUNCTION REPLACED.

        The first implementation built ``f"{node}::{project}"`` and asserted
        in its docstring that two pairs could not collide. They can:
        ``("a::b", "c")`` and ``("a", "b::c")`` both render ``a::b::c``, so
        one project's dispatch would have excluded another's. Comparing
        fields makes it unrepresentable rather than unlikely.
        """
        held = _lease(node="a::b", project="c")

        assert claims(held, node="a::b", project="c")
        assert not claims(held, node="a", project="b::c")


class TestExpiry:
    def test_a_lease_before_its_expiry_is_held(self) -> None:
        assert not is_expired(_lease(expires=_NOW + 1), now_unix=_NOW)

    def test_the_boundary_frees_the_resource(self) -> None:
        """Reaching the expiry counts as expired.

        The boundary has to fall one way, and the direction that frees a
        resource is the one that cannot deadlock.
        """
        assert is_expired(_lease(expires=_NOW), now_unix=_NOW)


class TestDescribeLease:
    def test_it_names_the_holder_and_the_time_left(self) -> None:
        described = describe_lease(_lease(expires=_NOW + 300), now_unix=_NOW)

        assert "opus-fleet-0904" in described
        assert "300s remaining" in described
        assert "services/Model-Trainer" in described

    def test_an_expired_lease_says_how_long_ago(self) -> None:
        described = describe_lease(_lease(acquired=_NOW - 600, expires=_NOW - 30), now_unix=_NOW)

        assert "expired 30s ago" in described


class TestDecode:
    def test_a_lease_survives_encoding(self) -> None:
        original = _lease()

        assert decode_lease(load_json_str(dump_json_str(encode_lease(original)))) == original

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_lease("lavender")

    def test_a_lease_that_expires_before_it_is_taken_is_refused(self) -> None:
        """Every other session would read the resource as free."""
        with pytest.raises(JSONTypeError, match="expires at"):
            decode_lease(encode_lease(_lease(acquired=_NOW, expires=_NOW - 1)))

    def test_a_zero_length_lease_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="expires at"):
            decode_lease(encode_lease(_lease(acquired=_NOW, expires=_NOW)))


class TestReadLeases:
    def test_an_absent_file_is_no_leases(self, tmp_path: pathlib.Path) -> None:
        """The first dispatch in a workspace has no history and is not refused."""
        assert leases.read_leases(tmp_path / "leases.json") == ()

    def test_expired_leases_are_read_but_not_held(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "leases.json"
        _write_leases(
            path, _lease(run_id="old", acquired=_NOW - 600, expires=_NOW - 1), _lease(run_id="live")
        )

        assert len(leases.read_leases(path)) == 2
        assert [held["run_id"] for held in leases.held_leases(path, now_unix=_NOW)] == ["live"]

    def test_a_file_that_is_not_a_list_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Reading it as empty would hand out a resource somebody holds."""
        path = tmp_path / "leases.json"
        path.write_text('{"node": "lavender"}', encoding="utf-8")

        with pytest.raises(AppError) as excinfo:
            leases.read_leases(path)

        assert excinfo.value.code is FleetErrorCode.LEASE_NOT_HELD


class TestAcquire:
    def test_a_second_dispatch_for_one_project_on_one_node_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        """THE REGRESSION TEST FOR THE 2026-09-04 INCIDENT.

        Two sessions ran in one project's environment and the second's
        `poetry sync` deleted the first's interpreter mid-run. This is the
        refusal that makes that unrepresentable, and it names the holder so
        the blocked session knows who to talk to.
        """
        path = tmp_path / "leases.json"
        clock = FakeClock(_NOW)
        _test_hooks.now = clock
        leases.acquire(path, _lease(run_id="first"), now_unix=_NOW)

        with pytest.raises(AppError) as excinfo:
            leases.acquire(path, _lease(run_id="second"), now_unix=_NOW)

        assert excinfo.value.code is FleetErrorCode.LEASE_HELD
        assert "first" in excinfo.value.message
        assert "opus-fleet-0904" in excinfo.value.message

    def test_two_projects_on_one_node_do_not_exclude_each_other(
        self, tmp_path: pathlib.Path
    ) -> None:
        """They have two .venv directories and cannot corrupt each other.

        Serialising them would give up most of what a 20-core node is for.
        """
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(run_id="a", project="libs/platform_core"), now_unix=_NOW)

        leases.acquire(path, _lease(run_id="b", project="tools/hpc3"), now_unix=_NOW)

        assert len(leases.held_leases(path, now_unix=_NOW)) == 2

    def test_one_project_on_two_nodes_does_not_exclude_itself(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(run_id="a", node="lavender"), now_unix=_NOW)

        leases.acquire(path, _lease(run_id="b", node="loki"), now_unix=_NOW)

        assert len(leases.held_leases(path, now_unix=_NOW)) == 2

    def test_an_expired_holder_does_not_block_and_is_dropped(self, tmp_path: pathlib.Path) -> None:
        """A wedge must not hold a project forever."""
        path = tmp_path / "leases.json"
        _write_leases(path, _lease(run_id="wedged", acquired=_NOW - 600, expires=_NOW - 1))

        leases.acquire(path, _lease(run_id="new"), now_unix=_NOW)

        assert [held["run_id"] for held in leases.held_leases(path, now_unix=_NOW)] == ["new"]


class TestFindHolder:
    def test_a_free_pair_has_no_holder(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "leases.json"
        _write_leases(path, _lease(node="loki"))

        assert (
            leases.find_holder(
                path, node="lavender", project="services/Model-Trainer", now_unix=_NOW
            )
            is None
        )


class TestRelease:
    def test_releasing_frees_the_pair(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "leases.json"
        leases.acquire(path, _lease(run_id="first"), now_unix=_NOW)

        leases.release(path, run_id="first", now_unix=_NOW)

        assert leases.held_leases(path, now_unix=_NOW) == ()

    def test_releasing_a_lease_nobody_holds_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Both ways to reach this are faults worth seeing.

        Releasing twice means the caller lost track of its own dispatch;
        releasing an expired one means the run outlived its declared window
        and another dispatch may already be inside the environment.
        """
        path = tmp_path / "leases.json"
        _write_leases(path, _lease(run_id="other"))

        with pytest.raises(AppError) as excinfo:
            leases.release(path, run_id="mine", now_unix=_NOW)

        assert excinfo.value.code is FleetErrorCode.LEASE_NOT_HELD


class TestFindByRun:
    def test_it_finds_the_lease_one_dispatch_holds(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "leases.json"
        _write_leases(path, _lease(run_id="mine"))

        found = leases.find_by_run(path, run_id="mine", now_unix=_NOW)

        assert found == _lease(run_id="mine")

    def test_it_scans_past_leases_other_runs_hold(self, tmp_path: pathlib.Path) -> None:
        """The loop must keep going, not stop at the first non-match.

        With one lease in the file both a working scan and a broken one
        return the same answer, so the ordering here is the test.
        """
        path = tmp_path / "leases.json"
        _write_leases(
            path,
            _lease(run_id="theirs", project="tools/hpc3"),
            _lease(run_id="mine", project="services/Model-Trainer"),
        )

        found = leases.find_by_run(path, run_id="mine", now_unix=_NOW)

        assert found == _lease(run_id="mine", project="services/Model-Trainer")

    def test_a_dispatch_whose_lease_lapsed_holds_none(self, tmp_path: pathlib.Path) -> None:
        """The wedge case, and the one `fleet-cancel` must still act on."""
        path = tmp_path / "leases.json"
        _write_leases(path, _lease(run_id="wedged", acquired=_NOW - 600, expires=_NOW - 1))

        assert leases.find_by_run(path, run_id="wedged", now_unix=_NOW) is None
