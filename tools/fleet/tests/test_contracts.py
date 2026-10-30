"""The remaining contracts: project, workspace, ledger and feed (machine and node: test_machine.py).

Every decoder is exercised on its own refusals rather than only its happy
path, because each refusal encodes a specific way a fleet goes wrong and a
decoder whose refusals are untested is a decoder that will be relaxed by
somebody who does not know why the check is there.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.feed import (
    TERMINAL_KINDS,
    FeedEvent,
    FeedKind,
    decode_feed_event,
    encode_feed_event,
    is_terminal,
    render_feed_line,
)
from fleet.contracts.ledger import (
    NO_EXIT_CODE,
    LedgerEntry,
    LedgerOutcome,
    decode_ledger_entry,
    encode_ledger_entry,
    is_live,
)
from fleet.contracts.machine import encode_machine
from fleet.contracts.project import (
    ProjectConfig,
    decode_project_config,
    encode_project_config,
    lease_seconds,
)
from fleet.contracts.workspace import (
    FleetWorkspace,
    decode_fleet_workspace,
    encode_fleet_workspace,
    require_node,
    require_project,
)
from tests._contract_fixtures import machine, node
from tests._roster_fixtures import machines_of


def _project(*, minimum_workers: int = 4, expected_minutes: int = 5) -> ProjectConfig:
    """Build a project declaration.

    Args:
        minimum_workers: Fewest workers worth dispatching with.
        expected_minutes: How long the suite takes.

    Returns:
        The project.
    """
    return ProjectConfig(
        worker_ram_gb=1.1,
        minimum_workers=minimum_workers,
        expected_minutes=expected_minutes,
        exclusive_resources=(),
        external_paths=(),
        required_tags=(),
        source=None,
        node=None,
        loopback_only=False,
    )


def _workspace() -> FleetWorkspace:
    """Build a two-node, one-project workspace.

    Returns:
        The workspace.
    """
    lavender = node()
    loki = node(gpu=None, host="loki")
    return FleetWorkspace(
        machines={"lavender": machine("lavender", lavender), "loki": machine("loki", loki)},
        nodes={"lavender": lavender, "loki": loki},
        not_dispatchable={},
        projects={"services/Model-Trainer": _project()},
        data_paths={
            "https://github.com/wagner-austin/API.git": (
                "services/covenant-radar-api/data/external",
            )
        },
        node_local_resources=(),
        node_serve_seconds=1200,
        node_poll_seconds=5,
        native_evidence_seconds=600,
        ledger="ledger.jsonl",
        feed="feed.jsonl",
        leases="leases.json",
    )


def _entry(*, outcome: str = "running", workers: int = 6, started: int = 100) -> LedgerEntry:
    """Build a ledger row.

    Args:
        outcome: How it ended, or ``running``.
        workers: Workers granted.
        started: When it began.

    Returns:
        The row, typed through its own decoder so the outcome word is
        narrowed to its LedgerOutcome member the way a real row's is.
    """
    return decode_ledger_entry(
        {
            "run_id": "run-1",
            "node": "lavender",
            "host": "lavender",
            "project": "services/Model-Trainer",
            "agent": "opus-fleet-0904",
            "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            "started_unix": started,
            "ended_unix": started,
            "outcome": outcome,
            "exit_code": NO_EXIT_CODE,
            "workers": workers,
            "detail": "",
        }
    )


class TestProject:
    def test_a_project_survives_encoding(self) -> None:
        original = _project()

        assert decode_project_config(
            load_json_str(dump_json_str(encode_project_config(original)))
        ) == (original)

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_project_config(42)

    def test_a_zero_worker_cost_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="worker_ram_gb must be positive"):
            decode_project_config({**encode_project_config(_project()), "worker_ram_gb": 0.0})

    def test_a_project_needing_no_workers_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="minimum_workers must be at least 1"):
            decode_project_config({**encode_project_config(_project()), "minimum_workers": 0})

    def test_a_zero_duration_is_refused(self) -> None:
        """It sizes the lease; a zero produces one already expired."""
        with pytest.raises(JSONTypeError, match="expected_minutes must be at least 1"):
            decode_project_config({**encode_project_config(_project()), "expected_minutes": 0})

    def test_required_tags_survive_encoding_in_order(self) -> None:
        encoded = encode_project_config(_project())
        decoded = decode_project_config({**encoded, "required_tags": ["gpu", "linux"]})

        assert decoded["required_tags"] == ("gpu", "linux")
        assert encode_project_config(decoded)["required_tags"] == ["gpu", "linux"]

    def test_a_project_without_required_tags_is_refused(self) -> None:
        """Absent is not empty: a GPU suite that forgot would land on a CPU box."""
        encoded = encode_project_config(_project())
        del encoded["required_tags"]

        with pytest.raises(JSONTypeError, match=r"project\.required_tags is required: \[\]"):
            decode_project_config(encoded)


class TestLeaseSeconds:
    def test_it_rounds_up(self) -> None:
        """60 * 1.005 is 60.3 seconds, and the lease gets 61.

        A lease that expires one second into a running suite is the failure
        the expiry exists to prevent, inverted, so the fraction is always
        spent in the holder's favour.
        """
        assert lease_seconds(_project(expected_minutes=1), slack=1.005) == 61

    def test_slack_scales_the_window(self) -> None:
        assert lease_seconds(_project(expected_minutes=10), slack=2.0) == 1200

    def test_a_slack_that_cannot_cover_a_slower_node_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"slack must be greater than 1\.0"):
            lease_seconds(_project(), slack=1.0)

    def test_a_shrinking_slack_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"slack must be greater than 1\.0"):
            lease_seconds(_project(), slack=0.5)


class TestWorkspace:
    def test_a_workspace_survives_encoding(self) -> None:
        original = _workspace()

        assert decode_fleet_workspace(
            load_json_str(dump_json_str(encode_fleet_workspace(original)))
        ) == (original)

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="workspace must be a JSON object"):
            decode_fleet_workspace([])

    def test_a_workspace_with_no_machines_is_refused(self) -> None:
        """It could dispatch nothing, and every command would blame the caller."""
        with pytest.raises(JSONTypeError, match="declares no machines"):
            decode_fleet_workspace({**encode_fleet_workspace(_workspace()), "machines": {}})

    def test_a_roster_where_no_machine_declares_capacity_is_refused(self) -> None:
        original = _workspace()
        original["machines"] = {"diphtheria": machine("diphtheria", None, held="the hub")}

        with pytest.raises(JSONTypeError, match="no machine declares what dispatch may take"):
            decode_fleet_workspace(encode_fleet_workspace(original))

    def test_a_workspace_with_no_projects_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="declares no projects"):
            decode_fleet_workspace({**encode_fleet_workspace(_workspace()), "projects": {}})

    def test_excluding_nothing_is_a_real_answer_unlike_declaring_no_nodes(self) -> None:
        """A roster in which every machine takes work excludes none, which is
        a legitimate state, unlike a roster in which none does."""
        assert decode_fleet_workspace(encode_fleet_workspace(_workspace()))["not_dispatchable"] == (
            {}
        )

    def test_each_reason_a_machine_takes_no_work_is_its_exclusion(self) -> None:
        """A machine that declares no capacity says why in its ``dispatch``,
        and a declared node held out says why in ``held``; both are what
        ``not_dispatchable`` names (MCPs board task e447de72), and neither
        node is dispatched to."""
        hub = "the hub the dispatcher itself runs on; it shares the .venv"
        retired = "its WSL VM holds the memory the Windows lane measures"
        original = _workspace()
        original["machines"]["diphtheria"] = machine("diphtheria", None, held=hub)
        original["machines"]["loki"] = machine("loki", node(gpu=None, host="loki"), held=retired)

        decoded = decode_fleet_workspace(
            load_json_str(dump_json_str(encode_fleet_workspace(original)))
        )

        assert decoded["not_dispatchable"] == {"diphtheria": hub, "loki": retired}
        assert sorted(decoded["nodes"]) == ["lavender", "loki"]
        assert decoded["nodes"]["loki"]["enabled"] is False
        assert decoded["machines"]["loki"]["held"] == retired

    def test_a_blank_reason_is_refused(self) -> None:
        """It says no more than leaving the machine out entirely, which is
        exactly the ambiguity the field was added to remove."""
        original = _workspace()
        encoded = encode_fleet_workspace(original)
        machines_of(encoded)["diphtheria"] = {
            **encode_machine(machine("diphtheria", None, held="x")),
            "dispatch": "",
        }

        with pytest.raises(JSONTypeError, match="dispatch must be a non-empty string"):
            decode_fleet_workspace(encoded)

    def test_an_unknown_node_names_the_declared_ones(self) -> None:
        with pytest.raises(AppError) as excinfo:
            require_node(_workspace(), "sedona")

        assert excinfo.value.code is FleetErrorCode.WORKSPACE_NODE_UNKNOWN
        assert "lavender" in excinfo.value.message
        assert "loki" in excinfo.value.message

    def test_a_declared_node_is_returned(self) -> None:
        assert require_node(_workspace(), "loki")["gpu"] is None

    def test_an_unknown_project_names_the_declared_ones(self) -> None:
        with pytest.raises(AppError) as excinfo:
            require_project(_workspace(), "tools/hpc3")

        assert excinfo.value.code is FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN
        assert "services/Model-Trainer" in excinfo.value.message

    def test_a_declared_project_is_returned(self) -> None:
        assert require_project(_workspace(), "services/Model-Trainer")["minimum_workers"] == 4


class TestLedgerEntry:
    def test_every_outcome_survives_encoding_as_its_word(self) -> None:
        for outcome in LedgerOutcome:
            original = _entry(outcome=outcome.value)
            encoded = encode_ledger_entry(original)

            assert encoded["outcome"] == outcome.value
            assert type(encoded["outcome"]) is str
            decoded = decode_ledger_entry(load_json_str(dump_json_str(encoded)))
            assert decoded == original
            assert decoded["outcome"] is outcome

    def test_a_running_row_is_live(self) -> None:
        assert is_live(_entry(outcome="running"))

    def test_a_finished_row_is_not_live(self) -> None:
        assert not is_live(_entry(outcome="passed"))

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="ledger entry must be a JSON object"):
            decode_ledger_entry("run-1")

    def test_an_unknown_outcome_is_refused(self) -> None:
        """It would read as finished and let a second dispatch onto a full node."""
        with pytest.raises(
            JSONTypeError,
            match=r"^ledger outcome 'probably-fine' is not one of "
            r"cancelled, failed, lost, passed, refused, running; ",
        ):
            decode_ledger_entry({**encode_ledger_entry(_entry()), "outcome": "probably-fine"})

    def test_a_row_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="cannot"):
            decode_ledger_entry({**encode_ledger_entry(_entry()), "ended_unix": 1})

    def test_a_negative_worker_count_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="workers must not be negative"):
            decode_ledger_entry({**encode_ledger_entry(_entry()), "workers": -1})

    def test_a_refused_dispatch_records_zero_workers(self) -> None:
        assert _entry(outcome="refused", workers=0)["workers"] == 0


class TestFeedEvent:
    def test_an_event_survives_encoding(self) -> None:
        original = FeedEvent(
            at_unix=100,
            run_id="run-1",
            node="lavender",
            project="services/Model-Trainer",
            kind=FeedKind.STARTED,
            detail="6 workers",
        )

        encoded = encode_feed_event(original)
        assert encoded["kind"] == "started"
        assert type(encoded["kind"]) is str
        assert decode_feed_event(load_json_str(dump_json_str(encoded))) == original

    def test_every_kind_decodes_to_its_member(self) -> None:
        """The enum IS the membership test, so nothing can be half-known."""
        for kind in FeedKind:
            decoded = decode_feed_event(
                {
                    "at_unix": 1,
                    "run_id": "r",
                    "node": "n",
                    "project": "p",
                    "kind": kind.value,
                    "detail": "",
                }
            )
            assert decoded["kind"] is kind

    def test_the_terminal_kinds_are_the_five_that_end_a_run(self) -> None:
        """Terminality is a judgement per kind, so the set is pinned whole."""
        assert {kind.value for kind in TERMINAL_KINDS} == {
            "refused",
            "passed",
            "failed",
            "cancelled",
            "lost",
        }

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="feed event must be a JSON object"):
            decode_feed_event(7)

    def test_an_unknown_kind_is_refused(self) -> None:
        """It would read as non-terminal and hang a subscriber forever."""
        with pytest.raises(
            JSONTypeError,
            match=r"^feed event kind 'finished-ish' is not one of cancelled, failed, heartbeat, "
            r"leased, lost, passed, phase, refused, staged, started; ",
        ):
            decode_feed_event(
                {
                    "at_unix": 1,
                    "run_id": "r",
                    "node": "n",
                    "project": "p",
                    "kind": "finished-ish",
                    "detail": "",
                }
            )

    def test_terminal_kinds_end_a_run_and_others_do_not(self) -> None:
        ended = FeedEvent(
            at_unix=1, run_id="r", node="n", project="p", kind=FeedKind.FAILED, detail=""
        )
        ongoing = FeedEvent(
            at_unix=1, run_id="r", node="n", project="p", kind=FeedKind.PHASE, detail="lint"
        )

        assert is_terminal(ended)
        assert not is_terminal(ongoing)

    def test_a_line_leads_with_its_kind(self) -> None:
        """A subscriber's grep is written against the kind.

        A leading timestamp would make every alternation start with a
        wildcard.
        """
        line = render_feed_line(
            FeedEvent(
                at_unix=1,
                run_id="run-1",
                node="lavender",
                project="services/Model-Trainer",
                kind=FeedKind.FAILED,
                detail="exit 1",
            )
        )

        assert line.startswith("FAILED ")
        assert "run-1" in line
        assert "exit 1" in line
