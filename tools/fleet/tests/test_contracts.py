"""The remaining contracts: node, project, workspace, ledger and feed.

Every decoder is exercised on its own refusals rather than only its happy
path, because each refusal encodes a specific way a fleet goes wrong and a
decoder whose refusals are untested is a decoder that will be relaxed by
somebody who does not know why the check is there.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.budget import NodeBudget
from fleet.contracts.feed import (
    KIND_BY_NAME,
    TERMINAL_KINDS,
    FeedEvent,
    decode_feed_event,
    encode_feed_event,
    is_terminal,
    render_feed_line,
)
from fleet.contracts.ledger import (
    NO_EXIT_CODE,
    LedgerEntry,
    decode_ledger_entry,
    encode_ledger_entry,
    is_live,
)
from fleet.contracts.node import (
    NodeConfig,
    NodeGpu,
    NodeState,
    decode_node_config,
    decode_node_gpu,
    describe_node,
    encode_node_config,
    encode_node_gpu,
)
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

_BUDGET = NodeBudget(
    reserved_cores=2,
    reserved_ram_gb=4.0,
    worker_ram_gb=1.1,
    max_concurrent_runs=2,
    max_disk_gb=20.0,
)

_GPU = NodeGpu(
    model="NVIDIA GeForce GTX 1630",
    vram_mib=4096,
    compute_capability="7.5",
    driver_version="591.86",
)


def _node(*, gpu: NodeGpu | None = _GPU, cores: int = 16) -> NodeConfig:
    """Build a node declaration.

    Args:
        gpu: Its CUDA device, or None for a CPU-only machine.
        cores: Logical processors.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender",
        stage_root="C:/fleet/stage",
        logical_cores=cores,
        ram_gb=32.0,
        gpu=gpu,
        budget=_BUDGET,
    )


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
    )


def _workspace() -> FleetWorkspace:
    """Build a two-node, one-project workspace.

    Returns:
        The workspace.
    """
    return FleetWorkspace(
        nodes={"lavender": _node(), "loki": _node(gpu=None)},
        projects={"services/Model-Trainer": _project()},
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
        The row, typed through its own decoder so the Literal is honest.
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


class TestNodeGpu:
    def test_a_device_survives_encoding(self) -> None:
        assert decode_node_gpu(load_json_str(dump_json_str(encode_node_gpu(_GPU)))) == _GPU

    def test_compute_capability_orders_as_a_version_not_a_float(self) -> None:
        """THE REASON IT IS A STRING, asserted on real architectures.

        Read as floats, 8.10 would sort BELOW 8.9 and a fleet comparing two
        cards would call the newer one older. Held as versions, the two are
        simply distinct values and nothing is tempted to order them
        arithmetically.
        """
        ampere = decode_node_gpu({**encode_node_gpu(_GPU), "compute_capability": "8.9"})
        hopper = decode_node_gpu({**encode_node_gpu(_GPU), "compute_capability": "8.10"})

        assert ampere["compute_capability"] != hopper["compute_capability"]
        assert float(hopper["compute_capability"]) < float(ampere["compute_capability"])

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_node_gpu("GTX 1630")

    def test_a_card_reporting_no_memory_is_a_failed_probe(self) -> None:
        with pytest.raises(JSONTypeError, match="vram_mib must be positive"):
            decode_node_gpu({**encode_node_gpu(_GPU), "vram_mib": 0})


class TestNodeConfig:
    def test_a_node_survives_encoding(self) -> None:
        original = _node()

        assert decode_node_config(load_json_str(dump_json_str(encode_node_config(original)))) == (
            original
        )

    def test_a_cpu_only_node_survives_encoding(self) -> None:
        original = _node(gpu=None)

        assert decode_node_config(load_json_str(dump_json_str(encode_node_config(original)))) == (
            original
        )

    def test_an_absent_gpu_key_is_refused(self) -> None:
        """Absence is indistinguishable from unfilled.

        The difference matters the moment a measurement is pinned to a card,
        so a CPU-only node says so with an explicit null.
        """
        encoded = encode_node_config(_node())
        del encoded["gpu"]

        with pytest.raises(JSONTypeError, match="must declare 'gpu'"):
            decode_node_config(encoded)

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_node_config(["lavender"])

    def test_a_node_with_no_cores_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="logical_cores must be at least 1"):
            decode_node_config({**encode_node_config(_node()), "logical_cores": 0})

    def test_an_empty_stage_root_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="stage_root must not be empty"):
            decode_node_config({**encode_node_config(_node()), "stage_root": ""})

    def test_a_node_with_no_memory_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="ram_gb must be positive"):
            decode_node_config({**encode_node_config(_node()), "ram_gb": 0.0})


class TestDescribeNode:
    def test_it_names_the_architecture_for_a_gpu_node(self) -> None:
        state = NodeState(host="lavender", free_ram_gb=27.4, free_disk_gb=860.0, live_runs=1)

        described = describe_node(_node(), state)

        assert "sm_7.5" in described
        assert "27.4/32.0 GB RAM free" in described
        assert "1 live run(s)" in described

    def test_a_cpu_only_node_says_so(self) -> None:
        state = NodeState(host="loki", free_ram_gb=18.3, free_disk_gb=592.0, live_runs=0)

        assert "cpu-only" in describe_node(_node(gpu=None), state)


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

    def test_a_workspace_with_no_nodes_is_refused(self) -> None:
        """It could dispatch nothing, and every command would blame the caller."""
        with pytest.raises(JSONTypeError, match="declares no nodes"):
            decode_fleet_workspace({**encode_fleet_workspace(_workspace()), "nodes": {}})

    def test_a_workspace_with_no_projects_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="declares no projects"):
            decode_fleet_workspace({**encode_fleet_workspace(_workspace()), "projects": {}})

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
    def test_a_row_survives_encoding(self) -> None:
        original = _entry()

        assert decode_ledger_entry(load_json_str(dump_json_str(encode_ledger_entry(original)))) == (
            original
        )

    def test_a_running_row_is_live(self) -> None:
        assert is_live(_entry(outcome="running"))

    def test_a_finished_row_is_not_live(self) -> None:
        assert not is_live(_entry(outcome="passed"))

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="ledger entry must be a JSON object"):
            decode_ledger_entry("run-1")

    def test_an_unknown_outcome_is_refused(self) -> None:
        """It would read as finished and let a second dispatch onto a full node."""
        with pytest.raises(JSONTypeError, match="is not one of"):
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
            kind="started",
            detail="6 workers",
        )

        assert decode_feed_event(load_json_str(dump_json_str(encode_feed_event(original)))) == (
            original
        )

    def test_every_kind_decodes(self) -> None:
        """The mapping IS the membership test, so nothing can be half-known."""
        for spelling in KIND_BY_NAME:
            decoded = decode_feed_event(
                {
                    "at_unix": 1,
                    "run_id": "r",
                    "node": "n",
                    "project": "p",
                    "kind": spelling,
                    "detail": "",
                }
            )
            assert decoded["kind"] == spelling

    def test_every_terminal_kind_is_a_kind(self) -> None:
        """A terminal kind absent from the mapping could never be decoded."""
        assert set(KIND_BY_NAME) >= TERMINAL_KINDS

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="feed event must be a JSON object"):
            decode_feed_event(7)

    def test_an_unknown_kind_is_refused(self) -> None:
        """It would read as non-terminal and hang a subscriber forever."""
        with pytest.raises(JSONTypeError, match="is not one of"):
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
        ended = FeedEvent(at_unix=1, run_id="r", node="n", project="p", kind="failed", detail="")
        ongoing = FeedEvent(
            at_unix=1, run_id="r", node="n", project="p", kind="phase", detail="lint"
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
                kind="failed",
                detail="exit 1",
            )
        )

        assert line.startswith("FAILED ")
        assert "run-1" in line
        assert "exit 1" in line
