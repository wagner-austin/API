"""The refusal that is the product.

Every case here is a node that would accept work it cannot finish, which is
the failure mode the package was written after: on 2026-09-04 nothing refused
two overlapping suites and they held 77.9 GB of commit doing nothing.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import (
    LiveLoad,
    NodeConfig,
    NodeGpu,
    NodePlatform,
    NodeState,
    SliceMemory,
)
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import NodeTag
from fleet.core.capacity import assess, first_fit, job_ceiling, plan_dispatch, room_for_any
from tests.conftest import IDLE

#: What a windows node with no other declaration carries, and a linux one.
WINDOWS_TAGS = frozenset({NodeTag.WINDOWS})
LINUX_TAGS = frozenset({NodeTag.LINUX})

#: lavender's card as fleet.json declares it, for the nodes that carry one.
GTX_1630 = NodeGpu(
    model="NVIDIA GeForce GTX 1630",
    vram_mib=4096,
    compute_capability="7.5",
    driver_version="591.86",
)


def _node(
    *,
    host: str = "lavender",
    cores: int = 16,
    reserved_cores: int = 2,
    reserved_ram_gb: float = 4.0,
    max_disk_gb: float = 20.0,
    platform: NodePlatform = NodePlatform.WINDOWS,
    gpu: NodeGpu | None = None,
    checks_at_once: int | None = None,
) -> NodeConfig:
    """Build a node declaration.

    Args:
        host: SSH alias.
        cores: Logical processors.
        reserved_cores: Cores left for the owner.
        reserved_ram_gb: Memory left for the owner.
        max_disk_gb: Disk reserved for staged trees.
        platform: The node's dialect.
        gpu: Its CUDA device, or None for a CPU-only node.
        checks_at_once: Budgeted checks its CPUs carry at once, or None.

    Returns:
        The node.
    """
    return NodeConfig(
        host=host,
        platform=platform,
        stage_root="C:/fleet/stage",
        logical_cores=cores,
        ram_gb=32.0,
        gpu=gpu,
        enabled=True,
        test_database=False,
        rust=None,
        cxx=None,
        docker=None,
        stack=None,
        elevated=False,
        wsl_host=None,
        budget=NodeBudget(
            reserved_cores=reserved_cores,
            reserved_ram_gb=reserved_ram_gb,
            worker_ram_gb=1.1,
            max_disk_gb=max_disk_gb,
            checks_at_once=checks_at_once,
        ),
    )


def _state(
    *,
    host: str = "lavender",
    free_ram_gb: float = 27.0,
    free_disk_gb: float = 800.0,
    live: LiveLoad = IDLE,
    ci_slice: SliceMemory | None = None,
) -> NodeState:
    """Build a probed state.

    Args:
        host: The node it came from.
        free_ram_gb: Memory free.
        free_disk_gb: Disk free.
        live: What the node's live fleet runs hold.
        ci_slice: Its runners.slice reading, or None for a node with none.

    Returns:
        The state.
    """
    return NodeState(
        host=host,
        free_ram_gb=free_ram_gb,
        free_disk_gb=free_disk_gb,
        live=live,
        ci_slice=ci_slice,
    )


def _project(
    *,
    minimum_workers: int = 4,
    worker_ram_gb: float = 1.1,
    required_tags: tuple[NodeTag, ...] = (),
) -> ProjectConfig:
    """Build a project declaration.

    Args:
        minimum_workers: Fewest workers worth dispatching with.
        worker_ram_gb: Memory one worker of this suite holds.
        required_tags: What the suite requires of a node.

    Returns:
        The project.
    """
    return ProjectConfig(
        worker_ram_gb=worker_ram_gb,
        minimum_workers=minimum_workers,
        expected_minutes=5,
        exclusive_resources=(),
        external_paths=(),
        required_tags=required_tags,
        source=None,
    )


class TestAssess:
    def test_a_healthy_node_accepts_and_names_no_code(self) -> None:
        verdict = assess(_node(), _state(), _project(), WINDOWS_TAGS)

        assert verdict["code"] is None
        assert verdict["reason"] == ""
        assert verdict["workers"] == 7

    def test_a_node_without_room_to_stage_is_refused(self) -> None:
        verdict = assess(
            _node(max_disk_gb=20.0), _state(free_disk_gb=5.0), _project(), WINDOWS_TAGS
        )

        assert verdict["code"] is FleetErrorCode.NODE_DISK_EXHAUSTED
        assert "reserves 20 GB" in verdict["reason"]

    def test_a_node_whose_owner_is_using_it_is_refused(self) -> None:
        verdict = assess(_node(), _state(free_ram_gb=3.0), _project(), WINDOWS_TAGS)

        assert verdict["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert "somebody is on this machine" in verdict["reason"]

    def test_a_node_whose_ci_slice_is_at_its_high_names_ci_not_the_owner(self) -> None:
        """lavender-wsl at 18:2xZ on 2026-09-29 (MCPs board task 5d6e57e7):
        8.2 GB available against its reservation, runners.slice at 16.0 of
        16.0 GB. Just under the line still counts, since the kernel keeps it
        there; well under it is the owner's case again."""
        at_high = SliceMemory(current_gb=15.9, high_gb=16.0)
        below = SliceMemory(current_gb=12.0, high_gb=16.0)
        node = _node(reserved_ram_gb=12.0)

        held = assess(node, _state(free_ram_gb=8.2, ci_slice=at_high), _project(), WINDOWS_TAGS)
        light = assess(node, _state(free_ram_gb=8.2, ci_slice=below), _project(), WINDOWS_TAGS)

        assert held["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert held["reason"].endswith(
            "Nothing is left for a dispatch; its CI runners hold it: runners.slice is at 15.9 "
            "GB of its 16.0 GB memory.high, so the lane waits for a CI job to end."
        )
        assert light["reason"].endswith("somebody is on this machine.")

    def test_a_node_its_own_runs_fill_names_them_and_one_its_owner_fills_does_not(self) -> None:
        """lavender-wsl in fleet_status at 09:09Z on 2026-10-03 (MCPs board
        task 939ec5c7): 18.8 GB free, two runs holding seven workers and
        15.5 GB, and the reason said somebody was on the machine. With those
        runs gone it would have room, so they are named; when even an idle
        node would have none, the owner still is."""
        runs = LiveLoad(runs=2, workers=7, ram_gb=15.5)
        node = _node(reserved_cores=8, reserved_ram_gb=6.0)

        own = assess(node, _state(free_ram_gb=18.8, live=runs), _project(), WINDOWS_TAGS)
        owner = assess(node, _state(free_ram_gb=5.0, live=runs), _project(), WINDOWS_TAGS)

        assert own["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert own["reason"].endswith(
            "Nothing is left for a dispatch; its own fleet runs hold the rest, so it takes the "
            "next job when one of them ends."
        )
        assert owner["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert owner["reason"].endswith("somebody is on this machine.")

    def test_a_node_too_small_for_the_suite_is_refused(self) -> None:
        """THE sedona CASE. 6 workers afforded, 8 declared as the minimum.

        Dispatching anyway runs the suite at a fraction of its workers until
        its own lease expires underneath it.
        """
        verdict = assess(
            _node(cores=20), _state(free_ram_gb=11.4), _project(minimum_workers=8), WINDOWS_TAGS
        )

        assert verdict["code"] is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "affords 6 worker(s)" in verdict["reason"]
        assert "minimum of 8" in verdict["reason"]

    def test_a_node_missing_a_required_tag_is_refused_before_capacity(self) -> None:
        """A CPU-only linux box with every byte free is still the wrong machine."""
        verdict = assess(
            _node(host="diphtheria", platform=NodePlatform.LINUX),
            _state(host="diphtheria"),
            _project(required_tags=(NodeTag.GPU, NodeTag.WINDOWS)),
            LINUX_TAGS,
        )

        assert verdict["code"] is FleetErrorCode.NODE_LACKS_TAG
        assert verdict["workers"] == 0
        assert verdict["reason"].startswith("diphtheria lacks gpu, windows: ")
        assert "requires gpu, windows and this node carries linux" in verdict["reason"]
        assert "or this one once the missing tool is installed" in verdict["reason"]

    def test_the_tag_refusal_names_only_the_tags_missing(self) -> None:
        verdict = assess(
            _node(host="loki"),
            _state(host="loki"),
            _project(required_tags=(NodeTag.WINDOWS, NodeTag.GPU)),
            WINDOWS_TAGS,
        )

        assert verdict["code"] is FleetErrorCode.NODE_LACKS_TAG
        assert verdict["reason"].startswith("loki lacks gpu: ")

    def test_a_tool_tag_is_judged_on_what_the_runner_carries(self) -> None:
        """pendragon on 2026-10-02 (MCPs board task 939ec5c7): grandma-api
        needs ffmpeg, so a runner whose probe did not find it is refused that
        one project, and the same node with ffmpeg found is weighed on its
        capacity like any other."""
        grandma = _project(required_tags=(NodeTag.WINDOWS, NodeTag.FFMPEG))
        node = _node(host="pendragon")

        lacking = assess(node, _state(host="pendragon"), grandma, WINDOWS_TAGS)
        found = assess(node, _state(host="pendragon"), grandma, WINDOWS_TAGS | {NodeTag.FFMPEG})

        assert lacking["code"] is FleetErrorCode.NODE_LACKS_TAG
        assert lacking["reason"].startswith("pendragon lacks ffmpeg: ")
        assert found["code"] is None
        assert found["workers"] == 7

    def test_a_node_carrying_every_required_tag_is_weighed_on_capacity(self) -> None:
        project = _project(required_tags=(NodeTag.GPU, NodeTag.WINDOWS))
        carried = frozenset({NodeTag.WINDOWS, NodeTag.GPU})

        assert assess(_node(gpu=GTX_1630), _state(), project, carried)["workers"] == 7
        full = assess(_node(gpu=GTX_1630), _state(free_ram_gb=3.0), project, carried)
        assert full["code"] is FleetErrorCode.NODE_OWNER_RESERVED

    def test_a_project_requiring_nothing_takes_any_platform(self) -> None:
        linux = _node(platform=NodePlatform.LINUX)
        assert assess(linux, _state(), _project(), LINUX_TAGS)["workers"] == 7

    def test_the_project_cost_overrides_the_node_default(self) -> None:
        """What a worker costs is a property of the suite, not the machine:
        at 0.2 GB a worker, 23 GB affords 115, under 198 spare cores."""
        heavy = assess(_node(cores=400), _state(), _project(), WINDOWS_TAGS)
        light = assess(_node(cores=400), _state(), _project(worker_ram_gb=0.2), WINDOWS_TAGS)

        assert heavy["workers"] == 20
        assert light["workers"] == 115


class TestLivePools:
    """MCPs board task 939ec5c7: what a node's live runs hold comes off its
    pools, and one job takes at most half the spare cores, so a node that
    once took one run at a time now takes as many as its cores and memory
    hold."""

    def test_one_job_takes_at_most_half_the_spare_cores(self) -> None:
        """16 cores, 2 reserved: memory affords 20 workers, the cores 14, the
        ceiling 7, which leaves the other 7 for a second job."""
        assert job_ceiling(_node(), _project()) == 7
        assert assess(_node(), _state(), _project(), WINDOWS_TAGS)["workers"] == 7

    def test_an_odd_spare_rounds_the_ceiling_up(self) -> None:
        assert job_ceiling(_node(cores=17), _project(minimum_workers=1)) == 8

    def test_a_minimum_above_the_ceiling_raises_it(self) -> None:
        """A suite that cannot run on fewer than 10 workers is granted 10,
        and the cores pool still bounds the sum."""
        verdict = assess(_node(), _state(), _project(minimum_workers=10), WINDOWS_TAGS)

        assert job_ceiling(_node(), _project(minimum_workers=10)) == 10
        assert verdict["workers"] == 10

    def test_a_second_job_fits_beside_a_live_one(self) -> None:
        """The first job's 7 workers and 7.7 GB come off; 7 cores and
        27.0 - 7.7 - 4.0 = 15.3 GB (13 workers) are left, so 7 again."""
        live = LiveLoad(runs=1, workers=7, ram_gb=7.7)

        assert assess(_node(), _state(live=live), _project(), WINDOWS_TAGS)["workers"] == 7

    def test_live_memory_shrinks_the_grant(self) -> None:
        """16.0 - 7.7 - 4.0 = 4.3 GB: three 1.1 GB workers, under the 7
        cores still free."""
        live = LiveLoad(runs=1, workers=7, ram_gb=7.7)

        verdict = assess(
            _node(), _state(free_ram_gb=16.0, live=live), _project(minimum_workers=2), WINDOWS_TAGS
        )

        assert verdict["code"] is None
        assert verdict["workers"] == 3

    def test_a_short_grant_names_what_the_live_runs_hold(self) -> None:
        live = LiveLoad(runs=1, workers=7, ram_gb=7.7)

        verdict = assess(_node(), _state(free_ram_gb=16.0, live=live), _project(), WINDOWS_TAGS)

        assert verdict["code"] is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert verdict["reason"] == (
            "lavender affords 3 worker(s) for a suite that declares a minimum of 4: 16.0 GB "
            "free, 1.1 GB per worker, 4.0 GB reserved for the node's owner, and its 1 live "
            "fleet run(s) hold 7 worker(s) and 7.7 GB. Dispatching anyway would run a suite at "
            "a fraction of its workers until its own lease expired underneath it."
        )

    def test_live_runs_holding_every_spare_core_leave_nothing(self) -> None:
        live = LiveLoad(runs=2, workers=14, ram_gb=15.4)

        verdict = assess(_node(), _state(free_ram_gb=60.0, live=live), _project(), WINDOWS_TAGS)

        assert verdict["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert verdict["workers"] == 0
        assert (
            "16 cores against 2 reserved, and its 2 live fleet run(s) hold 14 worker(s) and "
            "15.4 GB. Nothing is left for a dispatch" in verdict["reason"]
        )


class TestChecksAtOnce:
    """MCPs board task a85ef09e: serendipity, 8 cores with 4 reserved, ran
    MCPs/sms-gateway in 162 s alone and in 306 s beside a second check the
    pool admitted, 2 + 2 workers inside its 4 cores, so its budget declares
    how many checks its CPUs carry at once."""

    def test_a_node_at_its_check_count_takes_no_second_however_roomy_its_pool(self) -> None:
        node = _node(host="serendipity", cores=8, reserved_cores=4, checks_at_once=1)
        live = LiveLoad(runs=1, workers=2, ram_gb=2.2)

        verdict = assess(
            node,
            _state(host="serendipity", free_ram_gb=11.6, live=live),
            _project(minimum_workers=1),
            WINDOWS_TAGS,
        )

        assert verdict == {
            "workers": 0,
            "code": FleetErrorCode.NODE_OWNER_RESERVED,
            "reason": (
                "serendipity's CPUs carry 1 budgeted check(s) at once across its runners, and 1 "
                "are live or launching on it now, and its 1 live fleet run(s) hold 2 worker(s) "
                "and 2.2 GB. A check runs wider than its test workers, so a second would push "
                "both past their budget; it takes the next job when one of them ends."
            ),
        }

    def test_a_node_under_its_check_count_is_weighed_on_its_pools(self) -> None:
        node = _node(host="serendipity", cores=8, reserved_cores=4, checks_at_once=1)

        verdict = assess(
            node,
            _state(host="serendipity", free_ram_gb=11.6),
            _project(minimum_workers=1),
            WINDOWS_TAGS,
        )

        assert verdict == {"workers": 2, "code": None, "reason": ""}

    def test_a_node_with_no_check_count_takes_a_second_its_pool_holds(self) -> None:
        node = _node(host="serendipity", cores=8, reserved_cores=4)
        live = LiveLoad(runs=1, workers=2, ram_gb=2.2)

        verdict = assess(
            node,
            _state(host="serendipity", free_ram_gb=11.6, live=live),
            _project(minimum_workers=1),
            WINDOWS_TAGS,
        )

        assert verdict == {"workers": 2, "code": None, "reason": ""}


class TestPlanDispatch:
    def test_it_returns_the_worker_count_when_the_node_accepts(self) -> None:
        assert plan_dispatch(_node(), _state(), _project(), WINDOWS_TAGS) == 7

    def test_it_raises_the_verdict_s_own_code(self) -> None:
        with pytest.raises(AppError) as excinfo:
            plan_dispatch(_node(), _state(free_ram_gb=3.0), _project(), WINDOWS_TAGS)

        assert excinfo.value.code is FleetErrorCode.NODE_OWNER_RESERVED


class TestFirstFit:
    def test_it_picks_the_node_that_affords_the_most(self) -> None:
        """Not the first that fits: the fleet's nodes differ by over 2x."""
        candidates = (
            ("sedona", _node(host="sedona", cores=20), _state(host="sedona", free_ram_gb=11.4)),
            ("loki", _node(host="loki", cores=16), _state(host="loki", free_ram_gb=27.0)),
        )

        assert first_fit(candidates, _project()) == ("loki", 7)

    def test_a_tie_keeps_the_earlier_candidate(self) -> None:
        """Workspace order is a tie-break a person controls."""
        candidates = (
            ("alpha", _node(host="alpha"), _state(host="alpha")),
            ("beta", _node(host="beta"), _state(host="beta")),
        )

        assert first_fit(candidates, _project())[0] == "alpha"

    def test_a_refusing_node_is_skipped_for_one_that_accepts(self) -> None:
        candidates = (
            ("busy", _node(host="busy"), _state(host="busy", free_ram_gb=3.0)),
            ("free", _node(host="free"), _state(host="free")),
        )

        assert first_fit(candidates, _project())[0] == "free"

    def test_when_nothing_fits_every_reason_is_carried(self) -> None:
        """ "No room" and "one is full, one has no disk" want different actions.

        A message naming only the first refusal sends the reader to the wrong
        machine.
        """
        candidates = (
            ("busy", _node(host="busy"), _state(host="busy", free_ram_gb=3.0)),
            ("fullup", _node(host="fullup"), _state(host="fullup", free_disk_gb=1.0)),
        )

        with pytest.raises(AppError) as excinfo:
            first_fit(candidates, _project())

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "busy:" in excinfo.value.message
        assert "fullup:" in excinfo.value.message
        assert "somebody is on this machine" in excinfo.value.message
        assert "reserves 20 GB" in excinfo.value.message

    def test_an_empty_fleet_refuses(self) -> None:
        with pytest.raises(AppError) as excinfo:
            first_fit((), _project())

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED

    def test_a_fleet_with_no_node_of_the_right_kind_says_so(self) -> None:
        """Every node refused on its tags: nothing is full, and waiting fixes nothing."""
        candidates = (
            ("loki", _node(host="loki"), _state(host="loki")),
            (
                "diphtheria",
                _node(host="diphtheria", platform=NodePlatform.LINUX),
                _state(host="diphtheria"),
            ),
        )

        with pytest.raises(AppError) as excinfo:
            first_fit(candidates, _project(required_tags=(NodeTag.GPU, NodeTag.WINDOWS)))

        assert excinfo.value.code is FleetErrorCode.NODE_LACKS_TAG
        assert "loki lacks gpu:" in excinfo.value.message
        assert "diphtheria lacks gpu, windows:" in excinfo.value.message

    def test_one_full_node_of_the_right_kind_keeps_the_capacity_answer(self) -> None:
        """A tagged node that is merely busy will take the work later, so the reader waits."""
        candidates = (
            ("loki", _node(host="loki"), _state(host="loki")),
            ("lavender", _node(gpu=GTX_1630), _state(free_ram_gb=3.0)),
        )

        with pytest.raises(AppError) as excinfo:
            first_fit(candidates, _project(required_tags=(NodeTag.GPU,)))

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "loki lacks gpu:" in excinfo.value.message
        assert "somebody is on this machine" in excinfo.value.message

    def test_the_tagged_node_is_chosen_over_a_roomier_untagged_one(self) -> None:
        candidates = (
            ("loki", _node(host="loki", cores=32), _state(host="loki", free_ram_gb=60.0)),
            ("lavender", _node(gpu=GTX_1630), _state()),
        )

        assert first_fit(candidates, _project(required_tags=(NodeTag.GPU,))) == ("lavender", 7)


#: The tags diphtheria's runner claims with, and the roll gate it could not take.
_DOCKER_LANE = frozenset({NodeTag.LINUX, NodeTag.DOCKER})


class TestRoomForAny:
    """The node runner's gate before it claims (board task fd5cabfa): the
    three project-independent checks, on one worker of the smallest project
    the runner could take (board task 865287f3)."""

    def test_a_node_with_room_for_one_worker_passes(self) -> None:
        # 5.2 GB free: 1.2 GB past the 4.0 GB reservation, one 1.1 GB worker.
        assert room_for_any(_node(), _state(free_ram_gb=5.2), (_project(),), frozenset()) is None

    def test_the_owners_reservation_is_named_when_nothing_is_left(self) -> None:
        """lavender at 2.2 GB free on 2026-09-21T09:58Z."""
        line = str(room_for_any(_node(), _state(free_ram_gb=2.2), (_project(),), frozenset()))

        assert line.startswith("NODE_OWNER_RESERVED: lavender has 2.2 GB free against a reserv")

    def test_live_runs_holding_the_node_are_named(self) -> None:
        live = LiveLoad(runs=2, workers=14, ram_gb=15.4)

        line = str(room_for_any(_node(), _state(live=live), (_project(),), frozenset()))

        assert line.startswith("NODE_OWNER_RESERVED: lavender has 27.0 GB free against a reserv")
        assert "its 2 live fleet run(s) hold 14 worker(s) and 15.4 GB" in line

    def test_a_full_disk_is_named(self) -> None:
        line = str(room_for_any(_node(), _state(free_disk_gb=3.0), (_project(),), frozenset()))

        assert line.startswith("NODE_DISK_EXHAUSTED: lavender has 3 GB free")

    def test_diphtheria_takes_the_small_roll_gate_it_used_to_refuse(self) -> None:
        """16.6 GB free against 16.0 reserved: no room for a 1.1 GB worker,
        room for the 0.25 GB gate, which the runner's tags can serve."""
        node = _node(gpu=None, platform=NodePlatform.LINUX, reserved_ram_gb=16.0)
        state = _state(free_ram_gb=16.6)
        gate = _project(worker_ram_gb=0.25, required_tags=(NodeTag.LINUX, NodeTag.DOCKER))

        assert room_for_any(node, state, (_project(), gate), _DOCKER_LANE) is None
        refused = str(room_for_any(node, state, (_project(),), _DOCKER_LANE))
        assert refused.startswith("NODE_OWNER_RESERVED: lavender has 16.6 GB free against a reserv")

    def test_a_small_project_the_runner_cannot_serve_does_not_lower_the_bound(self) -> None:
        node = _node(gpu=None, platform=NodePlatform.LINUX, reserved_ram_gb=16.0)
        gate = _project(worker_ram_gb=0.25, required_tags=(NodeTag.LINUX, NodeTag.DOCKER))

        line = str(
            room_for_any(
                node, _state(free_ram_gb=16.6), (_project(), gate), frozenset({NodeTag.LINUX})
            )
        )

        assert line.startswith("NODE_OWNER_RESERVED: lavender has 16.6 GB free against a reserv")

    def test_a_runner_no_registered_project_fits_is_refused_by_name(self) -> None:
        gpu_only = _project(required_tags=(NodeTag.GPU,))

        assert room_for_any(_node(), _state(), (gpu_only,), frozenset({NodeTag.WINDOWS})) == (
            "NODE_LACKS_TAG: this runner carries windows, and no registered project's "
            "required tags fit it"
        )
        assert room_for_any(_node(), _state(), (gpu_only,), frozenset()) == (
            "NODE_LACKS_TAG: this runner carries no tags, and no registered project's "
            "required tags fit it"
        )
