"""The suites whose workers cost more than one core are charged what they measured.

A project's ``worker_cores`` is what the fleet books for each worker it
grants, against a node's core pool and against the quota systemd keeps on its
dispatch slice. A suite whose workers start more than their own process
there, charged one core, is admitted beside another check into a share it
fills alone, and the slice's throttling starves both. Split out of
``test_shipped_admission.py`` when the second such suite took that file past
the 600-line ceiling (MCPs board task 5cd7f739).

MCPs/execution-deploy is charged 2 cores a worker (MCPs board task
34a9877f). Its cases start containers on the docker lane, which runs in the
node's dispatch share since MCPs board task 34514037, so a worker costs more
than its own process: 1.3 to 1.7 cores in its runs with nothing beside them.
Charged one, it took 4 of lavender's 8 quota cores and the fleet placed
MCPs/repo-contracts' 4 workers beside it, where three 30 s cases timed out on
a share throttled 2069 s (fleet job 156810dd). Charged 2, lavender grants it
3 workers and places no suite with a minimum of 3 beside it, and the same
holds in the other order. serendipity's 6 spare cores grant it 3 workers, and
lenovold's 2 afford it 1.

MCPs/mcp-shared is charged 2 cores a worker too (MCPs board task 5cd7f739).
Each vitest worker's PDF parsing and coverage, and the browsers its netns
cases start, run beside it in the dispatch share: 1.1 to 1.6 cores a worker
in five passing runs. Charged one, it took 4 of lavender's 8 quota cores
beside MCPs/sms-gateway's check (fleet job 5e3c7b7a), the share was
throttled 367 s, and two CPU-bound 30 s cases timed out. Charged 2, lavender
grants it 3 workers, 6 of its 8, so what it books is what it uses: beside it
MCPs/sms-gateway's minimum of 2 still fits the 2 left, about 6.5 cores of
real use in an 8-core share, and a suite with a minimum of 3 does not;
serendipity grants it 3 and lenovold affords it 1.

MCPs/execution-twin is charged 8 cores a worker (MCPs board task d198701c).
Its cases start Keycloak and browser containers, so a run costs about 415
CPU-seconds whatever its workers: 8.5 and 5.6 cores at one worker, 8.6 at
two. Charged one, it took 4 of lavender's 8 quota cores, MCPs/repo-contracts
was placed beside it, the share was throttled 1042 s and a 30 s case timed
out (fleet job c4df04fb). Charged 8, a node grants it one worker, lavender
places nothing beside it, and serendipity, whose spare cores are 6, refuses
it.

MCPs/repo-contracts, the suite placed beside each of those three, is charged
2 cores a worker as well (MCPs board task 18f5d762). Its cases spawn git,
node and tsc beside each vitest worker, so a run draws about 5 cores whatever
its workers: 1.2 to 1.3 a worker at the 4 lavender and colossus grant it.
Charged one, it took 4 of lavender's 8 quota cores and MCPs/watcher's 4
workers were placed beside it (fleet jobs 86aa64d0 and 8bcc8e08); together
they wanted about 8.4 cores, the share was throttled 670 s, and a 19 s watcher
case timed out at 60 s. Charged 2, lavender grants it 3 workers, 6 of its 8,
so watcher is placed beside it only at its minimum of 2 workers, about 6.5
cores of real use, and repo-contracts is not placed beside watcher's 4.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad
from fleet.contracts.tags import node_tags
from fleet.core.capacity import assess
from tests._registry import idle_state, shipped_workspace

#: The runs MCPs/repo-contracts' registry row names beside its worker_cores 2.
CONTRACTS_CORES_MEASUREMENT = (
    "Fleet jobs bb4159fd and f5f9366d on lavender and 6b23ca14, 3d1a1569 and 1b8ee2a1 on "
    "colossus, each at 4 workers with no other fleet check on its node (2026-11-18), used "
    "4.84, 5.05, 4.79, 5.17 and 4.66 cores by their units' systemd CPU accounting: 1.2 to 1.3 "
    "cores a worker, because its cases spawn git, node and tsc beside each vitest worker in "
    "the dispatch share; a run draws about 5 cores whatever its workers (0.63 a worker at "
    "sedona's 9, 0141f188). Charged one core, it took 4 of lavender's 8-core share and the "
    "fleet placed MCPs/watcher's 4 workers beside it (fleet jobs 86aa64d0 and 8bcc8e08): "
    "their units used 4.5 and 2.5 cores, the share read throttled:670s, and a 19 s watcher "
    "case timed out at 60 s (MCPs board task 18f5d762). Charged 2, lavender grants it 3 "
    "workers, 6 of its 8, so beside it only a suite whose minimum fits the 2 left: "
    "MCPs/watcher at its minimum of 2, about 6.5 cores of real use, and no suite with a "
    "minimum of 3."
)

#: The runs MCPs/execution-deploy's registry row names beside its worker_cores 2.
DEPLOY_CORES_MEASUREMENT = (
    "Fleet jobs f5d3fcb6 and 598276ca on pendragon at 5 workers and 41763ac2 on sedona at 7, "
    "each with no other fleet run on its node (2026-11-15), read dispatch=788s/94s, 887s/124s "
    "and 807s/90s: 1.7, 1.4 and 1.3 cores a worker, because each worker's cases start "
    "containers on the docker lane, which runs in the node's dispatch share since MCPs board "
    "task 34514037. Charged one core, fleet job aceacffa took 4 workers on lavender, 4 of its "
    "8-core share, and the fleet placed MCPs/repo-contracts' 4 workers beside it: the share "
    "read dispatch=2834s/358s/throttled:2069s and repo-contracts timed out three 30 s cases "
    "(fleet job 156810dd, MCPs board task 34a9877f). Charged 2, lavender grants it 3 "
    "workers, 6 of its 8, and places no suite with a minimum of 3 workers beside it."
)

#: The runs MCPs/mcp-shared's registry row names beside its worker_cores 2.
SHARED_CORES_MEASUREMENT = (
    "Fleet jobs d918fcaa and c2db5223 on sedona at 9 workers, d9e2d371 and 827aca0c on "
    "pendragon at 7 and 1e3bfe08 on lacrymal at 8 (2026-11-17) read dispatch=667s/66s, "
    "638s/64s, 979s/97s, 1146s/103s and 1044s/100s: 1.1, 1.1, 1.4, 1.6 and 1.3 cores a "
    "worker, because each vitest worker's PDF parsing, coverage and the netns cases' browsers "
    "run beside it in the dispatch share. Charged one core, fleet job 5e3c7b7a took 4 workers "
    "on lavender, 4 of its 8-core share, beside MCPs/sms-gateway's check: the share read "
    "dispatch=913s/150s/throttled:367s, the PDF-parsing files ran 3 to 7 times their sedona "
    "time, and two CPU-bound cases timed out at 30 s (MCPs board task 5cd7f739); colossus at "
    "4 workers read throttled 64 s and 108 s (49126f2e, 87aceade) and serendipity at 6 "
    "throttled 75 s (26273051). Charged 2, lavender grants it 3 workers, 6 of its 8, so a "
    "check beside it is booked against what it uses: MCPs/sms-gateway's minimum of 2 still "
    "fits the 2 left, and no suite with a minimum of 3 does."
)

#: The runs MCPs/execution-twin's registry row names beside its worker_cores 8.
TWIN_CORES_MEASUREMENT = (
    "Fleet jobs 7c1a25c9 and 75d1c95c on sedona at 1 worker and 6f257615 there at 2 "
    "(2026-11-17), each running every one of its 20 or 21 cases, read dispatch=416s/49s, "
    "418s/75s and 413s/48s: 8.5, 5.6 and 8.6 cores for the run, about 415 CPU-seconds "
    "whatever the workers, because its twin cases start Keycloak and browser containers on "
    "the docker lane, which runs in the node's dispatch share since MCPs board task 34514037. "
    "Charged one core, fleet job 94ebe31d took 4 workers on lavender, 4 of its 8-core share, "
    "and the fleet placed MCPs/repo-contracts' check c4df04fb beside it: the share read "
    "dispatch=1389s/184s/throttled:1042s and a 30 s repo-contracts case timed out at 40 s "
    "(MCPs board task d198701c); 6b46dfff at 4 workers there read throttled:531s. Charged 8, "
    "a node grants it 1 worker, lavender places nothing beside it, and a node whose spare "
    "cores are fewer than 8 (serendipity's 6) refuses it."
)


def test_lavender_places_no_three_worker_suite_beside_the_deploy_rehearsal() -> None:
    """MCPs board task 34a9877f: charged one core a worker, MCPs/execution-deploy
    took 4 of lavender's 8 quota cores (fleet job aceacffa) and MCPs/repo-contracts
    the other 4 (156810dd), while the rehearsal's containers ran in the same
    share; the share was throttled 2069 s and three 30 s cases timed out.
    Charged 2, neither suite fits beside the other in either order."""
    workspace = shipped_workspace()
    deploy = workspace["projects"]["MCPs/execution-deploy"]
    contracts = workspace["projects"]["MCPs/repo-contracts"]
    node = workspace["nodes"]["lavender"]
    deploy_tags = node_tags(node) | frozenset(deploy["required_tags"])
    contracts_tags = node_tags(node) | frozenset(contracts["required_tags"])
    state = idle_state(node)
    state["check_cpu_quota"] = 8.0

    assert deploy["worker_cores"] == 2
    assert deploy["worker_cores_measurement"] == DEPLOY_CORES_MEASUREMENT
    assert contracts["worker_cores"] == 2
    assert assess(node, state, deploy, deploy_tags) == {"workers": 3, "code": None, "reason": ""}
    assert assess(node, state, contracts, contracts_tags) == {
        "workers": 3,
        "code": None,
        "reason": "",
    }

    state["live"] = LiveLoad(runs=1, workers=3, cores=6, ram_gb=4.5)
    beside_deploy = assess(node, state, contracts, contracts_tags)
    assert (beside_deploy["workers"], beside_deploy["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_deploy["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 2 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )

    state["live"] = LiveLoad(runs=1, workers=3, cores=6, ram_gb=1.5)
    beside_contracts = assess(node, state, deploy, deploy_tags)
    assert (beside_contracts["workers"], beside_contracts["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_contracts["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 2 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )


def test_lavender_books_mcp_shared_at_the_cores_its_workers_use() -> None:
    """MCPs board task 5cd7f739: charged one core a worker, MCPs/mcp-shared took
    4 of lavender's 8 quota cores beside MCPs/sms-gateway (fleet job 5e3c7b7a),
    using about 6 of them, the share was throttled 367 s and two 30 s cases
    timed out. Charged 2, it is granted 3 workers booked at 6 cores, so only a
    suite whose minimum fits the 2 left is placed beside it."""
    workspace = shipped_workspace()
    shared = workspace["projects"]["MCPs/mcp-shared"]
    gateway = workspace["projects"]["MCPs/sms-gateway"]
    contracts = workspace["projects"]["MCPs/repo-contracts"]
    node = workspace["nodes"]["lavender"]
    shared_tags = node_tags(node) | frozenset(shared["required_tags"])
    gateway_tags = node_tags(node) | frozenset(gateway["required_tags"])
    contracts_tags = node_tags(node) | frozenset(contracts["required_tags"])
    state = idle_state(node)
    state["check_cpu_quota"] = 8.0

    assert shared["worker_cores"] == 2
    assert shared["worker_cores_measurement"] == SHARED_CORES_MEASUREMENT
    assert (gateway["worker_cores"], gateway["minimum_workers"]) == (1, 2)
    assert (contracts["worker_cores"], contracts["minimum_workers"]) == (2, 3)
    assert assess(node, state, shared, shared_tags) == {"workers": 3, "code": None, "reason": ""}

    state["live"] = LiveLoad(runs=1, workers=3, cores=6, ram_gb=1.5)
    assert assess(node, state, gateway, gateway_tags) == {
        "workers": 2,
        "code": None,
        "reason": "",
    }
    beside_shared = assess(node, state, contracts, contracts_tags)
    assert (beside_shared["workers"], beside_shared["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_shared["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 2 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )

    state["live"] = LiveLoad(runs=1, workers=3, cores=6, ram_gb=1.5)
    beside_contracts = assess(node, state, shared, shared_tags)
    assert (beside_contracts["workers"], beside_contracts["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_contracts["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 2 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )


def test_execution_twin_takes_lavenders_whole_share_and_one_worker_anywhere() -> None:
    """MCPs board task d198701c: charged one core a worker, MCPs/execution-twin
    took 4 of lavender's 8 quota cores (fleet job 94ebe31d) while its
    containers used the whole share, and MCPs/repo-contracts' check placed
    beside it (c4df04fb) timed out a 30 s case on a share throttled 1042 s.
    A run costs about 415 CPU-seconds at any worker count, so charged 8 it is
    granted one worker, nothing is placed beside it on lavender, it is placed
    beside nothing there, and a node with 6 spare cores refuses it."""
    workspace = shipped_workspace()
    twin = workspace["projects"]["MCPs/execution-twin"]
    contracts = workspace["projects"]["MCPs/repo-contracts"]
    probe = workspace["projects"]["MCPs/login-probe"]
    lavender = workspace["nodes"]["lavender"]
    twin_tags = node_tags(lavender) | frozenset(twin["required_tags"])
    contracts_tags = node_tags(lavender) | frozenset(contracts["required_tags"])
    probe_tags = node_tags(lavender) | frozenset(probe["required_tags"])
    state = idle_state(lavender)
    state["check_cpu_quota"] = 8.0

    assert (twin["worker_cores"], twin["minimum_workers"]) == (8, 1)
    assert twin["worker_cores_measurement"] == TWIN_CORES_MEASUREMENT
    assert (contracts["worker_cores"], contracts["minimum_workers"]) == (2, 3)
    assert (probe["worker_cores"], probe["minimum_workers"]) == (1, 2)
    assert assess(lavender, state, twin, twin_tags) == {"workers": 1, "code": None, "reason": ""}

    state["live"] = LiveLoad(runs=1, workers=1, cores=8, ram_gb=4.5)
    beside = ((contracts, contracts_tags, 3, 6), (probe, probe_tags, 2, 2))
    for suite, tags, minimum, occupies in beside:
        beside_twin = assess(lavender, state, suite, tags)
        assert (beside_twin["workers"], beside_twin["code"]) == (
            0,
            FleetErrorCode.NODE_OWNER_RESERVED,
        )
        assert beside_twin["reason"].startswith(
            "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their "
            f"slice, so 0 core(s) are left where the suite's minimum of {minimum} worker(s) "
            f"occupies {occupies}"
        )

    state["live"] = LiveLoad(runs=1, workers=3, cores=6, ram_gb=1.5)
    beside_contracts = assess(lavender, state, twin, twin_tags)
    assert (beside_contracts["workers"], beside_contracts["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_contracts["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 2 core(s) are left where the suite's minimum of 1 worker(s) occupies 8"
    )

    sedona = workspace["nodes"]["sedona"]
    sedona_tags = node_tags(sedona) | frozenset(twin["required_tags"])
    assert assess(sedona, idle_state(sedona), twin, sedona_tags) == {
        "workers": 1,
        "code": None,
        "reason": "",
    }
    serendipity = workspace["nodes"]["serendipity"]
    serendipity_tags = node_tags(serendipity) | frozenset(twin["required_tags"])
    refused = assess(serendipity, idle_state(serendipity), twin, serendipity_tags)
    assert (refused["workers"], refused["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
    assert refused["reason"].startswith(
        "serendipity has 24.0 GB free against a reservation of 1.5 GB for whoever is using it, "
        "and 8 cores against 2 reserved. Nothing is left for a dispatch"
    )


def test_lavender_books_repo_contracts_so_watcher_runs_beside_it_at_its_minimum() -> None:
    """MCPs board task 18f5d762: charged one core a worker, MCPs/repo-contracts
    took 4 of lavender's 8 quota cores (fleet job 86aa64d0) and MCPs/watcher the
    other 4 (8bcc8e08), while their units used 4.5 and 2.5 cores; the share was
    throttled 670 s and a 19 s watcher case timed out at 60 s. Charged 2,
    repo-contracts is granted 3 workers booked at 6 cores, watcher fits beside
    it only at its minimum of 2, and repo-contracts does not fit beside
    watcher's 4."""
    workspace = shipped_workspace()
    contracts = workspace["projects"]["MCPs/repo-contracts"]
    watcher = workspace["projects"]["MCPs/watcher"]
    node = workspace["nodes"]["lavender"]
    contracts_tags = node_tags(node) | frozenset(contracts["required_tags"])
    watcher_tags = node_tags(node) | frozenset(watcher["required_tags"])
    state = idle_state(node)
    state["check_cpu_quota"] = 8.0

    assert (contracts["worker_cores"], contracts["minimum_workers"]) == (2, 3)
    assert contracts["worker_cores_measurement"] == CONTRACTS_CORES_MEASUREMENT
    assert (watcher["worker_cores"], watcher["minimum_workers"]) == (1, 2)
    assert assess(node, state, contracts, contracts_tags) == {
        "workers": 3,
        "code": None,
        "reason": "",
    }
    assert assess(node, state, watcher, watcher_tags) == {"workers": 4, "code": None, "reason": ""}

    state["live"] = LiveLoad(runs=1, workers=3, cores=6, ram_gb=1.5)
    assert assess(node, state, watcher, watcher_tags) == {"workers": 2, "code": None, "reason": ""}

    state["live"] = LiveLoad(runs=1, workers=4, cores=4, ram_gb=2.0)
    beside_watcher = assess(node, state, contracts, contracts_tags)
    assert (beside_watcher["workers"], beside_watcher["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_watcher["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 4 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )
