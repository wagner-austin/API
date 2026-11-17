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
grants it its whole share, 4 workers, and places no suite beside it;
serendipity grants it 3 and lenovold affords it 1.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad
from fleet.contracts.tags import node_tags
from fleet.core.capacity import assess
from tests._registry import idle_state, shipped_workspace

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
    "throttled 75 s (26273051). Charged 2, lavender grants it 4 workers, its whole 8-core "
    "share, and places no suite beside it."
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
    assert contracts["worker_cores"] == 1
    assert assess(node, state, deploy, deploy_tags) == {"workers": 3, "code": None, "reason": ""}
    assert assess(node, state, contracts, contracts_tags) == {
        "workers": 4,
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
        "so 2 core(s) are left where the suite's minimum of 3 worker(s) occupies 3"
    )

    state["live"] = LiveLoad(runs=1, workers=4, cores=4, ram_gb=2.0)
    beside_contracts = assess(node, state, deploy, deploy_tags)
    assert (beside_contracts["workers"], beside_contracts["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_contracts["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 4 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )


def test_lavender_gives_mcp_shared_its_whole_share_and_nothing_beside_it() -> None:
    """MCPs board task 5cd7f739: charged one core a worker, MCPs/mcp-shared took
    4 of lavender's 8 quota cores beside MCPs/sms-gateway (fleet job 5e3c7b7a),
    the share was throttled 367 s and two 30 s cases timed out. Charged 2, its
    4 workers fill the share, and neither suite fits beside the other."""
    workspace = shipped_workspace()
    shared = workspace["projects"]["MCPs/mcp-shared"]
    gateway = workspace["projects"]["MCPs/sms-gateway"]
    node = workspace["nodes"]["lavender"]
    shared_tags = node_tags(node) | frozenset(shared["required_tags"])
    gateway_tags = node_tags(node) | frozenset(gateway["required_tags"])
    state = idle_state(node)
    state["check_cpu_quota"] = 8.0

    assert shared["worker_cores"] == 2
    assert shared["worker_cores_measurement"] == SHARED_CORES_MEASUREMENT
    assert gateway["worker_cores"] == 1
    assert assess(node, state, shared, shared_tags) == {"workers": 4, "code": None, "reason": ""}

    state["live"] = LiveLoad(runs=1, workers=4, cores=8, ram_gb=2.0)
    beside_shared = assess(node, state, gateway, gateway_tags)
    assert (beside_shared["workers"], beside_shared["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_shared["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 0 core(s) are left where the suite's minimum of 2 worker(s) occupies 2"
    )

    state["live"] = LiveLoad(runs=1, workers=4, cores=4, ram_gb=2.0)
    beside_gateway = assess(node, state, shared, shared_tags)
    assert (beside_gateway["workers"], beside_gateway["code"]) == (
        0,
        FleetErrorCode.NODE_OWNER_RESERVED,
    )
    assert beside_gateway["reason"].startswith(
        "lavender's checks are held to 8.00 CPU(s) by the quota systemd keeps on their slice, "
        "so 4 core(s) are left where the suite's minimum of 3 worker(s) occupies 6"
    )
