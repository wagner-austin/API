"""A node's reservations come off its cores (MCPs board task 84a2b690).

Split from ``test_capacity.py``, which sits at its line ceiling, and built
with its builders. The shape is diphtheria's on 2026-10-14: 8 cores with
1 reserved for its owner, a clean-VM smoke holding 4 of them, and the
packages/db and doc-extract-api checks placed beside it running 411 s, 334 s
and 307 s of their 300 s budgets.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad, NodePlatform
from fleet.core.capacity import assess
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


def _diphtheria(*, reserved_cores: int, live: LiveLoad | None = None) -> tuple[int, str]:
    """Weigh doc-extract-api on an 8-core diphtheria with reservations held.

    Args:
        reserved_cores: The cores its reservations hold.
        live: What its live fleet runs hold, or None for none.

    Returns:
        The workers granted and the refusal's reason.
    """
    node = _node(host="diphtheria", cores=8, reserved_cores=1, platform=NodePlatform.LINUX)
    state = _state(host="diphtheria", free_ram_gb=20.0, reserved_cores=reserved_cores)
    if live is not None:
        state["live"] = live
    verdict = assess(node, state, _project(minimum_workers=2, worker_ram_gb=0.3), LINUX_TAGS)
    return verdict["workers"], verdict["reason"]


class TestReservedCores:
    def test_no_reservation_grants_what_it_did_before(self) -> None:
        """8 cores, 1 reserved: 7 spare, the job ceiling 4."""
        assert _diphtheria(reserved_cores=0) == (4, "")

    def test_a_reservation_s_cores_come_off_the_grant(self) -> None:
        """The VM's 4 cores leave 8 - 4 - 1 = 3."""
        assert _diphtheria(reserved_cores=4) == (3, "")

    def test_a_reservation_and_a_live_run_are_both_charged(self) -> None:
        """8 - 4 reserved - 2 live - 1 owner = 1 core, under the suite's minimum of 2."""
        workers, reason = _diphtheria(
            reserved_cores=4, live=LiveLoad(runs=1, workers=2, ram_gb=0.6)
        )

        assert workers == 0
        assert reason == (
            "diphtheria affords 1 worker(s) for a suite that declares a minimum of 2: 8 cores "
            "against 1 reserved, 20.0 GB free, 0.3 GB per worker, 4.0 GB reserved for the "
            "node's owner, and its 1 live fleet run(s) hold 2 worker(s) and 0.6 GB, and its "
            "reservations hold 4 core(s). "
            "Dispatching anyway would run a suite at a fraction of its workers until its own "
            "lease expired underneath it."
        )

    def test_a_reservation_holding_every_spare_core_is_named(self) -> None:
        workers, reason = _diphtheria(reserved_cores=7)

        assert workers == 0
        assert reason == (
            "diphtheria has 20.0 GB free against a reservation of 4.0 GB for whoever is using "
            "it, and 8 cores against 1 reserved, and its reservations hold 7 core(s). Nothing "
            "is left for a dispatch; its reservations hold the rest (work the fleet does not "
            "run, declared in ~/.fleet/reservations), so it takes the next job when they end."
        )

    def test_a_node_its_owner_fills_names_the_owner_whatever_is_reserved(self) -> None:
        node = _node(host="diphtheria", cores=8, reserved_cores=1, platform=NodePlatform.LINUX)
        state = _state(host="diphtheria", free_ram_gb=3.0, reserved_cores=2)

        verdict = assess(node, state, _project(), LINUX_TAGS)

        assert verdict["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert verdict["reason"].endswith("somebody is on this machine.")
