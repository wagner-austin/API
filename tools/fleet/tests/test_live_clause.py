"""The clause a capacity refusal names a node's live load with (MCPs board task 330b0465)."""

from __future__ import annotations

from fleet.contracts.node import LiveLoad
from fleet.core.live_clause import live_clause
from tests._registry import idle_state, shipped_workspace


def test_an_idle_node_with_no_reservation_names_nothing() -> None:
    state = idle_state(shipped_workspace()["nodes"]["pendragon"])
    assert live_clause(state) == ""


def test_live_runs_and_reservations_are_each_named_when_they_hold_something() -> None:
    state = idle_state(shipped_workspace()["nodes"]["pendragon"])
    state["live"] = LiveLoad(runs=2, workers=5, cores=6, ram_gb=3.25)
    assert live_clause(state) == (
        ", and its 2 live fleet run(s) hold 5 worker(s) on 6 core(s) and 3.2 GB"
    )
    state["reserved_cores"] = 4
    assert live_clause(state) == (
        ", and its 2 live fleet run(s) hold 5 worker(s) on 6 core(s) and 3.2 GB"
        ", and its reservations hold 4 core(s)"
    )
    state["live"] = LiveLoad(runs=0, workers=0, cores=0, ram_gb=0.0)
    assert live_clause(state) == ", and its reservations hold 4 core(s)"
