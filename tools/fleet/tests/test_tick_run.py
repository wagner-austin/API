"""The run a fleet-agent tick is, and how its liveness is read (MCPs board task 05dd6588).

The cgroup lines and systemctl answers here are the shapes measured on
diphtheria on 2026-10-21; ``test_tick_run_hub.py`` reads them from the hub's
own user manager, against a unit it starts and kills.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError

from fleet.core import _test_hooks, tick_run
from tests._agent_cli import TICK_CGROUP, TICK_UNIT, bind_hub_tick
from tests.conftest import FakeRun, failed, ok

#: A running tick's unit as ``systemctl --user show`` read it.
LIVE_SHOWN = (
    f"ActiveState=active\nControlGroup=/user.slice/user-1000.slice/user@1000.service/"
    f"app.slice/{TICK_UNIT}\n"
)

#: A unit the manager no longer knows, as a collected stopped run reads.
GONE_SHOWN = "ActiveState=inactive\nControlGroup=\n"


class TestTheTicksOwnRun:
    def test_a_tick_in_a_user_managers_unit_is_named_by_that_unit(self) -> None:
        assert tick_run.own_run(TICK_CGROUP, "diphtheria") == tick_run.TickRun(
            unit=TICK_UNIT, machine="diphtheria"
        )

    def test_a_scope_of_the_user_manager_names_its_run_too(self) -> None:
        cgroup = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-p41.scope\n"

        assert tick_run.own_run(cgroup, "ziggurat")["unit"] == "run-p41.scope"

    @pytest.mark.parametrize(
        ("cgroup", "named"),
        [
            # An ssh login's own scope is the system manager's, not the user's.
            ("0::/user.slice/user-1000.slice/session-3.scope\n", "0::/user.slice/"),
            ("0::/system.slice/cron.service\n", "0::/system.slice/cron.service"),
            ("", "no control group"),
        ],
    )
    def test_a_tick_in_no_user_managers_unit_is_refused_by_name(
        self, cgroup: str, named: str
    ) -> None:
        with pytest.raises(AppError) as refused:
            tick_run.own_run(cgroup, "diphtheria")

        assert str(refused.value).startswith(f"{tick_run.UNBOUND_CODE}: this tick runs in ")
        assert named in str(refused.value)

    def test_the_run_is_read_from_this_processs_cgroup_on_this_machine(self) -> None:
        bind_hub_tick()

        assert tick_run.read_own_run() == tick_run.TickRun(unit=TICK_UNIT, machine="austinpc")

    def test_a_machine_with_no_proc_is_refused_as_a_tick_in_no_unit(self) -> None:
        asked: list[pathlib.Path] = []

        def absent(path: pathlib.Path) -> bool:
            asked.append(path)
            return False

        _test_hooks.file_exists = absent

        with pytest.raises(AppError, match=f"{tick_run.UNBOUND_CODE}: .*no control group"):
            tick_run.read_own_run()
        assert asked == [tick_run.PROC_SELF_CGROUP]


class TestTheRunId:
    def test_a_run_id_names_the_unit_then_the_machine_and_reads_back(self) -> None:
        run = tick_run.TickRun(unit=TICK_UNIT, machine="diphtheria")

        assert tick_run.encode_run_id(run) == f"{TICK_UNIT} on diphtheria"
        assert tick_run.decode_run_id(tick_run.encode_run_id(run)) == run

    @pytest.mark.parametrize(
        "run_id",
        ["", "netboot-arm-aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa", "MCPs-search-lavender-1790775427"],
    )
    def test_an_id_naming_no_unit_reads_as_none(self, run_id: str) -> None:
        """A claimed job not yet started has none, and every job started
        before runs were named by their unit carries the old shape."""
        assert tick_run.decode_run_id(run_id) is None


class TestTheRunsState:
    def test_a_running_unit_is_live(self) -> None:
        run = FakeRun([ok(LIVE_SHOWN)])
        _test_hooks.run = run

        state = tick_run.read_run_state(TICK_UNIT)

        assert state["live"] is True
        assert state["shown"].startswith("ActiveState=active ControlGroup=/user.slice/")
        assert run.calls == [
            (
                "/usr/bin/systemctl",
                "--user",
                "show",
                "--property=ActiveState",
                "--property=ControlGroup",
                TICK_UNIT,
            )
        ]
        assert run.timeouts == [tick_run.SHOW_TIMEOUT_SECONDS]

    def test_a_unit_stopped_with_no_control_group_is_gone(self) -> None:
        _test_hooks.run = FakeRun([ok(GONE_SHOWN)])

        assert tick_run.read_run_state(TICK_UNIT) == tick_run.RunState(
            live=False, shown="ActiveState=inactive ControlGroup="
        )

    def test_a_failed_unit_with_no_control_group_is_gone(self) -> None:
        _test_hooks.run = FakeRun([ok("ActiveState=failed\nControlGroup=\n")])

        assert tick_run.read_run_state(TICK_UNIT)["live"] is False

    def test_a_stopped_state_with_a_member_left_is_still_live(self) -> None:
        """Nothing is said to be gone while a process of the run may remain."""
        _test_hooks.run = FakeRun(
            [ok(f"ActiveState=failed\nControlGroup=/user.slice/app.slice/{TICK_UNIT}\n")]
        )

        assert tick_run.read_run_state(TICK_UNIT)["live"] is True

    def test_a_manager_that_cannot_be_asked_is_refused_by_name(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "Failed to connect to bus: No medium found")])

        with pytest.raises(AppError) as refused:
            tick_run.read_run_state(TICK_UNIT)

        assert str(refused.value) == (
            f"{tick_run.UNREADABLE_CODE}: /usr/bin/systemctl --user show "
            f"--property=ActiveState --property=ControlGroup {TICK_UNIT} exited 1: "
            "Failed to connect to bus: No medium found; it answered ''"
        )

    def test_an_answer_without_both_properties_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("ActiveState=inactive\n")])

        with pytest.raises(AppError, match=f"{tick_run.UNREADABLE_CODE}: .* exited 0: ; it "):
            tick_run.read_run_state(TICK_UNIT)
