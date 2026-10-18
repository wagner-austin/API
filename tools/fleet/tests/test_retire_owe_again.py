"""A failed retire owed again once its fault is fixed (MCPs board task 3d9eec0d).

:func:`fleet.core.retire_owed.owe_again` and its command,
:mod:`fleet.cli.owe_again`, against a real record under ``tmp_path`` and the
real transport with the node's ssh answered by :class:`tests.conftest.FakeRun`.
A failed run owed again is retired by the node's next pass like any owed run;
a run whose last line is anything but ``failed`` is refused by name and the
record is left as it was.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.cli import _config, owe_again
from fleet.contracts.retire_record import RetireRecord, RetireState
from fleet.core import _test_hooks, records, retire_owed
from tests._toolchain_fixtures import node
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, failed, ok, retire_replies

#: What fixed the fault, as a session records it.
BECAUSE = "8e655f640 skips a transcript holder that exits under the retire"

#: What the node said when the retire failed.
HOLDER_GONE = "The property 'Name' cannot be found on this object"


def _failed(path: pathlib.Path) -> None:
    """Owe the demo run, then fail its retire on the node.

    Args:
        path: The record file.
    """
    retire_owed.owe(path, alias="lavender", run_id=DEMO_RUN_ID)
    _test_hooks.run = FakeRun([ok(""), failed(1, HOLDER_GONE)])
    with pytest.raises(AppError):
        retire_owed.retire_or_owe(path, node(), alias="lavender", run_id=DEMO_RUN_ID)


def _pass(path: pathlib.Path) -> None:
    """Send the demo node's owed retires, with no lease or ledger row live.

    Args:
        path: The record file.
    """
    retire_owed.retire_owed(
        path,
        node(),
        alias="lavender",
        leases_path=path.parent / "leases.json",
        ledger_path=path.parent / "ledger.jsonl",
    )


class TestOweAgain:
    def test_a_failed_run_is_owed_with_its_fix_and_the_next_pass_retires_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        _failed(path)

        alias = retire_owed.owe_again(path, run_id=DEMO_RUN_ID, because=BECAUSE)

        assert alias == "lavender"
        owed = records.read_retires(path)[-1]
        assert (owed["node"], owed["state"]) == ("lavender", RetireState.OWED)
        assert owed["detail"] == f"its failed retire is owed again: {BECAUSE}"
        answered = FakeRun(retire_replies())
        _test_hooks.run = answered
        _pass(path)
        assert any(f"retire-{DEMO_RUN_ID}.ps1" in " ".join(call) for call in answered.calls)
        assert records.read_retires(path)[-1]["state"] is RetireState.RETIRED

    @pytest.mark.parametrize(
        ("states", "said"),
        [
            ((), "not recorded"),
            ((RetireState.OWED,), "owed"),
            ((RetireState.OWED, RetireState.RETIRED), "retired"),
        ],
    )
    def test_a_run_whose_last_line_is_not_failed_is_refused_and_nothing_is_recorded(
        self, tmp_path: pathlib.Path, states: tuple[RetireState, ...], said: str
    ) -> None:
        path = tmp_path / "runs" / retire_owed.RETIRES_FILE
        for state in states:
            records.append_retire(
                path,
                RetireRecord(
                    run_id=DEMO_RUN_ID, node="lavender", state=state, at_unix=DEMO_NOW, detail=""
                ),
            )
        before = records.read_retires(path)

        with pytest.raises(AppError) as raised:
            retire_owed.owe_again(path, run_id=DEMO_RUN_ID, because=BECAUSE)

        assert raised.value.code is FleetErrorCode.RUN_UNKNOWN
        assert raised.value.message == (
            f"{DEMO_RUN_ID}'s retire is {said}, not failed, so it is not owed again"
        )
        assert records.read_retires(path) == before


class TestTheCommand:
    def test_owes_the_named_run_in_the_registry_s_records(self, config_path: pathlib.Path) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _failed(loaded.retires)

        code = owe_again.main(
            [
                _config.CONFIG_FLAG,
                str(config_path),
                owe_again.RUN_FLAG,
                DEMO_RUN_ID,
                owe_again.BECAUSE_FLAG,
                f"  {BECAUSE} ",
            ]
        )

        assert code == 0
        owed = records.read_retires(loaded.retires)[-1]
        assert (owed["state"], owed["detail"]) == (
            RetireState.OWED,
            f"its failed retire is owed again: {BECAUSE}",
        )

    def test_a_reason_that_says_nothing_is_refused_before_anything_is_recorded(
        self, config_path: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _failed(loaded.retires)
        before = records.read_retires(loaded.retires)

        with pytest.raises(ValueError, match="must say what fixed"):
            owe_again.main(
                [
                    _config.CONFIG_FLAG,
                    str(config_path),
                    owe_again.RUN_FLAG,
                    DEMO_RUN_ID,
                    owe_again.BECAUSE_FLAG,
                    "   ",
                ]
            )

        assert records.read_retires(loaded.retires) == before

    def test_running_it_as_a_module_actually_owes_the_run(self, config_path: pathlib.Path) -> None:
        """Without its ``if __name__`` block the module would run nothing and
        exit 0, a run reported owed that no pass will retire."""
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
        _failed(loaded.retires)
        saved_argv = sys.argv
        saved_module = sys.modules.pop("fleet.cli.owe_again", None)
        sys.argv = [
            "x",
            _config.CONFIG_FLAG,
            str(config_path),
            owe_again.RUN_FLAG,
            DEMO_RUN_ID,
            owe_again.BECAUSE_FLAG,
            BECAUSE,
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module("fleet.cli.owe_again", run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules["fleet.cli.owe_again"] = saved_module

        assert raised.value.code == 0
        assert records.read_retires(loaded.retires)[-1]["state"] is RetireState.OWED
