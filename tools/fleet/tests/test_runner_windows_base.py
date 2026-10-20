"""A Windows runner host's base: what the roster may declare, and what the scripts carry.

What the base lines DO is executed by tests/pester/rendered-runner-windows.Tests.ps1
over the committed install and audit.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.runner_windows_base import (
    MachineVariable,
    WindowsHostBase,
    decode_windows_host_base,
    encode_windows_host_base,
)
from fleet.core import runner_windows_base_render
from fleet.core.runner_audit_types import ExpectedCheck
from tests._runner_fixtures import raw_windows_base


def _variable(name: str, value: str = "C:\\fleet\\poetry") -> dict[str, JSONValue]:
    """One machine variable as the roster spells it.

    Args:
        name: Its name.
        value: Its value.

    Returns:
        The raw variable.
    """
    return {"name": name, "value": value, "reason": "the runners run as SYSTEM"}


class TestTheContract:
    def test_a_declared_base_decodes_field_for_field_and_round_trips(self) -> None:
        base = decode_windows_host_base(raw_windows_base())
        assert base == WindowsHostBase(
            execution_policy="RemoteSigned",
            machine_environment=[
                MachineVariable(
                    name="POETRY_CACHE_DIR",
                    value="C:\\fleet\\poetry",
                    reason="the runners run as SYSTEM",
                )
            ],
        )
        assert decode_windows_host_base(dict(encode_windows_host_base(base))) == base

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_host_base(None)
        assert str(refused.value) == "windows host base must be a JSON object, got NoneType"

    def test_a_policy_outside_the_set_is_refused(self) -> None:
        raw = raw_windows_base()
        raw["execution_policy"] = "Unrestricted"
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_host_base(raw)
        assert str(refused.value) == (
            "execution_policy must be one of RemoteSigned, AllSigned, got 'Unrestricted'"
        )

    @pytest.mark.parametrize(
        ("variable", "message"),
        [
            ("POETRY", "machine variable must be a JSON object, got str"),
            (
                _variable("2CACHE"),
                "a machine variable's name must be a plain identifier, got '2CACHE'",
            ),
            (_variable("A-B"), "a machine variable's name must be a plain identifier, got 'A-B'"),
            (_variable("Path"), "the machine PATH is the machine's own, not a declared variable"),
            (_variable("CACHE", ""), "machine variable CACHE needs a non-empty value and reason"),
        ],
    )
    def test_a_variable_the_install_could_not_hold_is_refused(
        self, variable: JSONValue, message: str
    ) -> None:
        raw = raw_windows_base()
        variables: list[JSONValue] = [variable]
        raw["machine_environment"] = variables
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_host_base(raw)
        assert str(refused.value) == message

    def test_a_name_declared_twice_in_any_case_is_refused(self) -> None:
        raw = raw_windows_base()
        variables: list[JSONValue] = [_variable("_CACHE"), _variable("_cache", "C:\\other")]
        raw["machine_environment"] = variables
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_host_base(raw)
        assert str(refused.value) == (
            "machine_environment declares a name twice: ['_cache', '_cache']"
        )


class TestTheRender:
    def test_the_parameters_carry_the_policy_and_the_variables_as_one_json_string(self) -> None:
        base = decode_windows_host_base(raw_windows_base())
        assert runner_windows_base_render.parameter_lines(base) == [
            "    [string]$Policy = 'RemoteSigned',",
            "    [string]$PolicyKey = "
            "'HKLM:\\SOFTWARE\\Microsoft\\PowerShell\\1\\ShellIds\\Microsoft.PowerShell',",
            "    [string]$FileSystemKey = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem',",
            "    [string]$Git = 'git',",
            "    [string]$VariableScope = 'Machine',",
            "    [string]$MachineEnvironment = "
            '\'[{"Name":"POETRY_CACHE_DIR","Value":"C:\\\\fleet\\\\poetry"}]\',',
        ]

    def test_a_variable_value_a_script_could_not_carry_is_refused(self) -> None:
        raw = raw_windows_base()
        variables: list[JSONValue] = [_variable("CACHE", "C:\\it's")]
        raw["machine_environment"] = variables
        with pytest.raises(ValueError, match=r"^machine variable CACHE \"C:\\\\it's\" contains"):
            runner_windows_base_render.parameter_lines(decode_windows_host_base(raw))

    def test_the_rows_are_the_policy_the_two_long_paths_and_one_per_variable(self) -> None:
        base = decode_windows_host_base(raw_windows_base())
        assert runner_windows_base_render.expected_checks(base) == [
            ExpectedCheck(
                check_id="execution-policy:LocalMachine:RemoteSigned",
                reason=runner_windows_base_render.POLICY_REASON,
            ),
            ExpectedCheck(
                check_id="long-paths", reason=runner_windows_base_render.LONG_PATHS_REASON
            ),
            ExpectedCheck(
                check_id="git-longpaths", reason=runner_windows_base_render.LONG_PATHS_REASON
            ),
            ExpectedCheck(
                check_id="machine-env:POETRY_CACHE_DIR", reason="the runners run as SYSTEM"
            ),
        ]

    def test_a_base_with_no_variable_carries_an_empty_array_and_no_variable_row(self) -> None:
        raw = raw_windows_base()
        variables: list[JSONValue] = []
        raw["machine_environment"] = variables
        base = decode_windows_host_base(raw)
        assert runner_windows_base_render.parameter_lines(base)[-1] == (
            "    [string]$MachineEnvironment = '[]',"
        )
        assert len(runner_windows_base_render.expected_checks(base)) == 3
