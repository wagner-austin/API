"""Tests for reading the kinds of failure a test names."""

from __future__ import annotations

import ast
import textwrap

from monorepo_guards.effect_failures import (
    FAILURE_NAME_KINDS,
    FILE_SWAP,
    NETWORK,
    PROCESS,
    SATISFIED_BY,
    SERVICE,
    SSH,
    failure_kinds,
)


def _kinds(source: str) -> frozenset[str]:
    """Read the failure kinds one snippet names.

    Args:
        source: Python statements.

    Returns:
        The kinds.
    """
    return failure_kinds(ast.parse(textwrap.dedent(source)))


class TestTables:
    def test_every_effect_kind_is_satisfied_by_its_own_failures(self) -> None:
        assert {
            PROCESS: frozenset({PROCESS}),
            NETWORK: frozenset({NETWORK}),
            SSH: frozenset({PROCESS, NETWORK}),
            FILE_SWAP: frozenset({FILE_SWAP}),
            SERVICE: frozenset({SERVICE}),
        } == SATISFIED_BY
        assert FAILURE_NAME_KINDS["TimeoutError"] == frozenset({PROCESS, NETWORK, SERVICE})
        assert FAILURE_NAME_KINDS["EBUSY"] == frozenset({FILE_SWAP})


class TestNames:
    def test_names_attributes_and_keys_count_for_their_kinds(self) -> None:
        assert _kinds("with raises(subprocess.TimeoutExpired):\n    pass\n") == {PROCESS}
        assert _kinds("raise ConnectError\n") == {NETWORK}
        assert _kinds("with raises(PermissionError):\n    pass\n") == {FILE_SWAP}
        assert _kinds("with raises(OperationalError):\n    pass\n") == {SERVICE}
        assert _kinds("assert error.errno == errno.ENOENT\n") == {FILE_SWAP}
        assert _kinds("assert result.timed_out\n") == {PROCESS}
        assert _kinds('assert result["killed"] is True\n') == {PROCESS}

    def test_names_that_are_not_failures(self) -> None:
        assert _kinds("timed_out = False\n") == frozenset()
        assert _kinds('assert result["stdout"] == "x"\n') == frozenset()
        assert _kinds("value = items[0]\n") == frozenset()
        assert _kinds("with raises(ValueError):\n    pass\n") == frozenset()


class TestProcessMarkers:
    def test_kill_calls_and_nonzero_exit_text(self) -> None:
        assert _kinds("child.terminate()\n") == {PROCESS}
        assert _kinds('script = "import sys; sys.exit(3)"\n') == {PROCESS}
        assert _kinds('script = "process.exit(1)"\n') == {PROCESS}
        assert _kinds('script = "exit 2"\n') == {PROCESS}
        assert _kinds('script = "sys.exit(0)"\n') == frozenset()
        assert _kinds("make()()\n") == frozenset()

    def test_exit_code_comparisons(self) -> None:
        for failing in (
            "assert result.returncode == 3\n",
            "assert 3 == result.returncode\n",
            'assert result["returncode"] != 0\n',
            "assert result.exit_code > 0\n",
            "assert 0 < result.code\n",
        ):
            assert _kinds(failing) == {PROCESS}
        for passing in (
            "assert result.returncode == 0\n",
            "assert result.returncode != 1\n",
            "assert result.returncode > 1\n",
            "assert 0 > result.code\n",
            "assert result.returncode == expected\n",
            "assert result.returncode == True\n",
            "assert size == 3\n",
        ):
            assert _kinds(passing) == frozenset()


class TestStatusMarkers:
    def test_a_status_of_400_or_more_is_a_network_failure(self) -> None:
        assert _kinds("assert response.status_code == 503\n") == {NETWORK}
        assert _kinds("assert 404 == response.status\n") == {NETWORK}
        assert _kinds("assert response.status_code == 200\n") == frozenset()
        assert _kinds("assert response.status == expected\n") == frozenset()

    def test_kinds_accumulate_across_a_function(self) -> None:
        source = """
        def test_both():
            assert result.returncode == 1
            with raises(FileNotFoundError):
                pass
        """
        assert _kinds(source) == {PROCESS, FILE_SWAP}
