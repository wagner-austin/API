"""Tests for reading the kinds of failure a test names."""

from __future__ import annotations

import ast
import textwrap

from monorepo_guards.effect_failures import (
    FAILURE_NAME_KINDS,
    FILE_SWAP,
    KILL_KINDS,
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
        assert FAILURE_NAME_KINDS["SIGKILL"] == KILL_KINDS == frozenset({PROCESS, FILE_SWAP})


class TestNames:
    def test_names_attributes_and_keys_count_for_their_kinds(self) -> None:
        assert _kinds("with raises(subprocess.TimeoutExpired):\n    pass\n") == {PROCESS}
        assert _kinds("raise ConnectError\n") == {NETWORK}
        assert _kinds("with raises(PermissionError):\n    pass\n") == {FILE_SWAP}
        assert _kinds("with raises(OperationalError):\n    pass\n") == {SERVICE}
        assert _kinds("assert error.errno == errno.ENOENT\n") == {FILE_SWAP}
        assert _kinds("assert refused.errno == errno.EADDRNOTAVAIL\n") == {NETWORK}
        assert _kinds("assert taken.errno == errno.EADDRINUSE\n") == {NETWORK}
        assert _kinds("assert result.timed_out\n") == {PROCESS}
        assert _kinds('assert result["killed"] is True\n') == KILL_KINDS
        assert _kinds("os.kill(pid, signal.SIGKILL)\n") == KILL_KINDS

    def test_a_failure_name_counts_as_any_word_of_a_string(self) -> None:
        assert _kinds('match = "connect ECONNREFUSED 127.0.0.1:1"\n') == {NETWORK, SERVICE}
        assert _kinds('match = "TypeError: fetch failed"\n') == {NETWORK}
        assert _kinds('match = "Connection Refused by host"\n') == {NETWORK}
        assert _kinds('match = "[Errno 13] PermissionError held"\n') == {FILE_SWAP}
        assert _kinds('match = "connection reset"\n') == frozenset()

    def test_names_that_are_not_failures(self) -> None:
        assert _kinds("timed_out = False\n") == frozenset()
        assert _kinds('assert result["stdout"] == "x"\n') == frozenset()
        assert _kinds("value = items[0]\n") == frozenset()
        assert _kinds("with raises(ValueError):\n    pass\n") == frozenset()


class TestProcessMarkers:
    def test_named_kills_count_for_the_process_and_an_interrupted_swap(self) -> None:
        assert _kinds("child.terminate()\n") == KILL_KINDS
        assert _kinds("terminate_process(pid)\n") == KILL_KINDS
        assert _kinds("nodeTerminateProcess(pid)\n") == KILL_KINDS
        assert _kinds("os.killpg(group, 9)\n") == KILL_KINDS
        assert _kinds("killer(pid)\n") == frozenset()

    def test_stated_exit_codes_and_statuses(self) -> None:
        assert _kinds("finished(returncode=3)\n") == {PROCESS}
        assert _kinds("finished(returncode=0)\n") == frozenset()
        assert _kinds("respond(status=502)\n") == {NETWORK}
        assert _kinds("respond(status=200, **extra)\n") == frozenset()
        assert _kinds('reply = {"status": 503, "exit_code": 0}\n') == {NETWORK}
        assert _kinds('reply = {"exit_code": 2, 3: 4, **rest}\n') == {PROCESS}
        assert _kinds("finished(returncode=code)\n") == frozenset()
        assert _kinds("run(argv, timeout=5)\n") == frozenset()

    def test_kill_calls_and_nonzero_exit_text(self) -> None:
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
