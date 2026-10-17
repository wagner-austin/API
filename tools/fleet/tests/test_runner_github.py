"""The two ``gh`` reads, against fakes at the command seam.

Each fake answers through ``_test_hooks.run`` exactly as
:mod:`tests.test_core_io` describes, so the argv each test asserts is the one
production would run.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, runner_github
from tests.conftest import FakeRun, failed, ok


class TestMintToken:
    """The registration token onboarding and rebuilding embed."""

    def test_a_minted_token_is_returned(self) -> None:
        runner = FakeRun([ok("ABCDEF123\n")])
        _test_hooks.run = runner
        assert runner_github.mint_registration_token("wagner-austin/x") == "ABCDEF123"
        argv = tuple(runner.calls[0])
        assert argv == (
            "gh",
            "api",
            "-X",
            "POST",
            "repos/wagner-austin/x/actions/runners/registration-token",
            "-q",
            ".token",
        )
        assert runner.timeouts[0] == runner_github.GH_API_TIMEOUT_SECONDS

    def test_a_gh_failure_is_a_local_auth_condition(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 403: no admin rights")])
        with pytest.raises(AppError) as fault:
            runner_github.mint_registration_token("wagner-austin/x")
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE
        assert "403" in fault.value.message

    def test_a_silent_gh_failure_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])
        with pytest.raises(AppError) as fault:
            runner_github.mint_registration_token("wagner-austin/x")
        assert "<no stderr>" in fault.value.message

    def test_an_empty_token_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("\n")])
        with pytest.raises(AppError) as fault:
            runner_github.mint_registration_token("wagner-austin/x")
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE

    def test_a_token_that_could_break_quoting_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("AB'; rm -rf /; '\n")])
        with pytest.raises(AppError) as fault:
            runner_github.mint_registration_token("wagner-austin/x")
        assert fault.value.code is FleetErrorCode.RUNNER_TOKEN_UNAVAILABLE
        assert "non-alphanumeric" in fault.value.message


class TestReadRunners:
    """Each repository's runner list, the dropped-job row's GitHub half."""

    def test_each_listed_runner_is_read_by_name(self) -> None:
        runner = FakeRun([ok("lavender-wsl offline false\nlavender-wsl-2 online true\n")])
        _test_hooks.run = runner
        assert runner_github.read_github_runners("wagner-austin/API") == {
            "lavender-wsl": runner_github.GitHubRunner(status="offline", busy=False),
            "lavender-wsl-2": runner_github.GitHubRunner(status="online", busy=True),
        }
        assert tuple(runner.calls[0]) == (
            "gh",
            "api",
            "--paginate",
            "repos/wagner-austin/API/actions/runners",
            "--jq",
            '.runners[] | "\\(.name) \\(.status) \\(.busy)"',
        )
        assert runner.timeouts[0] == runner_github.GH_API_TIMEOUT_SECONDS

    def test_a_repository_with_no_runners_reads_empty(self) -> None:
        _test_hooks.run = FakeRun([ok("")])
        assert runner_github.read_github_runners("wagner-austin/API") == {}

    def test_a_gh_failure_is_a_local_auth_condition(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 404: Not Found")])
        with pytest.raises(AppError) as fault:
            runner_github.read_github_runners("wagner-austin/API")
        assert fault.value.code is FleetErrorCode.RUNNER_GITHUB_UNREADABLE
        assert "404" in fault.value.message

    def test_a_silent_gh_failure_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])
        with pytest.raises(AppError) as fault:
            runner_github.read_github_runners("wagner-austin/API")
        assert "<no stderr>" in fault.value.message

    @pytest.mark.parametrize(
        "line", ["lavender-wsl offline", "lavender-wsl offline maybe", "a b c d"]
    )
    def test_a_line_of_any_other_shape_is_refused(self, line: str) -> None:
        _test_hooks.run = FakeRun([ok(line + "\n")])
        with pytest.raises(AppError) as fault:
            runner_github.read_github_runners("wagner-austin/API")
        assert fault.value.code is FleetErrorCode.RUNNER_GITHUB_UNREADABLE
        assert repr(line) in fault.value.message
