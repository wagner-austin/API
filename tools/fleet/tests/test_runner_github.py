"""The ``gh`` reads, the removal and the label changes, against fakes at the command seam.

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
        runner = FakeRun([ok("21 lavender-wsl offline false\n22 lavender-wsl-2 online true\n")])
        _test_hooks.run = runner
        assert runner_github.read_github_runners("wagner-austin/API") == {
            "lavender-wsl": runner_github.GitHubRunner(id=21, status="offline", busy=False),
            "lavender-wsl-2": runner_github.GitHubRunner(id=22, status="online", busy=True),
        }
        assert tuple(runner.calls[0]) == (
            "gh",
            "api",
            "--paginate",
            "repos/wagner-austin/API/actions/runners",
            "--jq",
            '.runners[] | "\\(.id) \\(.name) \\(.status) \\(.busy)"',
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
        "line",
        [
            "21 lavender-wsl offline",
            "21 lavender-wsl offline maybe",
            "lavender-wsl offline false",
            "x21 lavender-wsl offline false",
            "a b c d e",
        ],
    )
    def test_a_line_of_any_other_shape_is_refused(self, line: str) -> None:
        _test_hooks.run = FakeRun([ok(line + "\n")])
        with pytest.raises(AppError) as fault:
            runner_github.read_github_runners("wagner-austin/API")
        assert fault.value.code is FleetErrorCode.RUNNER_GITHUB_UNREADABLE
        assert fault.value.message == (
            f"gh listed a runner of wagner-austin/API as {line!r}, not "
            "'<id> <name> <status> <busy>'"
        )


class TestDeleteRegistration:
    """The prune's one write, by GitHub's id for the registration."""

    def test_the_registration_is_deleted_by_id(self) -> None:
        runner = FakeRun([ok("")])
        _test_hooks.run = runner
        runner_github.delete_registration("wagner-austin/API", 21)
        assert tuple(runner.calls[0]) == (
            "gh",
            "api",
            "-X",
            "DELETE",
            "repos/wagner-austin/API/actions/runners/21",
        )
        assert runner.timeouts[0] == runner_github.GH_API_TIMEOUT_SECONDS

    def test_a_gh_failure_is_a_local_auth_condition(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 403: Must have admin rights")])
        with pytest.raises(AppError) as fault:
            runner_github.delete_registration("wagner-austin/API", 21)
        assert fault.value.code is FleetErrorCode.RUNNER_DEREGISTRATION_REFUSED
        assert fault.value.message == (
            "gh could not delete runner 21 of wagner-austin/API: HTTP 403: Must have admin "
            "rights. This is local auth or repo permissions, not the fleet."
        )

    def test_a_silent_gh_failure_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])
        with pytest.raises(AppError) as fault:
            runner_github.delete_registration("wagner-austin/API", 21)
        assert "<no stderr>" in fault.value.message


#: Where one fixture runner's labels live.
_LABELS = "repos/wagner-austin/MCPs/actions/runners/7/labels"


class TestReadCustomLabels:
    """The drain's check of a runner's labels against the roster's (MCPs board task f19d66a5)."""

    def test_the_custom_labels_are_read_one_per_line(self) -> None:
        runner = FakeRun([ok("lavender\ngpu\n")])
        _test_hooks.run = runner
        assert runner_github.read_custom_labels("wagner-austin/MCPs", 7) == ["lavender", "gpu"]
        assert tuple(runner.calls[0]) == (
            "gh",
            "api",
            _LABELS,
            "--jq",
            '.labels[] | select(.type == "custom") | .name',
        )
        assert runner.timeouts[0] == runner_github.GH_API_TIMEOUT_SECONDS

    def test_a_runner_with_no_custom_label_reads_empty(self) -> None:
        _test_hooks.run = FakeRun([ok("")])
        assert runner_github.read_custom_labels("wagner-austin/MCPs", 7) == []

    def test_a_gh_failure_is_a_local_auth_condition(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 404: Not Found")])
        with pytest.raises(AppError) as fault:
            runner_github.read_custom_labels("wagner-austin/MCPs", 7)
        assert fault.value.code is FleetErrorCode.RUNNER_GITHUB_UNREADABLE
        assert fault.value.message == (
            "gh could not read the labels of runner 7 of wagner-austin/MCPs: HTTP 404: Not Found. "
            "This is local auth or repo permissions, not the fleet."
        )

    def test_a_silent_gh_failure_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])
        with pytest.raises(AppError) as fault:
            runner_github.read_custom_labels("wagner-austin/MCPs", 7)
        assert "<no stderr>" in fault.value.message


class TestClearCustomLabels:
    """The drain's removal of a runner from GitHub's pool."""

    def test_every_custom_label_is_removed_and_none_is_read_back(self) -> None:
        runner = FakeRun([ok("")])
        _test_hooks.run = runner
        runner_github.clear_custom_labels("wagner-austin/MCPs", 7)
        assert tuple(runner.calls[0]) == (
            "gh",
            "api",
            "-X",
            "DELETE",
            _LABELS,
            "--jq",
            runner_github.CUSTOM_LABELS_JQ,
        )
        assert runner.timeouts[0] == runner_github.GH_API_TIMEOUT_SECONDS

    def test_a_label_github_still_lists_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("lavender\n")])
        with pytest.raises(AppError) as fault:
            runner_github.clear_custom_labels("wagner-austin/MCPs", 7)
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_REFUSED
        assert fault.value.message == (
            "GitHub still lists custom labels ['lavender'] on runner 7 of wagner-austin/MCPs "
            f"after removing them. {runner_github.LABELS_REPAIR}"
        )

    def test_a_gh_failure_names_the_resume_that_repairs_it(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 403: Must have admin rights")])
        with pytest.raises(AppError) as fault:
            runner_github.clear_custom_labels("wagner-austin/MCPs", 7)
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_REFUSED
        assert fault.value.message == (
            "gh could not remove the labels of runner 7 of wagner-austin/MCPs: HTTP 403: Must "
            "have admin rights. This is local auth or repo permissions, not the fleet; "
            "fleet-runners --resume yes for the runner's host puts the roster's labels back."
        )

    def test_a_silent_gh_failure_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])
        with pytest.raises(AppError) as fault:
            runner_github.clear_custom_labels("wagner-austin/MCPs", 7)
        assert "<no stderr>" in fault.value.message


class TestSetCustomLabels:
    """The resume's return of a runner to GitHub's pool, with the roster's labels."""

    def test_the_labels_are_set_as_one_array_and_read_back(self) -> None:
        runner = FakeRun([ok("gpu\nlavender\n")])
        _test_hooks.run = runner
        runner_github.set_custom_labels("wagner-austin/MCPs", 7, ["lavender", "gpu"])
        assert tuple(runner.calls[0]) == (
            "gh",
            "api",
            "-X",
            "PUT",
            _LABELS,
            "-f",
            "labels[]=lavender",
            "-f",
            "labels[]=gpu",
            "--jq",
            runner_github.CUSTOM_LABELS_JQ,
        )
        assert runner.timeouts[0] == runner_github.GH_API_TIMEOUT_SECONDS

    def test_an_answer_with_other_labels_is_refused(self) -> None:
        _test_hooks.run = FakeRun([ok("lavender\nstale\n")])
        with pytest.raises(AppError) as fault:
            runner_github.set_custom_labels("wagner-austin/MCPs", 7, ["lavender"])
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_REFUSED
        assert fault.value.message == (
            "GitHub lists custom labels ['lavender', 'stale'] on runner 7 of wagner-austin/MCPs "
            f"after setting them to ['lavender']. {runner_github.LABELS_REPAIR}"
        )

    def test_a_gh_failure_is_refused_with_gh_s_words(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "HTTP 422: Validation Failed")])
        with pytest.raises(AppError) as fault:
            runner_github.set_custom_labels("wagner-austin/MCPs", 7, ["lavender"])
        assert fault.value.code is FleetErrorCode.RUNNER_LABELS_REFUSED
        assert fault.value.message == (
            "gh could not set the labels of runner 7 of wagner-austin/MCPs to ['lavender']: "
            f"HTTP 422: Validation Failed. {runner_github.LABELS_REPAIR}"
        )

    def test_a_silent_gh_failure_says_so(self) -> None:
        _test_hooks.run = FakeRun([failed(1, "")])
        with pytest.raises(AppError) as fault:
            runner_github.set_custom_labels("wagner-austin/MCPs", 7, ["lavender"])
        assert "<no stderr>" in fault.value.message

    def test_no_label_is_refused_before_gh_runs(self) -> None:
        runner = FakeRun([])
        _test_hooks.run = runner
        with pytest.raises(ValueError, match="clear_custom_labels is how"):
            runner_github.set_custom_labels("wagner-austin/MCPs", 7, [])
        assert runner.calls == []
