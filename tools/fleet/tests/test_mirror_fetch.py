"""Fetching a commit or a companion's ref into the hub's mirror (fleet.core.mirror_fetch).

git runs through the one command hook, so what is asserted is the exact argv
of every git call, its deadline, which of git's refusals becomes which fleet
code, and, since MCPs board task dfcdfb48, how often a lost fetch is sent
again before the job is refused: the clock is pinned and the pause between
attempts moves it, and :class:`_Hanging` moves it by each deadline it is
given, as a fetch ended at its deadline does.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, mirror_fetch
from tests.conftest import DEMO_NOW, FakeClock, FakeRun, failed, ok, pin_clock, timed_out

SHA = "4e3c6bc1d9f0a7b2c3e4f5061728394a5b6c7d8e"
REMOTE = "https://github.com/wagner-austin/API.git"

#: What diphtheria's git said for job 5aadc355 on 2026-10-18.
TLS_LOST = (
    "fatal: unable to access 'https://github.com/wagner-austin/API.git/': "
    "gnutls_handshake() failed: The TLS connection was non-properly terminated."
)

#: What austinpc's git said for jobs 3954f6c1 and b2469547 on 2026-10-17.
RESET = "error: RPC failed; curl 56 Recv failure: Connection was reset\nfatal: early EOF"

#: What diphtheria's git said for jobs dd3019d3 and e88466e8 on 2026-10-20,
#: once its gh login was rejected and the credential helper answered nothing.
NO_CREDENTIAL = "fatal: could not read Username for 'https://github.com': No such device or address"

#: What git 2.43 says when GitHub refuses the token a helper answered with,
#: measured on ziggurat against a private repository on 2026-10-20.
TOKEN_REFUSED = (
    "remote: Invalid username or token. Password authentication is not supported "
    "for Git operations.\n"
    "fatal: Authentication failed for 'https://github.com/wagner-austin/corvis-stick.git/'"
)

#: What git 2.43 says when GitHub refuses an ssh key it does not know, measured
#: the same way.
KEY_REFUSED = (
    "git@github.com: Permission denied (publickey).\n"
    "fatal: Could not read from remote repository.\n\n"
    "Please make sure you have the correct access rights\nand the repository exists."
)

#: Every spelling of a refused credential the cases below drive.
CREDENTIAL_REFUSALS = (NO_CREDENTIAL, TOKEN_REFUSED, KEY_REFUSED)


class _Hanging(FakeRun):
    """A remote that never answers: each fetch is ended at its deadline."""

    def __init__(self, clock: FakeClock) -> None:
        """Answer the mirror's commit probe absent, then every fetch by timing out.

        Args:
            clock: The pinned clock each timed-out fetch moves by its deadline.
        """
        super().__init__([failed(128, "missing")])
        self._clock = clock

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer the probe as scripted, and a fetch by spending its whole deadline.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            The probe's scripted answer, else a timed-out result.
        """
        if "fetch" not in argv:
            return super().__call__(
                argv,
                timeout_seconds=timeout_seconds,
                stdin_bytes=stdin_bytes,
                unset_env=unset_env,
                set_env=set_env,
            )
        self.calls.append(tuple(argv))
        self.timeouts.append(timeout_seconds)
        self._clock.seconds += timeout_seconds
        return timed_out(timeout_seconds)


class TestFetchCommit:
    def test_a_commit_already_held_costs_no_fetch(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([ok("")])
        _test_hooks.run = runner

        mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert runner.calls == [("git", "-C", str(tmp_path), "cat-file", "-e", f"{SHA}^{{commit}}")]

    def test_a_missing_commit_is_fetched_by_sha_without_tags(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([failed(128, "missing"), ok("")])
        _test_hooks.run = runner

        mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert runner.calls[1] == (
            "git",
            "-C",
            str(tmp_path),
            "fetch",
            "--quiet",
            "--no-tags",
            REMOTE,
            SHA,
        )
        assert runner.timeouts == [
            mirror_fetch.INIT_TIMEOUT_SECONDS,
            mirror_fetch.FETCH_TIMEOUT_SECONDS,
        ]

    @pytest.mark.parametrize("marker", mirror_fetch.NOT_ON_REMOTE_MARKERS)
    def test_every_spelling_of_an_unserved_object_is_sha_not_on_remote(
        self, tmp_path: pathlib.Path, marker: str
    ) -> None:
        _test_hooks.run = FakeRun([failed(128, "missing"), failed(128, f"fatal: {marker}: {SHA}")])

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.SHA_NOT_ON_REMOTE
        assert raised.value.message.startswith(f"{REMOTE} does not serve {SHA}")
        assert f"git said: fatal: {marker}: {SHA}" in raised.value.message


class TestALostFetch:
    def test_is_sent_again_after_a_pause_and_done_by_the_next_attempt(
        self, tmp_path: pathlib.Path
    ) -> None:
        clock = pin_clock(DEMO_NOW)
        runner = FakeRun([failed(128, "missing"), failed(128, TLS_LOST), ok("")])
        _test_hooks.run = runner

        mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert [call[3] for call in runner.calls] == ["cat-file", "fetch", "fetch"]
        assert runner.timeouts[1:] == [mirror_fetch.FETCH_TIMEOUT_SECONDS] * 2
        assert clock.seconds == DEMO_NOW + mirror_fetch.FETCH_PAUSE_SECONDS

    def test_one_lost_every_time_is_refused_naming_each_attempt(
        self, tmp_path: pathlib.Path
    ) -> None:
        clock = pin_clock(DEMO_NOW)
        _test_hooks.run = FakeRun(
            [failed(128, "missing"), failed(128, TLS_LOST), failed(128, RESET), failed(1, "  ")]
        )

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message == (
            f"git fetch {REMOTE} {SHA} into {tmp_path}: no attempt of 3 within 900 s was done; "
            f"attempt 1: {TLS_LOST}; attempt 2: {RESET}; attempt 3: (git wrote nothing to stderr)"
        )
        assert clock.seconds == DEMO_NOW + 2 * mirror_fetch.FETCH_PAUSE_SECONDS

    def test_a_remote_that_never_answers_spends_the_budget_and_no_more(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A fetch ended at its full deadline is sent once more with what is
        left, and none after that: three full deadlines would outlast the
        serve the claim runs in."""
        clock = pin_clock(DEMO_NOW)
        hanging = _Hanging(clock)
        _test_hooks.run = hanging

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        left = mirror_fetch.FETCH_BUDGET_SECONDS - 600 - mirror_fetch.FETCH_PAUSE_SECONDS
        assert hanging.timeouts[1:] == [600, left]
        assert clock.seconds == DEMO_NOW + mirror_fetch.FETCH_BUDGET_SECONDS
        assert raised.value.message.endswith(
            f"no attempt of 2 within 900 s was done; attempt 1: timed out after 600 s; "
            f"attempt 2: timed out after {left} s"
        )

    def test_an_unserved_object_after_a_lost_attempt_is_refused_at_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        pin_clock(DEMO_NOW)
        runner = FakeRun(
            [failed(128, "missing"), failed(128, RESET), failed(128, f"fatal: not our ref {SHA}")]
        )
        _test_hooks.run = runner

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.SHA_NOT_ON_REMOTE
        assert len(runner.calls) == 3


class TestARefusedCredential:
    """A remote that refuses the hub's credential refuses every attempt the
    same way, so the fetch is not sent again and the refusal sends its reader
    to the control host's credential, not to the network or the commit."""

    @pytest.mark.parametrize("stderr", CREDENTIAL_REFUSALS)
    def test_a_commit_fetch_is_refused_at_once_by_name(
        self, tmp_path: pathlib.Path, stderr: str
    ) -> None:
        clock = pin_clock(DEMO_NOW)
        runner = FakeRun([failed(128, "missing"), failed(128, stderr)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.REMOTE_CREDENTIAL_REJECTED
        assert raised.value.message == (
            f"{REMOTE} refused the credential this hub's git sent it, or asked for one "
            f"its credential helper did not answer; nothing from it can be exported "
            f"until that credential is renewed on the control host, and sending the "
            f"fetch again would send the same one. git said: {stderr.strip()[:300]}"
        )
        assert [call[3] for call in runner.calls] == ["cat-file", "fetch"]
        assert clock.seconds == DEMO_NOW

    @pytest.mark.parametrize("stderr", CREDENTIAL_REFUSALS)
    def test_a_ref_fetch_is_refused_at_once_by_name(
        self, tmp_path: pathlib.Path, stderr: str
    ) -> None:
        clock = pin_clock(DEMO_NOW)
        runner = FakeRun([failed(128, stderr)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_ref(tmp_path, REMOTE, "main")

        assert raised.value.code is FleetErrorCode.REMOTE_CREDENTIAL_REJECTED
        assert raised.value.message.startswith(
            f"{REMOTE} refused the credential this hub's git sent it"
        )
        assert raised.value.message.endswith(f"git said: {stderr.strip()[:300]}")
        assert len(runner.calls) == 1
        assert clock.seconds == DEMO_NOW

    def test_one_refused_after_a_lost_attempt_is_refused_at_once(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A login that lapses between two attempts ends the fetch at the
        attempt that meets it, whatever the attempts before it said."""
        clock = pin_clock(DEMO_NOW)
        runner = FakeRun([failed(128, "missing"), failed(128, RESET), failed(128, NO_CREDENTIAL)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.REMOTE_CREDENTIAL_REJECTED
        assert len(runner.calls) == 3
        assert clock.seconds == DEMO_NOW + mirror_fetch.FETCH_PAUSE_SECONDS

    def test_every_marker_is_one_git_was_measured_printing(self) -> None:
        """A marker no measured refusal carries would be a rule for words git
        never says; each one here matches exactly one of the three."""
        assert [
            sum(marker in refusal for refusal in CREDENTIAL_REFUSALS)
            for marker in mirror_fetch.CREDENTIAL_REFUSED_MARKERS
        ] == [1, 1, 1]

    def test_a_repository_github_will_not_name_is_still_sent_again(
        self, tmp_path: pathlib.Path
    ) -> None:
        """GitHub answers a repository the credential cannot see and one that
        does not exist with the same words, so they are not read as a refused
        credential."""
        pin_clock(DEMO_NOW)
        unnamed = "ERROR: Repository not found.\nfatal: Could not read from remote repository."
        runner = FakeRun([failed(128, "missing")] + [failed(128, unnamed)] * 3)
        _test_hooks.run = runner

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_commit(tmp_path, REMOTE, SHA)

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert len(runner.calls) == 1 + mirror_fetch.FETCH_ATTEMPTS


class TestFetchRef:
    def test_the_tip_is_fetched_into_a_ref_and_resolved_to_a_commit(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Into a ref rather than off FETCH_HEAD, so the commit is referenced
        between the fetch and the bundle; and never skipped the way a sha is,
        because a ref's tip is a moving answer."""
        runner = FakeRun([ok(""), ok(f"{SHA}\n")])
        _test_hooks.run = runner

        assert mirror_fetch.fetch_ref(tmp_path, REMOTE, "main") == SHA
        assert runner.calls == [
            (
                "git",
                "-C",
                str(tmp_path),
                "fetch",
                "--quiet",
                "--no-tags",
                "--force",
                REMOTE,
                f"+main:{mirror_fetch.COMPANION_REF}",
            ),
            ("git", "-C", str(tmp_path), "rev-parse", f"{mirror_fetch.COMPANION_REF}^{{commit}}"),
        ]
        assert runner.timeouts == [
            mirror_fetch.FETCH_TIMEOUT_SECONDS,
            mirror_fetch.INIT_TIMEOUT_SECONDS,
        ]

    @pytest.mark.parametrize("marker", mirror_fetch.NOT_ON_REMOTE_MARKERS)
    def test_a_ref_the_remote_lacks_is_its_own_code_and_not_the_shas(
        self, tmp_path: pathlib.Path, marker: str
    ) -> None:
        """A sha the remote lacks is the submitter's unpushed commit; a ref it
        lacks is a registry line that is wrong for every job of that project,
        so the two send their readers to different places."""
        _test_hooks.run = FakeRun([failed(128, f"fatal: {marker} main")])

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_ref(tmp_path, REMOTE, "main")

        assert raised.value.code is FleetErrorCode.COMPANION_REF_NOT_ON_REMOTE
        assert raised.value.message.startswith(f"{REMOTE} does not serve 'main'")
        assert "which the registry declares as a companion ref" in raised.value.message
        assert f"git said: fatal: {marker} main" in raised.value.message

    def test_a_lost_fetch_is_sent_again_as_a_commit_s_is(self, tmp_path: pathlib.Path) -> None:
        pin_clock(DEMO_NOW)
        runner = FakeRun([failed(128, RESET), ok(""), ok(f"{SHA}\n")])
        _test_hooks.run = runner

        assert mirror_fetch.fetch_ref(tmp_path, REMOTE, "main") == SHA
        assert [call[3] for call in runner.calls] == ["fetch", "fetch", "rev-parse"]

    def test_one_lost_every_time_is_refused_naming_each_attempt(
        self, tmp_path: pathlib.Path
    ) -> None:
        pin_clock(DEMO_NOW)
        _test_hooks.run = FakeRun([failed(128, RESET)] * mirror_fetch.FETCH_ATTEMPTS)

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_ref(tmp_path, REMOTE, "main")

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message.startswith(
            f"git fetch {REMOTE} main into {tmp_path}: no attempt of 3 within 900 s was done; "
            f"attempt 1: {RESET}; "
        )

    def test_a_ref_that_resolves_to_no_commit_is_an_export_failure(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A tag object pointing at a tree, or a ref written by something that
        is not git: the fetch succeeds and there is still no commit to
        bundle, which is not the same as a fetch that failed."""
        _test_hooks.run = FakeRun([ok(""), failed(128, "fatal: not a valid object name")])

        with pytest.raises(AppError) as raised:
            mirror_fetch.fetch_ref(tmp_path, REMOTE, "main")

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message == (
            f"git rev-parse {mirror_fetch.COMPANION_REF} in {tmp_path}: "
            "fatal: not a valid object name"
        )
