"""Finding and verifying the sealed session-verbs release a session verb runs.

The pointer is read from a real file in the running platform's store under
a per-test root, each platform's store is located by naming that platform,
and the verifier's answer is scripted through the command seam
(``tests._session_release_fixtures``). Every refusal is a named code with
nothing run after it: no fallback (MCPs board task c7c2527d).
"""

from __future__ import annotations

import pathlib
import sys

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from fleet.core import _test_hooks, session_release
from tests._queue_fakes import FakeEnv, queue_env
from tests._session_release_fixtures import (
    RELEASE_ID,
    REVISION,
    plant_release,
    point_environment,
    pointer_file,
    pointer_record,
    verified_line,
    verifier,
    verify_call,
    verify_reply,
    write_pointer,
)
from tests.conftest import FakeRun, failed, ok


def refusal(result: session_release.SessionRelease | str) -> str:
    """Narrow a result the test expects to be a refusal.

    Args:
        result: What :func:`~fleet.core.session_release.active_session_release`
            returned.

    Returns:
        The refusal detail.

    Raises:
        AssertionError: When a release was returned instead.
    """
    if not isinstance(result, str):
        raise AssertionError(f"expected a refusal, got release {result['release']}")
    return result


class TestActiveRelease:
    def test_the_pointed_release_is_returned_once_its_own_verifier_names_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        expected = plant_release(tmp_path)
        runner = FakeRun([verify_reply(tmp_path)])
        _test_hooks.run = runner

        assert session_release.active_session_release(sys.platform) == expected
        assert expected["release"] == RELEASE_ID
        assert expected["revision"] == REVISION
        assert expected["registry_dir"] == str(
            expected["root"] / "mcp-shared" / "src" / "source-registry"
        )
        assert runner.calls == [verify_call(tmp_path)]
        assert runner.timeouts == [session_release.VERIFY_TIMEOUT_SECONDS] == [300]
        assert runner.unset_env == [()]
        assert runner.set_env == [()]

    def test_the_pointer_is_where_the_planted_release_is_activated(
        self, tmp_path: pathlib.Path
    ) -> None:
        point_environment(tmp_path)

        assert session_release.pointer_path(sys.platform) == pointer_file(tmp_path)

    def test_on_linux_the_pointer_lives_in_the_store_under_home(
        self, tmp_path: pathlib.Path
    ) -> None:
        _test_hooks.env = FakeEnv({"HOME": str(tmp_path), "LOCALAPPDATA": "/elsewhere"})

        assert session_release.pointer_path("linux") == (
            tmp_path / ".local" / "share" / "corvis" / "session-verbs-releases" / "active.json"
        )

    def test_on_windows_the_pointer_lives_in_the_store_under_local_application_data(
        self, tmp_path: pathlib.Path
    ) -> None:
        _test_hooks.env = FakeEnv({"LOCALAPPDATA": str(tmp_path), "HOME": "/elsewhere"})

        assert session_release.pointer_path("win32") == (
            tmp_path / "Corvis" / "session-verbs-releases" / "active.json"
        )

    @pytest.mark.parametrize(
        ("operating_system", "variable"), [("linux", "HOME"), ("win32", "LOCALAPPDATA")]
    )
    def test_without_the_platforms_root_nothing_is_read_or_run(
        self, operating_system: str, variable: str
    ) -> None:
        _test_hooks.env = queue_env()
        runner = FakeRun([])
        _test_hooks.run = runner

        detail = refusal(session_release.active_session_release(operating_system))

        assert detail == (
            f"{session_release.NO_STORE_BASE_CODE}: {variable} is unset in this runner's "
            f"environment, so the {operating_system} session-verbs release store cannot be "
            "found; nothing was run"
        )
        assert runner.calls == []

    def test_on_a_platform_with_no_store_nothing_is_read_or_run(
        self, tmp_path: pathlib.Path
    ) -> None:
        point_environment(tmp_path)
        runner = FakeRun([])
        _test_hooks.run = runner

        detail = refusal(session_release.active_session_release("darwin"))

        assert detail == (
            f"{session_release.PLATFORM_UNSUPPORTED_CODE}: this runner is on 'darwin', and a "
            "session-verbs release store is kept only on win32, linux; nothing was run"
        )
        assert runner.calls == []

    def test_with_no_pointer_the_release_is_not_active_and_nothing_runs(
        self, tmp_path: pathlib.Path
    ) -> None:
        point_environment(tmp_path)
        runner = FakeRun([])
        _test_hooks.run = runner

        detail = refusal(session_release.active_session_release(sys.platform))

        assert detail == (
            f"{session_release.NOT_ACTIVE_CODE}: {pointer_file(tmp_path)} does not exist, so "
            "no sealed session-verbs release is active on this machine; MCPs activates one on "
            "the host of the sessions duty; nothing was run"
        )
        assert runner.calls == []


class TestPointerRefusals:
    @pytest.mark.parametrize(
        "record",
        [
            {"release": RELEASE_ID, "kind": "session-verbs", "revision": REVISION, "root": "C:/r"},
            {"kind": "session-verbs", "release": RELEASE_ID, "revision": REVISION},
            {
                "kind": "session-verbs",
                "release": RELEASE_ID,
                "revision": REVISION,
                "root": "C:/r",
                "extra": "x",
            },
        ],
    )
    def test_a_pointer_whose_keys_are_not_mcps_own_is_refused(
        self, tmp_path: pathlib.Path, record: JSONObject
    ) -> None:
        path = write_pointer(tmp_path, record)
        _test_hooks.run = FakeRun([])

        assert refusal(session_release.active_session_release(sys.platform)) == (
            f"{session_release.POINTER_INVALID_CODE}: {path} is not an object of "
            "kind, release, revision, root"
        )

    def test_a_pointer_that_is_not_an_object_is_refused(self, tmp_path: pathlib.Path) -> None:
        point_environment(tmp_path)
        path = pointer_file(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text('["session-verbs"]', encoding="utf-8")
        _test_hooks.run = FakeRun([])

        detail = refusal(session_release.active_session_release(sys.platform))

        assert detail.startswith(f"{session_release.POINTER_INVALID_CODE}: {path} is not an object")

    @pytest.mark.parametrize("key", ["kind", "release", "revision", "root"])
    def test_a_pointer_value_that_is_not_a_string_is_refused(
        self, tmp_path: pathlib.Path, key: str
    ) -> None:
        record: dict[str, JSONValue] = dict(pointer_record(tmp_path))
        record[key] = 7
        path = write_pointer(tmp_path, record)
        _test_hooks.run = FakeRun([])

        detail = refusal(session_release.active_session_release(sys.platform))

        code = session_release.POINTER_INVALID_CODE
        assert detail.startswith(f"{code}: {path} must hold strings")

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("kind", "supervisor"),
            ("release", ""),
            ("revision", "58efcb92d"),
            ("revision", REVISION.upper()),
        ],
    )
    def test_a_pointer_naming_another_kind_no_id_or_a_short_commit_is_refused(
        self, tmp_path: pathlib.Path, key: str, value: str
    ) -> None:
        record: dict[str, JSONValue] = dict(pointer_record(tmp_path))
        record[key] = value
        path = write_pointer(tmp_path, record)
        runner = FakeRun([])
        _test_hooks.run = runner

        detail = refusal(session_release.active_session_release(sys.platform))

        assert detail.startswith(f"{session_release.POINTER_INVALID_CODE}: {path} names kind ")
        assert detail.endswith("a session-verbs release with an id and a full commit is required")
        assert runner.calls == []


class TestVerification:
    def test_a_release_that_fails_its_seal_is_refused_with_the_verifiers_reason(
        self, tmp_path: pathlib.Path
    ) -> None:
        plant_release(tmp_path)
        _test_hooks.run = FakeRun([failed(1, "RELEASE_DIGEST_MISMATCH: payload changed\n")])

        detail = refusal(session_release.active_session_release(sys.platform))

        assert detail == (
            f"{session_release.UNSEALED_CODE}: {verifier(tmp_path)} exited 1: "
            "RELEASE_DIGEST_MISMATCH: payload changed"
        )

    @pytest.mark.parametrize(
        "answer",
        [
            "",
            "RELEASE_VERIFIED kind=session-verbs release=other",
            "RELEASE_VERIFIED kind=harness-gate",
        ],
    )
    def test_a_verifier_that_names_anything_but_the_pointers_release_is_refused(
        self, tmp_path: pathlib.Path, answer: str
    ) -> None:
        plant_release(tmp_path)
        _test_hooks.run = FakeRun([ok(f"{answer}\n")])

        detail = refusal(session_release.active_session_release(sys.platform))

        assert detail.startswith(f"{session_release.ANSWER_MISMATCH_CODE}: ")
        assert f"answered {answer!r}" in detail
        assert f"requires {verified_line(tmp_path)!r}; nothing was run" in detail
