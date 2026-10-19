"""Verifying a sealed release with its own verifier, for any kind (MCPs board
task 83102d9d): the check the session verbs and the netboot arm share.

The verifier's answer is scripted through the command seam; what is asserted
is that only the line for this kind, this root and the release directory the
root sits in is taken, and the revision it names is the one returned.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Final

import pytest

from fleet.core import _test_hooks, sealed_release
from tests.conftest import FakeRun, failed, ok

RELEASE_ID: Final = "7fcc96d6ca7f9057183df144ad8ea2bd"
REVISION: Final = "bade610b3a91eba2a7bab589770754d61e63a0e5"


def root(tmp_path: pathlib.Path) -> pathlib.Path:
    """A release checkout as MCPs lays one out, ``<store>/<id>/MCPs``.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The checkout's path.
    """
    return tmp_path / "netboot-releases" / RELEASE_ID / "MCPs"


class TestVerifyRelease:
    def test_a_release_whose_verifier_names_it_is_returned_with_the_seals_revision(
        self, tmp_path: pathlib.Path
    ) -> None:
        checkout = root(tmp_path)
        line = sealed_release.verified_line("netboot", RELEASE_ID, REVISION, checkout)
        runner = FakeRun([ok(f"{line}\n")])
        _test_hooks.run = runner

        assert sealed_release.verify_release(checkout, "netboot") == sealed_release.SealedRelease(
            release=RELEASE_ID, revision=REVISION, root=checkout
        )
        assert runner.calls == [
            (
                sys.executable,
                "-I",
                "-B",
                str(checkout / "packages" / "maketools" / "scripts" / "run.py"),
                "verify-release",
                "--root",
                str(checkout),
                "--kind",
                "netboot",
            )
        ]
        assert runner.timeouts == [sealed_release.VERIFY_TIMEOUT_SECONDS] == [300]

    def test_a_release_that_fails_its_seal_is_refused_with_the_verifiers_reason(
        self, tmp_path: pathlib.Path
    ) -> None:
        checkout = root(tmp_path)
        _test_hooks.run = FakeRun([failed(1, "RELEASE_DIGEST_MISMATCH: payload changed\n")])

        assert sealed_release.verify_release(checkout, "netboot") == (
            f"{sealed_release.UNSEALED_CODE}: "
            f"{checkout / 'packages' / 'maketools' / 'scripts' / 'run.py'} exited 1: "
            "RELEASE_DIGEST_MISMATCH: payload changed"
        )

    @pytest.mark.parametrize(
        ("kind", "release", "revision", "answered_root"),
        [
            ("session-verbs", RELEASE_ID, REVISION, None),
            ("netboot", "0" * 32, REVISION, None),
            ("netboot", RELEASE_ID, REVISION, "/elsewhere/MCPs"),
            ("netboot", RELEASE_ID, "bade610b3", None),
        ],
    )
    def test_a_verification_of_anything_else_is_refused(
        self,
        tmp_path: pathlib.Path,
        kind: str,
        release: str,
        revision: str,
        answered_root: str | None,
    ) -> None:
        """Another kind, another release, another root or a short commit."""
        checkout = root(tmp_path)
        where = pathlib.Path(answered_root) if answered_root is not None else checkout
        answer = sealed_release.verified_line(kind, release, revision, where)
        _test_hooks.run = FakeRun([ok(f"{answer}\n")])

        detail = sealed_release.verify_release(checkout, "netboot")

        assert detail == (
            f"{sealed_release.ANSWER_MISMATCH_CODE}: "
            f"{checkout / 'packages' / 'maketools' / 'scripts' / 'run.py'} answered "
            f"{answer!r} where a netboot release {RELEASE_ID} at {checkout} is required; "
            "nothing was run"
        )

    def test_an_answer_that_is_no_verified_line_is_refused(self, tmp_path: pathlib.Path) -> None:
        checkout = root(tmp_path)
        _test_hooks.run = FakeRun([ok("\n")])

        assert sealed_release.verify_release(checkout, "netboot") == (
            f"{sealed_release.ANSWER_MISMATCH_CODE}: "
            f"{checkout / 'packages' / 'maketools' / 'scripts' / 'run.py'} answered '' where a "
            f"netboot release {RELEASE_ID} at {checkout} is required; nothing was run"
        )
