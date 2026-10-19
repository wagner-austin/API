"""Verifying a sealed MCPs release with its own verifier, for every kind this runner runs.

ONE CHECK, TWO CALLERS. A session verb runs the active sealed session-verbs
release (:mod:`fleet.core.session_release`, MCPs board task c7c2527d), and a
netboot arm runs the installed sealed netboot release
(:mod:`fleet.core.netboot_arm`, MCPs board task 83102d9d). Both verify the
release the same way before anything in it runs, so the verification is
written once, here, rather than a second time beside the first.

WHY THE RELEASE'S OWN MAKETOOLS. The seal is MCPs' digest, one
implementation for every release kind; a second copy of it here would be the
fork that drifts first. The verifier is inside the payload it checks, as
every kind's registration is, and maketools runs on the standard library
alone, so this process's own interpreter runs it with ``-I`` and ``-B``: no
environment of ours reaches it and it writes no bytecode into the sealed
tree. It replaced ``verify-release.ps1``, which only Windows could run.

WHAT IS TRUSTED. The verifier exits 0 only when the payload digest and the
detached commit match the seal, and then prints one line naming the kind,
the release, the revision and the root it checked. That line is read in
full: a verifier that checked another kind, another root, or a release other
than the candidate directory the root sits in is refused, so a release is
never run on the strength of a verification of something else. The revision
it names is the commit the release was cut at, which a caller records.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Final, TypedDict

from fleet.core import _test_hooks
from fleet.core.commit_tree import step_refusal

#: The verifier inside a release, relative to its checkout: maketools'
#: launcher, which puts the release's own maketools source on the path.
VERIFIER: Final = pathlib.PurePosixPath("packages/maketools/scripts/run.py")

#: The maketools command that checks a release's seal and names it.
VERIFY_COMMAND: Final = "verify-release"

#: The verifier's deadline. It hashes the whole payload, the private
#: environment included, which took seconds when measured; a verifier still
#: running after five minutes has stopped answering.
VERIFY_TIMEOUT_SECONDS: Final[int] = 300

#: A full commit id, as a release receipt records it.
REVISION_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")

#: Detail prefix when the release fails its seal.
UNSEALED_CODE: Final = "SEALED_RELEASE_UNSEALED"

#: Detail prefix when the verifier's answer is not the line for this release.
ANSWER_MISMATCH_CODE: Final = "SEALED_RELEASE_ANSWER_MISMATCH"


class SealedRelease(TypedDict):
    """A release whose own verifier checked its seal.

    Attributes:
        release: The release id, the candidate directory MCPs created.
        revision: The full commit the release was checked out at.
        root: The release's checkout.
    """

    release: str
    revision: str
    root: pathlib.Path


def verify_argv(root: pathlib.Path, kind: str) -> tuple[str, ...]:
    """The release's own verifier, run against its checkout.

    Args:
        root: The release's checkout.
        kind: The release kind, as MCPs ``release-kinds.json`` names it.

    Returns:
        This interpreter, isolated and writing no bytecode, on the release's
        maketools launcher with ``verify-release --root <root> --kind <kind>``.
    """
    return (
        sys.executable,
        "-I",
        "-B",
        str(root / VERIFIER),
        VERIFY_COMMAND,
        "--root",
        str(root),
        "--kind",
        kind,
    )


def _around_revision(kind: str, release: str, root: pathlib.Path) -> tuple[str, str]:
    """The verifier's line for one release, before and after its revision.

    Args:
        kind: The release kind.
        release: The release id.
        root: Its checkout.

    Returns:
        ``RELEASE_VERIFIED kind=... release=... revision=`` and `` root=...``.
    """
    return f"RELEASE_VERIFIED kind={kind} release={release} revision=", f" root={root}"


def verified_line(kind: str, release: str, revision: str, root: pathlib.Path) -> str:
    """The line the verifier prints for one release whose seal holds.

    Args:
        kind: The release kind.
        release: The release id.
        revision: The commit it was cut at.
        root: Its checkout.

    Returns:
        ``RELEASE_VERIFIED kind=... release=... revision=... root=...``.
    """
    opening, closing = _around_revision(kind, release, root)
    return f"{opening}{revision}{closing}"


def verify_release(root: pathlib.Path, kind: str) -> SealedRelease | str:
    """Verify one release's seal with its own verifier.

    Args:
        root: The release's checkout, ``<store>/<release id>/MCPs``.
        kind: The kind it must have been sealed as.

    Returns:
        The release, named by the candidate directory its checkout sits in
        and the revision the verifier read from its seal, or a ``CODE:
        message`` refusal detail when the seal does not verify or the
        verifier's answer names any other kind, root or release.
    """
    result = _test_hooks.run(verify_argv(root, kind), timeout_seconds=VERIFY_TIMEOUT_SECONDS)
    verifier = str(root / VERIFIER)
    if result["returncode"] != 0:
        return step_refusal(UNSEALED_CODE, verifier, result)
    answered = result["stdout"].strip()
    release = root.parent.name
    # Everything but the revision is known, so the line is read around it:
    # the kind, release and root it names must be this one's.
    opening, closing = _around_revision(kind, release, root)
    revision = answered.removeprefix(opening).removesuffix(closing)
    if (
        not answered.startswith(opening)
        or not answered.endswith(closing)
        or REVISION_PATTERN.fullmatch(revision) is None
    ):
        return (
            f"{ANSWER_MISMATCH_CODE}: {verifier} answered {answered[:400]!r} where a {kind} "
            f"release {release} at {root} is required; nothing was run"
        )
    return SealedRelease(release=release, revision=revision, root=root)


__all__ = [
    "ANSWER_MISMATCH_CODE",
    "REVISION_PATTERN",
    "UNSEALED_CODE",
    "VERIFIER",
    "VERIFY_COMMAND",
    "VERIFY_TIMEOUT_SECONDS",
    "SealedRelease",
    "verified_line",
    "verify_argv",
    "verify_release",
]
