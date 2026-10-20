"""What every netboot tick test lays out and expects: the installed release
and env file as real files, the release verifier's answer, and the adopt
pass every netboot tick ends with (MCPs board tasks 83102d9d and 49d366ac).
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.cli import agent
from fleet.core import _test_hooks, netboot_adopt, netboot_arm, sealed_release
from tests.conftest import ok

#: The installed release's candidate directory and the commit it was cut at.
RELEASE_ID: Final = "7fcc96d6ca7f9057183df144ad8ea2bd"
REVISION: Final = "bade610b3a91eba2a7bab589770754d61e63a0e5"

#: The tailnet API key the workspace env file holds for the netboot lane.
API_KEY: Final = "tskey-api-kTEST1234CNTRL-secret"

#: What ``adopt`` prints when no check-in waits.
NOTHING_PENDING: Final = (
    "NETBOOT ADOPT nothing pending: 0 of 1 arm(s) checked in within the window with no record"
)


def place(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Lay out the installed release and the env file, keyed by the flag naming each.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The release checkout, the served root and the env file.
    """
    release = tmp_path / "netboot-releases" / RELEASE_ID / "MCPs"
    release.mkdir(parents=True)
    env_file = tmp_path / "MCPs" / ".env"
    env_file.parent.mkdir()
    env_file.write_text(
        f"POSTGRES_PASSWORD=unrelated\nTAILSCALE_API_KEY={API_KEY}\nVM_HOST=lavender-wsl\n",
        encoding="utf-8",
    )
    return {
        agent.NETBOOT_RELEASE_FLAG: release,
        agent.SERVED_ROOT_FLAG: tmp_path / "srv" / "corvis-netboot",
        agent.ENV_FILE_FLAG: env_file,
    }


def verify_reply(release: pathlib.Path) -> _test_hooks.CommandResult:
    """The release's verifier, answering that its seal holds.

    Args:
        release: The release's checkout.

    Returns:
        Exit 0 with the line MCPs maketools prints.
    """
    return ok(f"{sealed_release.verified_line('netboot', RELEASE_ID, REVISION, release)}\n")


def netboot_place(
    paths: dict[str, pathlib.Path], registry: pathlib.Path
) -> netboot_arm.NetbootPlace:
    """The place a tick reads from :func:`place`'s flags.

    Args:
        paths: What :func:`place` laid out.
        registry: The registry directory the tick loaded.

    Returns:
        The place, with ``node`` as the node binary.
    """
    return netboot_arm.NetbootPlace(
        release=paths[agent.NETBOOT_RELEASE_FLAG],
        nodejs="node",
        served_root=paths[agent.SERVED_ROOT_FLAG],
        env_file=paths[agent.ENV_FILE_FLAG],
        registry=registry,
    )


def sealed(paths: dict[str, pathlib.Path]) -> sealed_release.SealedRelease:
    """The release :func:`verify_reply` reports verified.

    Args:
        paths: What :func:`place` laid out.

    Returns:
        The release, its revision and its checkout.
    """
    return sealed_release.SealedRelease(
        release=RELEASE_ID, revision=REVISION, root=paths[agent.NETBOOT_RELEASE_FLAG]
    )


def adopt_replies(
    paths: dict[str, pathlib.Path], printed: str = NOTHING_PENDING
) -> list[_test_hooks.CommandResult]:
    """The adopt pass's two answers: the seal holds, then ``adopt`` exits 0.

    Args:
        paths: What :func:`place` laid out.
        printed: What ``adopt`` prints.

    Returns:
        The verifier's reply and ``adopt``'s.
    """
    return [verify_reply(paths[agent.NETBOOT_RELEASE_FLAG]), ok(f"{printed}\n")]


def adopt_calls(paths: dict[str, pathlib.Path], registry: pathlib.Path) -> list[tuple[str, ...]]:
    """The adopt pass's two commands: the release's verifier, then its ``adopt``.

    Args:
        paths: What :func:`place` laid out.
        registry: The registry directory the tick loaded.

    Returns:
        Both argvs, in order.
    """
    return [
        sealed_release.verify_argv(paths[agent.NETBOOT_RELEASE_FLAG], "netboot"),
        netboot_adopt.adopt_argv(netboot_place(paths, registry), sealed(paths)),
    ]
