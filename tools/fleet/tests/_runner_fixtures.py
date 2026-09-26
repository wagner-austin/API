"""The runner-host base every roster fixture in these tests carries.

One definition, shaped like lavender's real entry in ``runners.json``, so a
test that builds a host spec does not restate the base it is not about, and
a change to the base contract lands in one fixture rather than in every
test file's own copy.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from fleet.contracts.runner_base import DiskCeiling, HostBase, PinnedDownload
from fleet.core import _test_hooks
from tests.conftest import ok


def a_base() -> HostBase:
    """A lavender-shaped host base.

    Returns:
        The base: two features, the pinned WSL release and image, docker in
        the packages, one machine PATH entry, RemoteSigned, and a 150 GB
        ceiling over a 46 GB baseline.
    """
    return HostBase(
        windows_features=["VirtualMachinePlatform", "Microsoft-Windows-Subsystem-Linux"],
        wsl_msi=PinnedDownload(
            version="2.7.14",
            url="https://github.com/microsoft/WSL/releases/download/2.7.14/wsl.2.7.14.0.x64.msi",
            sha256="db" * 32,
        ),
        rootfs=PinnedDownload(
            version="24.04-20240423",
            url="https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/"
            "ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz",
            sha256="82" * 32,
        ),
        distro_dir="C:/wsl/Ubuntu",
        apt_packages=["build-essential", "docker.io"],
        machine_path_entries=["C:\\Program Files (x86)\\GnuWin32\\bin"],
        execution_policy="RemoteSigned",
        disk=DiskCeiling(ceiling_gb=150, baseline_gb=46, baseline_measured="2026-09-26"),
    )


def base_json() -> dict[str, JSONValue]:
    """The same base as the roster's JSON spells it.

    Returns:
        The encoded mapping, built by hand rather than by the encoder, so a
        decoder test is not a round trip through the code it is testing.
    """
    return {
        "windows_features": ["VirtualMachinePlatform", "Microsoft-Windows-Subsystem-Linux"],
        "wsl_msi": {
            "version": "2.7.14",
            "url": "https://github.com/microsoft/WSL/releases/download/2.7.14/wsl.2.7.14.0.x64.msi",
            "sha256": "db" * 32,
        },
        "rootfs": {
            "version": "24.04-20240423",
            "url": "https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/"
            "ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz",
            "sha256": "82" * 32,
        },
        "distro_dir": "C:/wsl/Ubuntu",
        "apt_packages": ["build-essential", "docker.io"],
        "machine_path_entries": ["C:\\Program Files (x86)\\GnuWin32\\bin"],
        "execution_policy": "RemoteSigned",
        "disk": {"ceiling_gb": 150, "baseline_gb": 46, "baseline_measured": "2026-09-26"},
    }


def quiet_rebuild_answers(
    transcript: str, *, repos: int, linux_base: str = "linux base ready\n"
) -> list[_test_hooks.CommandResult]:
    """Every command a rebuild of an already-built host runs, in order.

    Written out rather than generated from the code, because the sequence is
    the claim: each stage sends its script and runs it, and a distro stage
    sends a payload and a driver before running the driver.

    Args:
        transcript: What the closing audit prints.
        repos: How many repositories the roster names, one token each.
        linux_base: What the Linux base stage prints.

    Returns:
        The replies: windows base (send, run), import (send, run), wsl.conf
        (send, send, run), linux base (send, send, run), one ``gh`` mint per
        repository, provision.ps1 (send, run), provision.sh (send, send,
        run), and the audit (send, run).
    """
    return [
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(linux_base),
        *(ok(f"TOKEN{index}\n") for index in range(repos)),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(""),
        ok(transcript),
    ]


__all__ = ["a_base", "base_json", "quiet_rebuild_answers"]
