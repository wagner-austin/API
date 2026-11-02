"""Every runner host as a JSON document, for the NixOS flake that builds it.

MCPs board task 9aba868d. WHY A DOCUMENT. lavender, the one runner host, moves from Ubuntu to the
fleet's NixOS, which MCPs ``fleet-mcp/nixos`` builds from files inside its
own source. A NixOS host's ``/etc/systemd/system`` and ``/usr/local`` are
its generation's, so a provision that writes them over ssh has nowhere to
write: the generation must carry every unit, drop-in and script itself. The
runner modules here stay their one source. This document lays out what they
render, each body byte for byte as Ubuntu's provision wrote it, and MCPs
``fleet-reach render`` carries it into the flake, as it carries
``toolchain-pins.json``.

WHAT IT CARRIES, per host of ``runners.json``:

- ``account``: the account the runners run as.
- ``runner``: the pinned release, with its tarball's SHA-256, which the
  store fetches it by.
- ``localBin``: what each runner's ``.path`` ends with.
- ``tokenDirectory``: where each repository's registration token is kept.
- ``packages``: the nixpkgs attributes of the base's packages.
- ``installs``: each install's unit, root, repository, runner name, labels
  and the name its registration token is kept under.
- ``files``: every file with its path, mode and text. These are the
  hygiene, reaper, job-scope, action-cache and CPU-set scripts, their units
  and timers, both slices, every runner unit's five drop-ins and the
  job-scope sudoers rule.
- ``directories``: the action cache, and each writable asset, owned by the
  runners.
- ``timers`` and ``services``: what the generation starts.
- ``assets``: the fetched asset paths the generation must provide.
- ``manualAssets``: what no build can provide.

WHAT IT LEAVES OUT. The imperative steps of the Ubuntu provision. A
generation declares its units from boot, so there is no unit to move into a
slice, no restart to adopt the cache, no live check of a hook and no retire
of a unit the roster dropped. ``tools/fleet-execution-runners`` reads all of
those on the live host.

AT BOOT, NOT AT PROVISION. Two acts the provision ran once, the generation
runs each boot. The first is the CPU set, which needs the host's own
topology (:data:`~fleet.core.runner_cpuset_render.CPUSET_SERVICE_NAME`). The
second is the first hygiene pass: ci-clean's timer is ``Persistent=true``
and so never fires for a run it has no record of, and the runner host's
suite holds each host to a pass within 26 hours.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from platform_core.json_utils import dump_json_str
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall, RunnerSpec
from fleet.core.ci_clean import (
    CI_CLEAN_PATH,
    CI_CLEAN_SCRIPT,
    CI_CLEAN_SERVICE,
    CI_CLEAN_SERVICE_NAME,
    CI_CLEAN_TIMER,
    CI_CLEAN_TIMER_NAME,
)
from fleet.core.runner_action_cache import (
    CACHE_DIR,
    CACHE_DROP_IN,
    CACHE_DROP_IN_NAME,
    FILL_PATH,
    FILL_SCRIPT,
    FILL_SERVICE_NAME,
    FILL_TIMER,
    FILL_TIMER_NAME,
    render_fill_service,
)
from fleet.core.runner_cache_adopt import (
    ADOPT_PATH,
    ADOPT_SCRIPT,
    ADOPT_SERVICE_NAME,
    ADOPT_TIMER,
    ADOPT_TIMER_NAME,
    REPORT_PATH,
    REPORT_SCRIPT,
    render_adopt_service,
)
from fleet.core.runner_cpuset_render import (
    CPUSET_PATH,
    CPUSET_SERVICE,
    CPUSET_SERVICE_NAME,
    render_ci_cpuset_script,
)
from fleet.core.runner_install import (
    LOCAL_BIN,
    RUNNER_ACCOUNT,
    RUNNER_SHA256,
    RUNNER_VERSION,
    TOKEN_DIRECTORY,
    install_root_for,
    token_variable,
)
from fleet.core.runner_job_scope_render import (
    HOOK_DROP_IN,
    HOOK_DROP_IN_NAME,
    JOB_HOOK_PATH,
    JOB_HOOK_SCRIPT,
    JOB_SCOPE_PATH,
    JOB_SCOPE_SCRIPT,
    SUDOERS_PATH,
    SUDOERS_RULE,
)
from fleet.core.runner_reaper_render import (
    KILL_DROP_IN,
    KILL_DROP_IN_NAME,
    REAPER_PATH,
    REAPER_SCRIPT,
    REAPER_SERVICE_NAME,
    REAPER_TIMER,
    REAPER_TIMER_NAME,
    render_reaper_service,
)
from fleet.core.runner_recovery import SYSTEMD_DROP_IN, SYSTEMD_DROP_IN_NAME
from fleet.core.runner_slice_render import (
    LISTENER_SLICE_UNIT,
    LISTENER_SLICE_UNIT_PATH,
    SLICE_DROP_IN,
    SLICE_DROP_IN_NAME,
    SLICE_UNIT_PATH,
    render_slice_unit,
)
from fleet.core.runner_unit_files import drop_in_path
from fleet.core.runner_work_clear import JOB_COMPLETED_HOOK_PATH, JOB_COMPLETED_SCRIPT

#: The committed copy's file name under ``rendered/``.
RUNNER_HOSTS_DOCUMENT: Final = "runner-hosts.json"

#: Where systemd reads the units a generation carries.
UNIT_DIRECTORY: Final = "/etc/systemd/system"

#: An executable script's mode, a unit's or drop-in's, and the sudoers
#: rule's, which sudo refuses when anyone but root may write it.
SCRIPT_MODE: Final = "0755"
UNIT_MODE: Final = "0644"
SUDOERS_MODE: Final = "0440"


class HostFile(TypedDict):
    """One file the generation carries.

    Attributes:
        path: Where it lives on the host.
        mode: Its octal mode, four digits.
        text: Its bytes, newline-terminated.
    """

    path: str
    mode: str
    text: str


class HostDirectory(TypedDict):
    """One directory the generation makes.

    Attributes:
        path: Where.
        mode: Its octal mode, four digits.
        owner: The account that owns it.
    """

    path: str
    mode: str
    owner: str


class DocumentRelease(TypedDict):
    """The pinned runner release.

    Attributes:
        version: :data:`~fleet.core.runner_install.RUNNER_VERSION`.
        sha256: :data:`~fleet.core.runner_install.RUNNER_SHA256`.
    """

    version: str
    sha256: str


class DocumentInstall(TypedDict):
    """One runner install as the generation declares it.

    Attributes:
        service: Its unit's name.
        root: Its install directory, the ``_work`` tree's parent.
        repo: Its repository, ``owner/name``.
        name: Its runner's name.
        labels: Its labels.
        tokenVariable: The name its registration token is kept under.
    """

    service: str
    root: str
    repo: str
    name: str
    labels: list[str]
    tokenVariable: str


class ManualAsset(TypedDict):
    """An asset no build can provide.

    Attributes:
        path: Where it must be.
        reason: The job class that needs it.
    """

    path: str
    reason: str


class RunnerHostDocument(TypedDict):
    """One host's entry: everything its generation carries for its runners.

    Attributes:
        account: The account the runners run as.
        runner: The pinned release.
        localBin: What each runner's ``.path`` ends with.
        tokenDirectory: Where each repository's registration token is kept.
        packages: The nixpkgs attributes of the base's packages, in roster order.
        installs: Every install, in roster order.
        files: Every file, scripts first, then units, then drop-ins.
        directories: Every directory, the action cache first.
        timers: The timers the generation starts.
        services: The services it starts each boot.
        assets: The fetched asset paths it must provide, in roster order.
        manualAssets: What no build can provide, in roster order.
    """

    account: str
    runner: DocumentRelease
    localBin: str
    tokenDirectory: str
    packages: list[str]
    installs: list[DocumentInstall]
    files: list[HostFile]
    directories: list[HostDirectory]
    timers: list[str]
    services: list[str]
    assets: list[str]
    manualAssets: list[ManualAsset]


def _script(path: str, text: str) -> HostFile:
    """An executable script."""
    return HostFile(path=path, mode=SCRIPT_MODE, text=text)


def _unit(name: str, text: str) -> HostFile:
    """A unit file under :data:`UNIT_DIRECTORY`."""
    return HostFile(path=f"{UNIT_DIRECTORY}/{name}", mode=UNIT_MODE, text=text)


def _drop_ins(install: RunnerInstall) -> list[HostFile]:
    """A runner unit's five drop-ins, as the provision wrote them.

    Args:
        install: The install.

    Returns:
        The restart policy, the kill mode, the job hooks, the action cache
        and the listener slice, in the order the provision wrote them.

    Raises:
        ValueError: When the unit name cannot be embedded verbatim.
    """
    bodies = (
        (SYSTEMD_DROP_IN_NAME, SYSTEMD_DROP_IN),
        (KILL_DROP_IN_NAME, KILL_DROP_IN),
        (HOOK_DROP_IN_NAME, HOOK_DROP_IN),
        (CACHE_DROP_IN_NAME, CACHE_DROP_IN),
        (SLICE_DROP_IN_NAME, SLICE_DROP_IN),
    )
    return [
        HostFile(path=drop_in_path(install, name), mode=UNIT_MODE, text=body)
        for name, body in bodies
    ]


def host_files(spec: HostRunnerSpec, reserved_cores: int) -> list[HostFile]:
    """Every file a runner host's generation carries for its runners.

    Args:
        spec: The host's roster entry.
        reserved_cores: The logical cores its roster file reserves for its
            runners, the CPU set's size.

    Returns:
        The scripts, then the units, then each install's drop-ins.

    Raises:
        ValueError: When ``reserved_cores`` is below 1, or a unit name
            cannot be embedded verbatim.
    """
    scripts = [
        _script(CI_CLEAN_PATH, CI_CLEAN_SCRIPT),
        _script(CPUSET_PATH, render_ci_cpuset_script(reserved_cores)),
        _script(REAPER_PATH, REAPER_SCRIPT),
        _script(JOB_SCOPE_PATH, JOB_SCOPE_SCRIPT),
        _script(JOB_HOOK_PATH, JOB_HOOK_SCRIPT),
        _script(JOB_COMPLETED_HOOK_PATH, JOB_COMPLETED_SCRIPT),
        _script(FILL_PATH, FILL_SCRIPT),
        _script(ADOPT_PATH, ADOPT_SCRIPT),
        _script(REPORT_PATH, REPORT_SCRIPT),
        HostFile(path=SUDOERS_PATH, mode=SUDOERS_MODE, text=SUDOERS_RULE),
    ]
    units = [
        _unit(CI_CLEAN_SERVICE_NAME, CI_CLEAN_SERVICE),
        _unit(CI_CLEAN_TIMER_NAME, CI_CLEAN_TIMER),
        HostFile(path=SLICE_UNIT_PATH, mode=UNIT_MODE, text=render_slice_unit(spec["ci_slice"])),
        HostFile(path=LISTENER_SLICE_UNIT_PATH, mode=UNIT_MODE, text=LISTENER_SLICE_UNIT),
        _unit(CPUSET_SERVICE_NAME, CPUSET_SERVICE),
        _unit(REAPER_SERVICE_NAME, render_reaper_service(spec)),
        _unit(REAPER_TIMER_NAME, REAPER_TIMER),
        _unit(FILL_SERVICE_NAME, render_fill_service(spec)),
        _unit(FILL_TIMER_NAME, FILL_TIMER),
        _unit(ADOPT_SERVICE_NAME, render_adopt_service(spec)),
        _unit(ADOPT_TIMER_NAME, ADOPT_TIMER),
    ]
    return [
        *scripts,
        *units,
        *(drop_in for install in spec["installs"] for drop_in in _drop_ins(install)),
    ]


def host_document(spec: HostRunnerSpec, reserved_cores: int) -> RunnerHostDocument:
    """One host's entry.

    Args:
        spec: The host's roster entry.
        reserved_cores: The logical cores its roster file reserves for its
            runners.

    Returns:
        The entry.

    Raises:
        ValueError: As :func:`host_files` describes.
    """
    assets = spec["assets"]
    return RunnerHostDocument(
        account=RUNNER_ACCOUNT,
        runner=DocumentRelease(version=RUNNER_VERSION, sha256=RUNNER_SHA256),
        localBin=LOCAL_BIN,
        tokenDirectory=TOKEN_DIRECTORY,
        packages=list(spec["base"]["nix_packages"]),
        installs=[
            DocumentInstall(
                service=install["service"],
                root=install_root_for(install),
                repo=install["repo"],
                name=install["runner_name"],
                labels=list(install["labels"]),
                tokenVariable=token_variable(install["repo"]),
            )
            for install in spec["installs"]
        ],
        files=host_files(spec, reserved_cores),
        directories=[
            HostDirectory(path=CACHE_DIR, mode=SCRIPT_MODE, owner="root"),
            *(
                HostDirectory(path=asset["path"], mode=SCRIPT_MODE, owner=RUNNER_ACCOUNT)
                for asset in assets
                if asset["writable"]
            ),
        ],
        timers=[CI_CLEAN_TIMER_NAME, REAPER_TIMER_NAME, FILL_TIMER_NAME, ADOPT_TIMER_NAME],
        services=[CPUSET_SERVICE_NAME, CI_CLEAN_SERVICE_NAME],
        assets=[asset["path"] for asset in assets if not asset["writable"] and not asset["manual"]],
        manualAssets=[
            ManualAsset(path=asset["path"], reason=asset["reason"])
            for asset in assets
            if asset["manual"]
        ],
    )


def render_runner_hosts_document(spec: RunnerSpec, reserved: Mapping[str, int]) -> str:
    """The document, as committed.

    Args:
        spec: The runner roster.
        reserved: Each host's ``dispatch.budget.reserved_cores``, by name.

    Returns:
        Every host's entry keyed by its name, as indented JSON ending in one
        newline.

    Raises:
        KeyError: When ``reserved`` names no figure for a host.
        ValueError: As :func:`host_files` describes.
    """
    document = {host["name"]: host_document(host, reserved[host["name"]]) for host in spec["hosts"]}
    return dump_json_str(document, indent=2) + "\n"


__all__ = [
    "RUNNER_HOSTS_DOCUMENT",
    "SCRIPT_MODE",
    "SUDOERS_MODE",
    "UNIT_DIRECTORY",
    "UNIT_MODE",
    "DocumentInstall",
    "DocumentRelease",
    "HostDirectory",
    "HostFile",
    "ManualAsset",
    "RunnerHostDocument",
    "host_document",
    "host_files",
    "render_runner_hosts_document",
]
