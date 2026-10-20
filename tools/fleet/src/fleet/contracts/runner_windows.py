"""The roster's Windows runner hosts: the one machine that stays Windows, and its runners.

MCPs board task 1e9ab738, decision of 2026-10-20 (update 5eab8a7b). Every
runner host is a Linux machine (:mod:`fleet.contracts.runners`) except the
one that carries the CI arms that test Windows itself: chat's PowerShell
Makefile, corvis-stick's and tree-bot's windows arms and MCPs' powershell
job, which no Linux runner can take. That machine is loki, and its runners
are declared here so ``fleet-runners`` installs, audits and prunes them as it
does lavender's, and nothing is installed on it by hand.

A Windows host declares only its installs and where its payloads land. It
carries no CI slice, apt base, systemd timers or assets: each runner is a
Windows service of the machine, run as SYSTEM by ``config.cmd`` (corvis-stick's
images arm mounts a VHDX, which needs it), and restarted by the service
manager when it fails.

EVERY STRING IS EMBEDDED IN A RENDERED POWERSHELL SCRIPT inside single quotes
(:mod:`fleet.core.runner_windows_render`), so the decoder refuses any value
holding a character that would end or reinterpret one, rather than the
renderer escaping it: a roster value that needs escaping is a value nobody
meant.
"""

from __future__ import annotations

import re

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict

#: A value a single-quoted PowerShell string carries verbatim: no quote of
#: either kind, no backtick, no dollar sign and no line break.
_EMBEDDABLE = re.compile(r"^[^'\"`$\r\n]+$")

#: An absolute Windows path, drive letter first, forward slashes, the form
#: every rendered script here spells a path in.
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:/[^'\"`$\r\n\\]*$")


class WindowsRunnerInstall(TypedDict):
    """One GitHub Actions runner installed as a Windows service.

    Attributes:
        repo: The repository it is registered to, ``owner/name``.
        runner_name: The registration's name, e.g. ``loki``.
        service: The Windows service ``config.cmd`` creates
            (``actions.runner.<owner>-<repo>.<name>``), recorded exactly
            because "is the service running" is the audit question.
        root: The install's directory, holding ``config.cmd`` and the
            runner's ``.runner`` file, e.g. ``C:/actions-runner/chat``.
        labels: The custom labels it answers to, e.g. ``["loki"]``.
    """

    repo: str
    runner_name: str
    service: str
    root: str
    labels: list[str]


class WindowsRunnerHost(TypedDict):
    """One Windows machine serving GitHub Actions.

    Attributes:
        name: The machine's fleet name, a machine of ``registry/machines``.
        host: Its ssh destination, an alias from ``~/.ssh/config``, whose
            session holds an elevated token (High Mandatory Level), since
            the install creates services.
        payload_dir: Where the scripts fleet-runners sends land, e.g.
            ``C:/fleet/runner-payloads``.
        installs: Its runners.
    """

    name: str
    host: str
    payload_dir: str
    installs: list[WindowsRunnerInstall]


def _embeddable(value: JSONObject, key: str) -> str:
    """A string field a single-quoted PowerShell string can carry verbatim.

    Args:
        value: The object holding the field.
        key: The field.

    Returns:
        The string.

    Raises:
        JSONTypeError: When it is missing, not a string, empty, or holds a
            character that would end or reinterpret the quoted string.
    """
    text = require_str(value, key)
    if not _EMBEDDABLE.match(text):
        raise JSONTypeError(
            f"{key} must be non-empty with no quote, backtick, dollar sign or line break, "
            f"since a rendered PowerShell script embeds it verbatim; got {text!r}"
        )
    return text


def _windows_path(value: JSONObject, key: str) -> str:
    """An absolute Windows path field in the form the renders spell paths.

    Args:
        value: The object holding the field.
        key: The field.

    Returns:
        The path.

    Raises:
        JSONTypeError: When it is not ``<drive>:/...`` with forward slashes
            and nothing a quoted PowerShell string cannot carry.
    """
    path = require_str(value, key)
    if not _WINDOWS_PATH.match(path):
        raise JSONTypeError(
            f"{key} must be an absolute Windows path with forward slashes, like "
            f"C:/actions-runner/chat; got {path!r}"
        )
    return path


def encode_windows_runner_install(install: WindowsRunnerInstall) -> JSONObject:
    """Encode one Windows runner install.

    Args:
        install: The install.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "repo": install["repo"],
        "runner_name": install["runner_name"],
        "service": install["service"],
        "root": install["root"],
        "labels": list(install["labels"]),
    }


def decode_windows_runner_install(value: JSONValue) -> WindowsRunnerInstall:
    """Decode and validate one Windows runner install.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated install.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            not embeddable, ``repo`` is not ``owner/name``, ``labels`` is
            empty, or ``service`` is not the name ``config.cmd`` gives this
            repository and runner.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"windows runner install must be a JSON object, got {type(value).__name__}"
        )
    repo = _embeddable(value, "repo")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise JSONTypeError(f"repo must be 'owner/name', got {repo!r}")
    runner_name = _embeddable(value, "runner_name")
    service = _embeddable(value, "service")
    expected = f"actions.runner.{repo.replace('/', '-')}.{runner_name}"
    if service != expected:
        raise JSONTypeError(
            f"service must be {expected!r}, the name config.cmd gives {repo}'s runner "
            f"{runner_name}; got {service!r}"
        )
    labels = require_str_list(value, "labels")
    if not labels:
        raise JSONTypeError(
            "labels must be non-empty; a runner with no custom label cannot be targeted "
            "by any workflow here and would sit registered but unreachable"
        )
    for label in labels:
        if not _EMBEDDABLE.match(label) or "," in label:
            raise JSONTypeError(f"a label must hold no comma or quoting character, got {label!r}")
    return WindowsRunnerInstall(
        repo=repo,
        runner_name=runner_name,
        service=service,
        root=_windows_path(value, "root"),
        labels=labels,
    )


def encode_windows_runner_host(host: WindowsRunnerHost) -> JSONObject:
    """Encode one Windows runner host.

    Args:
        host: The host.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "name": host["name"],
        "host": host["host"],
        "payload_dir": host["payload_dir"],
        "installs": [encode_windows_runner_install(each) for each in host["installs"]],
    }


def decode_windows_runner_host(value: JSONValue) -> WindowsRunnerHost:
    """Decode and validate one Windows runner host.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated host.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            invalid, it declares no install, or two installs share a service
            or a root: one service cannot run two runners, and two runners
            cannot share one directory's ``.runner`` file.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"windows runner host must be a JSON object, got {type(value).__name__}"
        )
    installs = [decode_windows_runner_install(each) for each in require_list(value, "installs")]
    if not installs:
        raise JSONTypeError("a windows runner host must declare at least one install")
    for key in ("service", "root"):
        seen = [install[key] for install in installs]
        repeated = sorted({each for each in seen if seen.count(each) > 1})
        if repeated:
            raise JSONTypeError(f"two installs share a {key}: {', '.join(repeated)}")
    return WindowsRunnerHost(
        name=_embeddable(value, "name"),
        host=_embeddable(value, "host"),
        payload_dir=_windows_path(value, "payload_dir"),
        installs=installs,
    )


__all__ = [
    "WindowsRunnerHost",
    "WindowsRunnerInstall",
    "decode_windows_runner_host",
    "decode_windows_runner_install",
    "encode_windows_runner_host",
    "encode_windows_runner_install",
]
