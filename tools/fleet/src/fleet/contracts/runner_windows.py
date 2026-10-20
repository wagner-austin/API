"""The roster's Windows runner hosts: the one machine that stays Windows, and its runners.

MCPs board task 1e9ab738, decision of 2026-10-20 (update 5eab8a7b). Every
runner host is a Linux machine (:mod:`fleet.contracts.runners`) except the
one that carries the CI arms that test Windows itself: chat's PowerShell
Makefile, corvis-stick's and tree-bot's windows arms and MCPs' powershell
job, which no Linux runner can take. That machine is loki, and its runners
are declared here so ``fleet-runners`` installs, audits and prunes them as it
does lavender's, and nothing is installed on it by hand.

A Windows host declares its installs, where its payloads land and the
machine settings its runners need (:mod:`fleet.contracts.runner_windows_base`).
It carries no CI slice, apt base, systemd timers or assets: each runner is a
Windows service of the machine, run as SYSTEM by ``config.cmd`` (corvis-stick's
images arm mounts a VHDX, which needs it), and restarted by the service
manager when it fails.

EVERY STRING IS EMBEDDED IN A RENDERED POWERSHELL SCRIPT, and the renderer
refuses a value it cannot carry verbatim through
:func:`fleet.core.script_values.scriptable`, as every renderer here does;
the decoder holds each value to its shape.
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

from fleet.contracts.runner_windows_base import (
    WindowsHostBase,
    decode_windows_host_base,
    encode_windows_host_base,
)

#: An absolute Windows path, drive letter first, forward slashes, the form
#: every rendered script here spells a path in.
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:/[^\\]*$")

#: An exact CPython version, the only form the NuGet package is published
#: under, and the only one that cannot drift between two seeds.
_PYTHON_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


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
        python_toolcache: The exact CPython versions seeded into its tool
            cache, so ``actions/setup-python`` finds each rather than
            installing it (:mod:`fleet.core.runner_windows_python_render`);
            empty for a runner whose jobs set up no Python.
    """

    repo: str
    runner_name: str
    service: str
    root: str
    labels: list[str]
    python_toolcache: list[str]


class WindowsRunnerHost(TypedDict):
    """One Windows machine serving GitHub Actions.

    Attributes:
        name: The machine's fleet name, a machine of ``registry/machines``.
        host: Its ssh destination, an alias from ``~/.ssh/config``, whose
            session holds an elevated token (High Mandatory Level), since
            the install creates services.
        payload_dir: Where the scripts fleet-runners sends land, e.g.
            ``C:/fleet/runner-payloads``.
        base: The machine settings every runner on it needs.
        installs: Its runners.
    """

    name: str
    host: str
    payload_dir: str
    base: WindowsHostBase
    installs: list[WindowsRunnerInstall]


def _nonempty(value: JSONObject, key: str) -> str:
    """A string field that must say something.

    Args:
        value: The object holding the field.
        key: The field.

    Returns:
        The string.

    Raises:
        JSONTypeError: When it is missing, not a string or empty.
    """
    text = require_str(value, key)
    if not text:
        raise JSONTypeError(f"{key} must be non-empty")
    return text


def _windows_path(value: JSONObject, key: str) -> str:
    """An absolute Windows path field in the form the renders spell paths.

    Args:
        value: The object holding the field.
        key: The field.

    Returns:
        The path.

    Raises:
        JSONTypeError: When it is not ``<drive>:/...`` with forward slashes.
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
        "python_toolcache": list(install["python_toolcache"]),
    }


def decode_windows_runner_install(value: JSONValue) -> WindowsRunnerInstall:
    """Decode and validate one Windows runner install.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated install.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            empty, ``repo`` is not ``owner/name``, ``labels`` is empty or a
            label holds the comma ``config.cmd`` splits labels on,
            ``service`` is not the name ``config.cmd`` gives this repository
            and runner, or a tool cache version is not exact or is named
            twice.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"windows runner install must be a JSON object, got {type(value).__name__}"
        )
    repo = _nonempty(value, "repo")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise JSONTypeError(f"repo must be 'owner/name', got {repo!r}")
    runner_name = _nonempty(value, "runner_name")
    service = _nonempty(value, "service")
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
        if not label or "," in label:
            raise JSONTypeError(f"a label must be non-empty and hold no comma, got {label!r}")
    toolcache = require_str_list(value, "python_toolcache")
    for version in toolcache:
        if not _PYTHON_VERSION.match(version):
            raise JSONTypeError(
                f"python_toolcache versions must be exact X.Y.Z, got {version!r}; a floating "
                "version would seed whichever release was newest the day it ran"
            )
    if len(set(toolcache)) != len(toolcache):
        raise JSONTypeError(f"python_toolcache names a version twice: {toolcache}")
    return WindowsRunnerInstall(
        repo=repo,
        runner_name=runner_name,
        service=service,
        root=_windows_path(value, "root"),
        labels=labels,
        python_toolcache=toolcache,
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
        "base": encode_windows_host_base(host["base"]),
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
        name=_nonempty(value, "name"),
        host=_nonempty(value, "host"),
        payload_dir=_windows_path(value, "payload_dir"),
        base=decode_windows_host_base(value.get("base")),
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
