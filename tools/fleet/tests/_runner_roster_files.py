"""Roster files for the fleet-runners CLI suites, written as runners.json spells them.

Support module, not a test module. Every CLI case runs
:func:`fleet.cli.runners.main` against a real roster file in its temporary
directory; the entries here are built by hand rather than by the encoder, so
the CLI's decode of them is not a round trip through the code under test.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONValue, dump_json_str

from fleet.cli import runners
from fleet.contracts.runners import HostRunnerSpec
from fleet.core import runner_audit
from tests._runner_fixtures import base_json, ci_slice_json


def raw_host(name: str) -> dict[str, JSONValue]:
    """A valid raw host entry with one install, API's.

    Args:
        name: The host's name, also its ssh alias, its runner's name and its
            label.

    Returns:
        The raw mapping.
    """
    return {
        "name": name,
        "host": name,
        "memory_gb": 31,
        "payload_dir": "/home/corvis/fleet/runner-payloads",
        "gpu_required": False,
        "systemd_timers": [],
        "job_timeout_minutes": 360,
        "installs": [
            {
                "repo": "wagner-austin/API",
                "runner_name": name,
                "service": f"actions.runner.wagner-austin-API.{name}.service",
                "workdir": "/home/gharunner/actions-runner-api-1/_work",
                "labels": [name],
            }
        ],
        "assets": [],
        "base": base_json(),
        "ci_slice": ci_slice_json(),
    }


def manual_asset(reason: str) -> dict[str, JSONValue]:
    """A raw manual asset, the licensed game tree.

    Args:
        reason: Its reason.

    Returns:
        The raw mapping.
    """
    return {
        "path": "/opt/corvis/rw-game/game-lib.jar",
        "sha256": None,
        "writable": False,
        "reason": reason,
        "manual": True,
        "provision_command": None,
    }


def write_roster(tmp_path: pathlib.Path, hosts: list[dict[str, JSONValue]]) -> str:
    """Write a roster file.

    Args:
        tmp_path: The case's directory.
        hosts: The raw host entries.

    Returns:
        The roster's path, as the ``--spec`` flag takes it.
    """
    path = tmp_path / "runners.json"
    path.write_text(dump_json_str({"hosts": hosts}), encoding="utf-8")
    return str(path)


def roster_host(spec_path: str, name: str) -> HostRunnerSpec:
    """One host of a roster file, decoded.

    Args:
        spec_path: The roster's path.
        name: The host.

    Returns:
        Its entry.
    """
    hosts = runners.load_runner_spec(spec_path)["hosts"]
    return next(host for host in hosts if host["name"] == name)


def clean_transcript(spec: HostRunnerSpec) -> str:
    """The audit transcript of a host whose every row passes.

    Args:
        spec: The host.

    Returns:
        One OK line per expected check, ids taken from the audit itself.
    """
    return (
        "\n".join(f"CHECK {check['check_id']} OK" for check in runner_audit.expected_checks(spec))
        + "\n"
    )


__all__ = ["clean_transcript", "manual_asset", "raw_host", "roster_host", "write_roster"]
