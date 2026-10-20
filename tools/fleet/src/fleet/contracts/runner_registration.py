"""What every GitHub Actions runner install declares, on Linux and on Windows alike.

MCPs board task 1e9ab738, its decision 7cd7182a's review: a runner install
on lavender (:class:`fleet.contracts.runners.RunnerInstall`) and one on loki
(:class:`fleet.contracts.runner_windows.WindowsRunnerInstall`) are the same
registration in two places, so the four facts GitHub knows it by are
declared, validated and spelled here once, and each platform's install adds
only where it lives. Until then each platform declared the four itself and
held them to different rules: the Windows decoder refused a runner name that
was empty, a label holding a comma and a service the configuration would not
create, and the Linux decoder refused none of them, while the registrations
audit in :mod:`fleet.cli._runner_acts` read ``repo`` from both and so relied
on the two staying parallel.

THE SERVICE IS THE CONFIGURATION'S OWN NAME. ``config.sh`` and ``config.cmd``
both name a runner's service ``actions.runner.<owner>-<repo>.<name>``, which
systemd spells with its ``.service`` suffix and the Windows service manager
without one. The roster records it exactly rather than deriving it, because
"is the service running" is the audit question, and the decoder holds it to
the name the configuration gives, so a roster cannot name a service no
install will ever create.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, require_str, require_str_list
from typing_extensions import TypedDict

#: The suffix systemd gives a Linux runner's unit; a Windows service has none.
LINUX_UNIT_SUFFIX: Final = ".service"


class RunnerRegistration(TypedDict):
    """What GitHub knows one runner install by.

    Attributes:
        repo: The repository it is registered to, ``owner/name`` form;
            registrations are per repository, which is why one machine
            carries many.
        runner_name: The registration's name, e.g. ``lavender-2`` or
            ``loki``. Names are per repository: two installs on one machine
            may share a name across repositories without being one install.
        service: The service the configuration creates for it
            (:func:`runner_service_name`), recorded exactly because "is the
            service running" is the audit question.
        labels: The custom labels it answers to, e.g. ``["lavender"]``.
            Labels are the pool: a workflow targets a label, never a
            machine, which is what makes a new machine a join rather than a
            workflow edit.
    """

    repo: str
    runner_name: str
    service: str
    labels: list[str]


def runner_service_name(repo: str, runner_name: str, *, unit_suffix: str) -> str:
    """The service the runner's configuration creates for one registration.

    Args:
        repo: The repository, ``owner/name``.
        runner_name: The registration's name.
        unit_suffix: :data:`LINUX_UNIT_SUFFIX` for a systemd unit, empty for
            a Windows service.

    Returns:
        ``actions.runner.<owner>-<repo>.<name>`` and the suffix.
    """
    return f"actions.runner.{repo.replace('/', '-')}.{runner_name}{unit_suffix}"


def encode_runner_registration(registration: RunnerRegistration) -> JSONObject:
    """Encode the four registration fields of an install.

    Args:
        registration: The install, or its registration.

    Returns:
        JSON-serialisable mapping carrying the four fields.
    """
    return {
        "repo": registration["repo"],
        "runner_name": registration["runner_name"],
        "service": registration["service"],
        "labels": list(registration["labels"]),
    }


def decode_runner_registration(value: JSONObject, *, unit_suffix: str) -> RunnerRegistration:
    """Decode and validate the four registration fields of an install.

    Args:
        value: The install's JSON object.
        unit_suffix: The platform's service suffix (:func:`runner_service_name`).

    Returns:
        The registration.

    Raises:
        JSONTypeError: If a field is missing or mistyped, ``repo`` is not
            ``owner/name``, ``runner_name`` is empty, ``service`` is not the
            name the configuration gives this registration, ``labels`` is
            empty, or a label is empty or holds the comma the configuration
            splits labels on. A runner with no custom label cannot be
            targeted by any workflow here and would sit registered but
            unreachable.
    """
    repo = require_str(value, "repo")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise JSONTypeError(f"repo must be 'owner/name', got {repo!r}")
    runner_name = require_str(value, "runner_name")
    if not runner_name:
        raise JSONTypeError("runner_name must be non-empty")
    service = require_str(value, "service")
    expected = runner_service_name(repo, runner_name, unit_suffix=unit_suffix)
    if service != expected:
        raise JSONTypeError(
            f"service must be {expected!r}, the name the runner's configuration gives "
            f"{repo}'s runner {runner_name}; got {service!r}"
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
    return RunnerRegistration(repo=repo, runner_name=runner_name, service=service, labels=labels)


__all__ = [
    "LINUX_UNIT_SUFFIX",
    "RunnerRegistration",
    "decode_runner_registration",
    "encode_runner_registration",
    "runner_service_name",
]
