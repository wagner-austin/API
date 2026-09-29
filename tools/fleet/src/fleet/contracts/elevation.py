"""Whether a node's elevated runner may claim: its ssh session's token, measured every tick.

MCPs' Task Scheduler installers register tasks that only an administrator may
register, and every fleet build launches as an S4U task at RunLevel Limited
(:mod:`fleet.core.windows_task`), so no dispatched job could execute them
(MCPs board task a98d7083). A node that declares ``elevated``
(:class:`fleet.contracts.node.NodeConfig`) gets a second runner, which claims
only the jobs that require the ``elevated`` tag and launches them at RunLevel
Highest.

WHY THE TOKEN IS MEASURED AND NOT TRUSTED. Registering a Highest task needs the
registering session to hold an administrator's full token. Windows OpenSSH
gives an administrator account one, and a filtered token to anyone else, so
whether the account is in Administrators today decides whether every elevated
build can launch. A declaration alone would stay true after the account was
demoted, and the runner would claim job after job only to fail each at
launch. So the Windows toolchain probe answers one more line, ``integrity``,
from ``WindowsPrincipal.IsInRole(Administrator)``, which is true only for an
unfiltered token (measured 2026-09-29 on serendipity: ``serendipity\\austi``,
``BUILTIN\\Administrators`` enabled, ``Mandatory Label\\High``). An elevated
runner whose probe does not report it claims nothing, and says why.

THE ORDINARY RUNNER IGNORES THE LINE. It launches at Limited, which any
account can register, so what the token is changes nothing it does.
"""

from __future__ import annotations

from typing import Final

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodeConfig
from fleet.contracts.toolchain import ToolReport

#: The toolchain probe's line for the session's token. Its answer is
#: :data:`ADMINISTRATOR` exactly when the token is an administrator's full one.
INTEGRITY_PROBE: Final = "integrity"

#: What the ``integrity`` line answers for an administrator's full token.
ADMINISTRATOR: Final = "administrator"


def elevated_session(reports: tuple[ToolReport, ...]) -> bool:
    """Whether a node's probe reported an administrator's full token.

    Args:
        reports: What the node answered.

    Returns:
        True only when an ``integrity`` line is present and answers
        :data:`ADMINISTRATOR`.
    """
    return any(
        report["name"] == INTEGRITY_PROBE
        and report["present"]
        and report["version"] == ADMINISTRATOR
        for report in reports
    )


def elevation_gap(
    node_name: str, node: NodeConfig, reports: tuple[ToolReport, ...]
) -> AppError[FleetErrorCode] | None:
    """What stands between an elevated runner and a claim, as a value.

    Args:
        node_name: The node's workspace name.
        node: Its declaration, for the host in the message.
        reports: What its toolchain probe answered this tick.

    Returns:
        None when the session holds an administrator's full token;
        otherwise ``NODE_NOT_ELEVATED``, naming the account's fix, not
        raised, because a runner that cannot launch claims nothing rather
        than failing.
    """
    if elevated_session(reports):
        return None
    return AppError(
        FleetErrorCode.NODE_NOT_ELEVATED,
        f"{node_name} ({node['host']}) declares an elevated runner, but its ssh session does "
        "not hold an administrator's token, so every build it launched at RunLevel Highest "
        "would fail to register. Add the ssh account to Administrators on the node, or set "
        "elevated to false",
    )


__all__ = [
    "ADMINISTRATOR",
    "INTEGRITY_PROBE",
    "elevated_session",
    "elevation_gap",
]
