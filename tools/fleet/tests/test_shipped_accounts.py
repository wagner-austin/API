"""Every Linux machine the fleet logs in to is reached as one account, corvis.

MCPs board task c09d4d29. MCPs fleet-mcp ``netboot-install.ts`` names
``FLEET_ACCOUNT`` corvis as the account every fleet machine is reached as,
and the hub's node-reimage job dials a netbooted machine as that account
before it measures, enrols and rolls it. The NixOS flake makes the account
its roster entry names (``nixos/roster.json``, rendered from this registry),
so a machine file still carrying an older install's account is installed
with that one and the job waits at its reach for ever: colossus kept the
Ubuntu install's ``user`` into its 2026-10-30 netboot, and the tick read
``Permission denied (publickey)`` as corvis every two minutes. This case
reads the shipped registry, so a machine file that names another account is
refused before the netboot can install it. A Linux client the fleet logs
in to nowhere names no account and is outside it.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONObject, require_dict, require_str

from tests._registry import shipped_document

#: The account every fleet machine is reached as (MCPs ``FLEET_ACCOUNT``).
FLEET_ACCOUNT: Final = "corvis"


def linux_accounts(document: JSONObject) -> dict[str, str | None]:
    """Each Linux machine's declared account.

    Args:
        document: A workspace document.

    Returns:
        Every machine whose platform is linux, with its ``user``, or None
        for a client the fleet logs in to nowhere.
    """
    machines = require_dict(document, "machines")
    accounts: dict[str, str | None] = {}
    for name in sorted(machines):
        machine = require_dict(machines, name)
        if require_str(machine, "platform") == "linux":
            user = machine["user"]
            accounts[name] = None if user is None else require_str(machine, "user")
    return accounts


def test_every_linux_machine_the_fleet_logs_in_to_is_reached_as_corvis() -> None:
    accounts = linux_accounts(shipped_document())
    others = {name: user for name, user in accounts.items() if user not in (None, FLEET_ACCOUNT)}

    assert others == {}
    assert (accounts["colossus"], accounts["diphtheria"]) == (FLEET_ACCOUNT, FLEET_ACCOUNT)


def test_a_machine_file_naming_another_account_is_found() -> None:
    document: JSONObject = {
        "machines": {
            "kept": {"platform": "linux", "user": "user"},
            "client": {"platform": "linux", "user": None},
            "fleet": {"platform": "linux", "user": FLEET_ACCOUNT},
            "elsewhere": {"platform": "windows", "user": "Test"},
        }
    }

    assert linux_accounts(document) == {"client": None, "fleet": FLEET_ACCOUNT, "kept": "user"}
