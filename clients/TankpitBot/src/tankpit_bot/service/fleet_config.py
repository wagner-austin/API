"""What the fleet offers as configuration, and how it validates it.

Split out of :mod:`tankpit_bot.service.fleet_manager` 2026-09-01 at
the 600-line ceiling. The cut is by role: this module answers "what
may an operator ask for" -- the port, the accounts, the rooms, the
colours, the roles, and the instance name an account implies -- while
``fleet_manager`` owns the registry of running bots and their
lifecycle. Nothing here holds state; every answer is read from config
at the moment it is asked for.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.browser.accounts import _ACCOUNTS_PATH, load_accounts
from tankpit_bot.fleetshare.types import ENGAGEMENT_DOCTRINES, FLEET_ROLES, FleetRole
from tankpit_bot.runtime_artifacts import TANK_REGISTRY_PATH
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.types.constants import TROOP_COLOR_NAMES
from tankpit_bot.types.rooms import LOBBY_ROOMS

log = get_logger(__name__)

FLEET_PORT_DEFAULT = 27300


def resolve_fleet_port() -> int:
    """Resolve the fleet manager's port from the environment.

    Returns:
        ``TANKPIT_FLEET_PORT`` when set, else :data:`FLEET_PORT_DEFAULT`.

    Raises:
        ValueError: If the value is not an integer in [1024, 65535].
    """
    raw = top_hooks.get_env("TANKPIT_FLEET_PORT")
    if raw is None or raw == "":
        return FLEET_PORT_DEFAULT
    port = int(raw)
    if not 1024 <= port <= 65535:
        raise ValueError(f"TANKPIT_FLEET_PORT {port} outside [1024, 65535]")
    return port


def resolve_role(role: str) -> FleetRole:
    """Resolve a spawn request's role selector to a fleet role.

    Args:
        role: Role selector; empty means fighter — the full doctrine
            is the primary configuration, a gatherer is an explicit
            operator choice ([[fleet-coordination]]).

    Returns:
        The resolved role.

    Raises:
        FleetError: If the selector is not a fleet role.
    """
    candidate = role or "fighter"
    for known in FLEET_ROLES:
        if candidate == known:
            return known
    known_roles = ", ".join(FLEET_ROLES)
    raise FleetError(f"role {role!r} is not a fleet role (one of: {known_roles})")


def resolve_troop(troop: str) -> str:
    """Resolve a spawn request's color selector to a tank color name.

    Args:
        troop: Color name, or ``""`` to keep the account's own default
            tank color for the map it joins.

    Returns:
        The validated color name, or ``""``.

    Raises:
        FleetError: If the selector is not a tank color.
    """
    if troop == "":
        return ""
    for known in TROOP_COLOR_NAMES:
        if troop == known:
            return known
    known_colors = ", ".join(TROOP_COLOR_NAMES)
    raise FleetError(f"troop {troop!r} is not a tank color (one of: {known_colors})")


def resolve_doctrine(doctrine: str) -> str:
    """Resolve a spawn request's doctrine selector.

    Validated HERE as well as in the child's own resolver, because a
    typo caught at spawn is an HTTP 409 the operator reads, while the
    same typo caught in the child is a process that starts, logs a
    ValueError and dies with no tank in the world.

    Args:
        doctrine: Doctrine name, or ``""`` to take the child's default
            (skirmish, the unset behaviour).

    Returns:
        The validated doctrine name, or ``""``.

    Raises:
        FleetError: If the selector is not an engagement doctrine.
    """
    if doctrine == "":
        return ""
    for known in ENGAGEMENT_DOCTRINES:
        if doctrine == known:
            return known
    known_doctrines = ", ".join(ENGAGEMENT_DOCTRINES)
    raise FleetError(
        f"doctrine {doctrine!r} is not an engagement doctrine (one of: {known_doctrines})"
    )


def configured_accounts() -> list[str]:
    """Return the configured account usernames.

    Accounts are CONFIG (``accounts.json``), never free text — the
    spawn surface only accepts a selector from this list, and the
    control page renders it as a dropdown. Usernames only; passwords
    never leave the file.

    Returns:
        Usernames in file order (the first is the default), empty
        when no accounts file exists.
    """
    if not top_hooks.path_exists(_ACCOUNTS_PATH):
        return []
    return [account["username"] for account in load_accounts(_ACCOUNTS_PATH)]


def lobby_rooms() -> list[str]:
    """Return the room selectors the control page offers.

    The lobby lists two rooms, and the world's display name carries
    the current map, so the page offers the durable PREFIXES the join
    resolver matches on ([[game-rules]],
    :mod:`tankpit_bot.types.rooms`) rather than asking a human to type
    a name that rotates. Spawn still accepts any selector: this list
    is what the dropdown shows, not a closed set.

    Returns:
        Room selectors in lobby order; the first is the default.
    """
    return list(LOBBY_ROOMS)


def troop_colors() -> list[str]:
    """Return the tank colors the control page offers.

    Four colors, in TEAM ID order — the index is the wire's team id,
    so the list doubles as the name->id table the spawn environment
    converts through. An account holds FOUR TANKS PER WORLD, one per
    color, each with its own RANK, inventory, fuel and points (awards
    alone are shared) — so picking a color picks WHICH TANK plays, not
    a skin, and a fresh color starts that world from scratch. The
    worlds are independent: four on the main world plus four on
    Practice. Switching is throttled per world — 5 minutes between
    exiting a world and re-entering it on a different color
    ([[game-rules]]).

    Returns:
        Color names in team-id order.
    """
    return list(TROOP_COLOR_NAMES)


def engagement_doctrines() -> list[str]:
    """Return the engagement doctrines the control page offers.

    A doctrine names how a bot picks and presses a fight, not what it
    collects: ``skirmish`` is today's behaviour and the default,
    ``swarm`` musters, ``duelist`` holds a single opponent, ``passive``
    declines to open one. Landed 2026-09-02 alongside the doctrines
    themselves, so the operator states it at spawn instead of setting
    ``TANKPIT_DOCTRINE`` by hand.

    Vocabulary order is the doctrine tuple's own, and the first entry
    is what an unset environment resolves to.

    Returns:
        Doctrine names, the first being the default.
    """
    return list(ENGAGEMENT_DOCTRINES)


def tank_registry() -> JSONObject:
    """Return the measured per-colour tank registry.

    An account holds four tanks per world with INDEPENDENT rank
    ([[game-rules]]), and rank sets both the fuel cap
    (``1000 + 100*rank``) and the radar radius (``2 + rank//3``), so
    which colour an operator picks decides how strong the tank is.
    Nothing on the wire reports the ranks of colours the account is
    not currently playing -- the lobby names only the last-played one
    -- so this is MEASURED state, filled by entering each colour once,
    not something the page can derive.

    Returns:
        The registry as stored, or an empty object when the file is
        absent (an operator who has never run the census sees an empty
        panel, not an error).
    """
    try:
        raw = top_hooks.read_text(TANK_REGISTRY_PATH)
    except OSError as error:
        log.info("Fleet: no tank registry at %s: %s", TANK_REGISTRY_PATH, error)
        return {}
    return narrow_json_to_dict(load_json_str(raw))


def derive_instance(account: str) -> str:
    """Derive the instance name from the account — programmatic, reliable.

    One account can hold at most one live tank (the game refuses a
    second login), so the account IS the natural bot identity: the
    instance is its username lowered and sanitized to the namespace
    grammar. No account configured falls back to ``bot``. Callers may
    still name instances explicitly through the API; the control page
    never asks a human to invent one.

    Args:
        account: Selected account username, empty for the default.

    Returns:
        A valid instance name.
    """
    configured = configured_accounts()
    source = account or (configured[0] if configured else "bot")
    cleaned = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "-" for ch in source.lower()
    )[:32]
    if not cleaned or not (cleaned[0].isascii() and cleaned[0].isalnum()):
        cleaned = f"b{cleaned}"[:32]
    return cleaned


__all__ = [
    "FLEET_PORT_DEFAULT",
    "configured_accounts",
    "derive_instance",
    "engagement_doctrines",
    "lobby_rooms",
    "resolve_doctrine",
    "resolve_fleet_port",
    "resolve_role",
    "resolve_troop",
    "tank_registry",
    "troop_colors",
]
