"""``tankpit-sim-accounts``: set up the sim's database and issue its accounts.

Three actions, each against the ``tankpit_sim`` database whose
connection string the variable ``--database-env`` names:

* ``init`` creates the tables where they are missing.
* ``add --id ID --name NAME [--rank N]`` issues an account and prints its
  token, once: the database keeps only the token's SHA-256, so the token
  printed here is the only copy there will ever be.
* ``list`` prints every account's id, name, rank and decoration levels.

The accounts are this server's own, never tankpit.com's.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from tankpit_bot import _test_hooks
from tankpit_bot.sim.net_store import (
    PostgresAccountBook,
    connect_store,
    encode_levels,
    ensure_schema,
    issue_account,
)

_ACTIONS = ("init", "add", "list")
_FLAGS = frozenset({"--database-env", "--id", "--name", "--rank"})


class AccountsUsageError(ValueError):
    """A command line ``tankpit-sim-accounts`` cannot run (``SIM_ACCOUNTS_USAGE``)."""


def _game_start() -> str:
    """Today, as a join confirm reports when an account began (``Oct. 05, 2026``)."""
    now = datetime.fromtimestamp(_test_hooks.get_current_time_ms() / 1000, tz=UTC)
    return now.strftime("%b. %d, %Y")


def _flags(argv: Sequence[str]) -> dict[str, str]:
    """Read the flags after the action, each with its one value.

    Args:
        argv: The arguments after the action.

    Returns:
        Each flag's value.

    Raises:
        AccountsUsageError: If a flag is unknown or lacks its value, or
            ``--database-env`` is missing.
    """
    values: dict[str, str] = {}
    for index in range(0, len(argv), 2):
        flag = argv[index]
        if flag not in _FLAGS or index + 1 >= len(argv):
            raise AccountsUsageError(
                f"SIM_ACCOUNTS_USAGE: unknown flag or missing value at {flag!r}"
            )
        values[flag] = argv[index + 1]
    if "--database-env" not in values:
        raise AccountsUsageError(
            "SIM_ACCOUNTS_USAGE: --database-env NAME names the variable holding"
            " the tankpit_sim connection string"
        )
    return values


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``tankpit-sim-accounts``.

    Args:
        argv: ``init|add|list`` and its flags; ``sys.argv[1:]`` when None.

    Returns:
        0 once the action is done.

    Raises:
        AccountsUsageError: For an unknown action or flag, or ``add``
            without ``--id`` and ``--name``.
        StoreError: If the connection string is not in the named variable.
        ValueError: If ``--rank`` is not a number.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] not in _ACTIONS:
        raise AccountsUsageError(
            f"SIM_ACCOUNTS_USAGE: the first argument is one of {', '.join(_ACTIONS)}"
        )
    action, values = args[0], _flags(args[1:])
    connection = connect_store(values["--database-env"])
    try:
        ensure_schema(connection)
        book = PostgresAccountBook(connection)
        if action == "add":
            if "--id" not in values or "--name" not in values:
                raise AccountsUsageError("SIM_ACCOUNTS_USAGE: add needs --id ID and --name NAME")
            account, token = issue_account(
                values["--id"], values["--name"], int(values.get("--rank", "0")), _game_start()
            )
            book.add(account)
            sys.stdout.write(
                f"account {account['account_id']} ({account['name']}) issued; its token,"
                f" shown once and stored only as a digest:\n{token}\n"
            )
        elif action == "list":
            for stored in book.accounts():
                sys.stdout.write(
                    f"{stored['account_id']}\t{stored['name']}\trank {stored['rank']}"
                    f"\tdecorations {encode_levels(stored['decorations'])}\n"
                )
        else:
            sys.stdout.write("tankpit_sim tables are in place\n")
    finally:
        connection.close()
    return 0


__all__ = ["AccountsUsageError", "main"]
