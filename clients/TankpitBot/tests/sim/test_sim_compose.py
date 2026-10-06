"""The sim server's compose file agrees with the server it starts.

``sim-server.compose.json`` puts ``tankpit-sim-serve`` behind the
platform's Traefik v3. It is JSON, which compose reads as it reads YAML,
so this package's own JSON decoders hold it to the CLI: its command must
parse as ``tankpit-sim-serve``'s flags, the port Traefik balances to must
be the port the server binds, the variable the server reads must be the
one the environment sets, and the network Traefik routes on must be the
external network the service joins.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_dict,
    require_list,
    require_str,
)

from tankpit_bot.sim.net_server import AccountSource, ServeArgs, build_host, parse_serve_args
from tests.sim._net_client import account_book

_COMPOSE = Path(__file__).resolve().parents[2] / "sim-server.compose.json"
_ROUTER = "traefik.http.routers.tankpit-sim"


def _service() -> JSONObject:
    """The compose file's one service."""
    compose = narrow_json_to_dict(load_json_str(_COMPOSE.read_text(encoding="utf-8")))
    return require_dict(require_dict(compose, "services"), "sim")


def _labels() -> dict[str, str]:
    """The service's labels, key to value."""
    labels: dict[str, str] = {}
    for label in require_list(_service(), "labels"):
        key, value = narrow_json_to_str(label).split("=", 1)
        labels[key] = value
    return labels


def _args() -> ServeArgs:
    """The service's command, read as the server reads it."""
    command = [narrow_json_to_str(word) for word in require_list(_service(), "command")]
    assert command[0] == "tankpit-sim-serve"
    return parse_serve_args(command[1:])


def test_the_command_serves_from_the_database_its_environment_names() -> None:
    """The variable the server reads is the one the service sets, on the platform's Postgres."""
    args = _args()
    assert args.accounts == AccountSource("database", "TANKPIT_SIM_DATABASE_URL")
    dsn = require_str(require_dict(_service(), "environment"), "TANKPIT_SIM_DATABASE_URL")
    assert dsn.endswith("@platform-postgres:5432/tankpit_sim}")
    assert (args.bind, args.ticks) == ("0.0.0.0", None)


def test_traefik_balances_to_the_port_the_server_binds() -> None:
    """The load balancer's port and the --port flag are one number."""
    assert _labels()["traefik.http.services.tankpit-sim.loadbalancer.server.port"] == str(
        _args().port
    )


def test_the_route_strips_the_prefix_it_matches() -> None:
    """The server sees the socket at its root, as it serves it."""
    labels = _labels()
    assert labels["traefik.enable"] == "true"
    assert labels[f"{_ROUTER}.rule"] == "PathPrefix(`/tankpit-sim`)"
    middleware = labels[f"{_ROUTER}.middlewares"]
    assert labels[f"traefik.http.middlewares.{middleware}.stripprefix.prefixes"] == "/tankpit-sim"


def test_traefik_routes_on_the_external_network_the_service_joins() -> None:
    """platform-network is the root compose's, joined, not made."""
    network = _labels()["traefik.docker.network"]
    assert [narrow_json_to_str(name) for name in require_list(_service(), "networks")] == [network]
    compose = narrow_json_to_dict(load_json_str(_COMPOSE.read_text(encoding="utf-8")))
    assert require_dict(require_dict(compose, "networks"), network) == {"external": True}


def test_the_rooms_it_names_open() -> None:
    """Each --room is a room the server can build on a shipped field."""
    host = build_host(_args(), account_book())
    assert [(room.info["room_id"], room.info["image"]) for room in host.rooms] == [
        ("1", "field01.gif"),
        ("5", "field05.gif"),
    ]
