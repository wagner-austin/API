"""The sim server's compose file agrees with the server it starts.

``sim-server.compose.json`` runs ``tankpit-sim-serve`` on sedona beside its
own Postgres (MCPs board task ff035b0d), staged and started by
``scripts.sim_host``. It is JSON, which compose reads as it reads YAML, so
this package's own JSON decoders hold it to the CLI: its command must parse
as ``tankpit-sim-serve``'s flags, the port it publishes must be the port the
server binds and only on loopback, the database the server reads must be the
``sim-db`` service beside it, and neither service may start on its own.
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


def _compose() -> JSONObject:
    """The whole compose file."""
    return narrow_json_to_dict(load_json_str(_COMPOSE.read_text(encoding="utf-8")))


def _service(name: str) -> JSONObject:
    """One service of the compose file."""
    return require_dict(require_dict(_compose(), "services"), name)


def _args() -> ServeArgs:
    """The sim's command, read as the server reads it."""
    command = [narrow_json_to_str(word) for word in require_list(_service("sim"), "command")]
    assert command[0] == "tankpit-sim-serve"
    return parse_serve_args(command[1:])


def test_the_command_serves_from_the_database_beside_it() -> None:
    """The variable the server reads is the one the service sets, on the sim-db service."""
    args = _args()
    assert args.accounts == AccountSource("database", "TANKPIT_SIM_DATABASE_URL")
    dsn = require_str(require_dict(_service("sim"), "environment"), "TANKPIT_SIM_DATABASE_URL")
    assert dsn.startswith("postgresql://tankpit_sim:")
    assert dsn.endswith("@sim-db:5432/tankpit_sim")
    database = require_dict(_service("sim-db"), "environment")
    assert (require_str(database, "POSTGRES_USER"), require_str(database, "POSTGRES_DB")) == (
        "tankpit_sim",
        "tankpit_sim",
    )
    assert (args.bind, args.ticks) == ("0.0.0.0", None)


def test_it_publishes_the_port_the_server_binds_on_loopback_only() -> None:
    """The one published port is the --port flag, on sedona's 127.0.0.1 alone."""
    port = str(_args().port)
    assert [narrow_json_to_str(p) for p in require_list(_service("sim"), "ports")] == [
        f"127.0.0.1:{port}:{port}"
    ]
    assert "ports" not in _service("sim-db")


def test_neither_service_starts_on_its_own() -> None:
    """The operator starts it by hand: no restart policy brings either back.

    The operator, 2026-10-13: "make sure it doesnt auro stsrt or anything i
    csn maniallt stsrt it wheb needed".
    """
    assert require_str(_service("sim"), "restart") == "no"
    assert require_str(_service("sim-db"), "restart") == "no"


def test_the_accounts_live_in_a_named_volume() -> None:
    """The database's directory is a named volume, which outlives every stop."""
    volumes = [narrow_json_to_str(v) for v in require_list(_service("sim-db"), "volumes")]
    assert volumes == ["sim-db:/var/lib/postgresql/data"]
    assert require_dict(require_dict(_compose(), "volumes"), "sim-db") == {}


def test_the_sim_waits_for_a_healthy_database() -> None:
    """The sim starts only once pg_isready answers for its database."""
    depends = require_dict(require_dict(_service("sim"), "depends_on"), "sim-db")
    assert require_str(depends, "condition") == "service_healthy"
    health = [
        narrow_json_to_str(word)
        for word in require_list(require_dict(_service("sim-db"), "healthcheck"), "test")
    ]
    assert health == ["CMD-SHELL", "pg_isready -U tankpit_sim -d tankpit_sim"]


def test_the_web_root_is_where_the_image_puts_the_client() -> None:
    """--web-root names the directory the Dockerfile copies play.html and dist/ into."""
    web_root = _args().web_root
    dockerfile = (_COMPOSE.parent / "Dockerfile").read_text(encoding="utf-8")
    assert f"COPY ${{APP_DIR}}/web/play.html {web_root}/play.html" in dockerfile
    assert f"COPY --from=web /web/dist {web_root}/dist" in dockerfile


def test_the_rooms_it_names_open() -> None:
    """Each --room is a room the server can build on a shipped field."""
    host = build_host(_args(), account_book())
    assert [(room.info["room_id"], room.info["image"]) for room in host.rooms] == [
        ("1", "field01.gif"),
        ("5", "field05.gif"),
    ]
