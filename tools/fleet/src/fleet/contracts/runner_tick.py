"""What a node runner records about one tick (MCPs board task 939ec5c7, A4).

Each runner probes its node and decides whether to claim every three
minutes, and until now the answer lived only in that runner's daily log on
the hub: on 2026-10-02 thirteen jobs queued behind four running and finding
out why each node took nothing meant reading seven logs. The runner now
sends this record through MCPs ``dispatch_tick`` at the end of every tick,
which keeps one row per runner (MCPs mig 640), and ``fleet_status`` renders
them node by node: the tags its probe detected, the runs and workers it
holds, its free memory, the projects it could launch and why it is or is not
claiming.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict

from fleet.contracts.tags import NodeTag, decode_required_tags, encode_tags


class TickLoad(TypedDict):
    """What a runner's node holds and has free this tick.

    Attributes:
        runs: The fleet runs live on it.
        workers: The workers those runs were granted.
        free_ram_gb: The memory its capacity probe read free.
    """

    runs: int
    workers: int
    free_ram_gb: float


class RunnerTick(TypedDict):
    """One tick of one node runner.

    Attributes:
        node: The node it serves.
        elevated: Whether it is the node's elevated runner.
        tags: The tags its probe detected this tick, in vocabulary order;
            empty when the tick ended before its toolchain probe.
        fits: The projects its node could launch now; empty when it fits
            none or the tick ended earlier.
        load: What the node holds, or None when it did not answer its probe.
        claiming: Whether this tick asked the queue for a job.
        verdict: The line saying why it is or is not claiming.
    """

    node: str
    elevated: bool
    tags: tuple[NodeTag, ...]
    fits: tuple[str, ...]
    load: TickLoad | None
    claiming: bool
    verdict: str


def encode_runner_tick(tick: RunnerTick) -> JSONObject:
    """Encode a tick as ``dispatch_tick``'s arguments, identity aside.

    Args:
        tick: The tick.

    Returns:
        The arguments; ``load`` is omitted when the node did not answer,
        since the tool's schema is strict and reads an absent load as none.
    """
    encoded: JSONObject = {
        "node": tick["node"],
        "elevated": tick["elevated"],
        "tags": encode_tags(tick["tags"]),
        "fits": list(tick["fits"]),
        "claiming": tick["claiming"],
        "verdict": tick["verdict"],
    }
    load = tick["load"]
    if load is not None:
        encoded["load"] = {
            "runs": load["runs"],
            "workers": load["workers"],
            "freeRamGb": load["free_ram_gb"],
        }
    return encoded


def decode_runner_tick(value: JSONValue) -> RunnerTick:
    """Decode a tick from the shape :func:`encode_runner_tick` writes.

    Args:
        value: The encoded tick.

    Returns:
        The tick.

    Raises:
        JSONTypeError: If it or its load is not an object, a field is missing
            or mistyped, or its tags repeat or leave the vocabulary.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"runner tick must be a JSON object, got {type(value).__name__}")
    raw_load = value.get("load")
    load: TickLoad | None = None
    if raw_load is not None:
        if not isinstance(raw_load, dict):
            raise JSONTypeError(f"load must be a JSON object, got {type(raw_load).__name__}")
        load = TickLoad(
            runs=require_int(raw_load, "runs"),
            workers=require_int(raw_load, "workers"),
            free_ram_gb=require_float(raw_load, "freeRamGb"),
        )
    return RunnerTick(
        node=require_str(value, "node"),
        elevated=require_bool(value, "elevated"),
        tags=decode_required_tags(value.get("tags"), field="tags"),
        fits=tuple(require_str_list(value, "fits")),
        load=load,
        claiming=require_bool(value, "claiming"),
        verdict=require_str(value, "verdict"),
    )


def decode_recorded_at(answer: str) -> str:
    """Read when ``dispatch_tick`` stored a tick, from its answer.

    Args:
        answer: The tool's text: JSON whose ``tick`` object carries the
            stored row, ``tickedAt`` among it.

    Returns:
        The stored row's ``tickedAt``, the instant ``fleet_status`` ages it
        from.

    Raises:
        InvalidJsonError: If the answer is not JSON.
        JSONTypeError: If it is not an object holding a ``tick`` object with
            a string ``tickedAt``.
    """
    body = load_json_str(answer)
    if not isinstance(body, dict):
        raise JSONTypeError(f"dispatch_tick must answer an object, got {type(body).__name__}")
    return require_str(require_dict(body, "tick"), "tickedAt")


__all__ = [
    "RunnerTick",
    "TickLoad",
    "decode_recorded_at",
    "decode_runner_tick",
    "encode_runner_tick",
]
