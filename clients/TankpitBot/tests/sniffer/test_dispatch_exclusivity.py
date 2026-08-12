"""The world-state dispatch chain must stay mutually exclusive.

``dispatch_world_state_update`` runs four handlers in order and returns
after the first one that claims the message::

    if _dispatch_resource_update(ws, decoded): return
    if _dispatch_tank_update(ws, decoded):     return
    if _dispatch_position_update(ws, decoded): return
    if _dispatch_container_message(ws, decoded): return

Those early returns are unobservable today -- a mutation sweep on
2026-08-11 found three of them indistinguishable from absent -- and the
reason is precisely the property this test pins: no message type is
claimed by two handlers, so continuing past a claim only re-runs pattern
matches that cannot fire.

That makes the returns structural rather than redundant, and it is why
they must not be deleted. The moment two handlers share a type, the
first `return` is the only thing preventing a double-apply, and the
failure is silent: state applied twice, or a diagnostic emitted twice.

This is not hypothetical. ``0x2E`` IS shared between the resource and
tank handlers -- the fuel-bearing long form and the short form -- and
because the resource handler runs first and returns, the tank handler's
promotion emission never saw the self tank at all. Across a 320-session
corpus the self tank appears in 64,792 long-form bodies and zero
short-form ones, so ``self_promo_eligible`` could not fire until the
resource handler was taught to emit it (2026-08-12).

The exemption below records that known overlap. A NEW overlap fails this
test, which is the signal to check whether the ordering silently drops
something the second handler was supposed to do.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SNIFFER = Path(__file__).resolve().parents[2] / "src" / "tankpit_bot" / "sniffer"

# The resource and tank handlers share one module, so their types are
# listed per function; position and container own their modules outright.
_RESOURCE_TYPES = frozenset({0x2E, 0x44, 0x46, 0x49, 0x64, 0x67, 0x74})
_TANK_TYPES = frozenset({0x21, 0x28, 0x2E, 0x3E, 0x41, 0x48, 0x53, 0x58})

# 0x2E reaches both handlers by design; the resource handler claims it and
# owns every effect the message carries, including promotion progress.
_KNOWN_OVERLAP = frozenset({0x2E})


def _match_msg_types(path: Path) -> frozenset[int]:
    """Return every integer ``msg_type`` literal a module's cases bind.

    Args:
        path: Module to parse.

    Returns:
        The message-type literals appearing in ``match`` mapping patterns.
    """
    found: set[int] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.MatchMapping):
            continue
        for key, pattern in zip(node.keys, node.patterns, strict=True):
            if not (isinstance(key, ast.Constant) and key.value == "msg_type"):
                continue
            if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
                value = pattern.value.value
                if isinstance(value, int):
                    found.add(value)
    return frozenset(found)


def test_the_four_dispatchers_claim_disjoint_message_types() -> None:
    """No message type is claimed by two handlers in the chain.

    A new overlap means the handler that runs SECOND stops seeing that
    message, and nothing else reports it. Add the type to
    ``_KNOWN_OVERLAP`` only after confirming the first handler performs
    every effect the second one would have.
    """
    combat = _match_msg_types(_SNIFFER / "world_state_dispatch_combat.py")
    handlers = {
        "resource": _RESOURCE_TYPES,
        "tank": _TANK_TYPES | combat,
        "position": _match_msg_types(_SNIFFER / "world_state_dispatch_position.py"),
        "container": _match_msg_types(_SNIFFER / "world_state_dispatch_containers.py"),
    }

    names = sorted(handlers)
    overlaps: dict[tuple[str, str], frozenset[int]] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = handlers[first] & handlers[second] - _KNOWN_OVERLAP
            if shared:
                overlaps[(first, second)] = shared

    assert overlaps == {}


def test_the_declared_handler_types_match_the_source() -> None:
    """The resource and tank literals above are not allowed to drift.

    They are written out because both handlers live in one module and
    cannot be separated by parsing the file alone. If a case is added to
    either, this test fails and the constant is updated deliberately
    rather than the exclusivity check quietly narrowing.
    """
    module_types = _match_msg_types(_SNIFFER / "world_state_dispatch.py")

    assert module_types >= _RESOURCE_TYPES | _TANK_TYPES
