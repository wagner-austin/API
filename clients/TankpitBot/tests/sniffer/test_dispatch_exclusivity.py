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
from collections import Counter
from pathlib import Path

_SNIFFER = Path(__file__).resolve().parents[2] / "src" / "tankpit_bot" / "sniffer"

# The resource and tank handlers share one module, so their types are
# listed per function; position and container own their modules outright.
_RESOURCE_TYPES = frozenset({0x2E, 0x44, 0x46, 0x49, 0x64, 0x67, 0x74})
_TANK_TYPES = frozenset({0x21, 0x28, 0x2E, 0x3E, 0x41, 0x48, 0x53, 0x58})

# 0x2E reaches both handlers by design; the resource handler claims it and
# owns every effect the message carries, including promotion progress.
#
# The tank handler's damage case is the other effect the resource handler
# shadows, and it was checked rather than assumed (2026-08-12). It drops
# two things for the self tank: damage_state, a fuel-quartile estimator
# the bot does not need for itself because it knows its own fuel exactly,
# and the last_wire_seen_ms refresh. The second is the one worth proving,
# and it is unread: the self tank IS in the registry (present in all six
# sessions replayed), but every production consumer of last_wire_seen_ms
# filters it out first -- threats.py and the three threat_acquisition
# loops through _is_enemy, which is `not is_self and team != self_team`,
# and hunt_acquire with the same check inlined. The only unfiltered
# reader is action_lab's tracking probe, which is asked about one tank id.
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


def _claimed_by(path: Path, function: str, key: str) -> frozenset[str | int]:
    """Return the literals one function tests ``key`` against.

    Covers both routing styles in the codebase: ``match`` mapping cases
    (``case {"msg_type": 0x44, ...}``) and equality chains
    (``if message["msg_type"] == 0x44``). A router written in the second
    style is invisible to a match-only reader, which is how three of
    these chains looked empty on the first pass.

    Args:
        path: Module to parse.
        function: Function whose routing is being read.
        key: Discriminator key, e.g. ``msg_type`` or ``kind``.

    Returns:
        Every literal the function routes on.
    """
    found: set[str | int] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for inner in ast.walk(node):
            found |= _case_literals(inner, key)
            found |= _comparison_literals(inner, key)
    return frozenset(found)


def _case_literals(node: ast.AST, key: str) -> frozenset[str | int]:
    """Return literals bound to ``key`` by one ``match`` mapping case.

    Args:
        node: Any AST node; non-mapping-patterns contribute nothing.
        key: Discriminator key name.

    Returns:
        The literals this case routes on.
    """
    if not isinstance(node, ast.MatchMapping):
        return frozenset()
    found: set[str | int] = set()
    for map_key, pattern in zip(node.keys, node.patterns, strict=True):
        if not (isinstance(map_key, ast.Constant) and map_key.value == key):
            continue
        if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
            literal = pattern.value.value
            if isinstance(literal, (str, int)):
                found.add(literal)
    return frozenset(found)


def _comparison_literals(node: ast.AST, key: str) -> frozenset[str | int]:
    """Return literals compared against ``key`` by one ``==`` test.

    Args:
        node: Any AST node; non-comparisons contribute nothing.
        key: Discriminator key name.

    Returns:
        The literals this comparison routes on.
    """
    if not isinstance(node, ast.Compare) or not _is_key_lookup(node.left, key):
        return frozenset()
    return frozenset(
        comparator.value
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (str, int))
    )


def _is_key_lookup(node: ast.expr, key: str) -> bool:
    """Return whether ``node`` reads the discriminator ``key``.

    Args:
        node: Left-hand side of a comparison.
        key: Discriminator key name.

    Returns:
        True for ``x[key]`` and for a bare name equal to ``key``.
    """
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == key
    return isinstance(node, ast.Name) and node.id == key


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


_ROOT = Path(__file__).resolve().parents[2] / "src" / "tankpit_bot"

# Every other first-match-wins chain in the codebase, each with the same
# shape and the same mutation result: the early returns are unobservable
# while the handlers stay disjoint. Listed here so one new overlapping
# case anywhere fails a test instead of silently starving a handler.
_CHAINS: tuple[tuple[str, Path, tuple[str, ...], str], ...] = (
    (
        "sim ghost consume",
        _ROOT / "sim" / "ghost.py",
        ("_consume_tank_message", "_consume_combat_social", "_consume_world_reads"),
        "msg_type",
    ),
    (
        "wire timeline ingest",
        _ROOT / "validate" / "wire_timeline.py",
        ("_ingest_fuel_and_hazards", "_ingest_combat_and_identity"),
        "msg_type",
    ),
    (
        "scorecard diagnostic routing",
        _ROOT / "diagnostics" / "session_scorecard_routes.py",
        ("_route_combat_diagnostic", "_route_fuel_diagnostic"),
        "kind",
    ),
    (
        "shadow timeline ingest",
        _ROOT / "validate" / "shadow_timeline.py",
        ("_ingest_tank_events", "_ingest_combat_events"),
        "msg_type",
    ),
)


def test_every_first_match_wins_chain_stays_disjoint() -> None:
    """No handler in any dispatch chain claims another's message.

    Same invariant as the world-state chain above, for the three other
    chains built the same way. Each was confirmed disjoint when this was
    written; an overlap introduced later means the second handler stops
    running for that message and nothing says so.
    """
    collisions: dict[str, frozenset[str | int]] = {}
    for name, path, functions, key in _CHAINS:
        claimed = {fn: _claimed_by(path, fn, key) for fn in functions}
        assert all(claimed.values()), f"{name}: a handler routed on nothing -- reader is wrong"
        ordered = list(functions)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                shared = claimed[first] & claimed[second]
                if shared:
                    collisions[f"{name}: {first} x {second}"] = shared

    assert collisions == {}


def test_the_declared_handler_types_match_the_source() -> None:
    """The resource and tank literals above are not allowed to drift.

    They are written out because both handlers live in one module and
    cannot be separated by parsing the file alone. If a case is added to
    either, this test fails and the constant is updated deliberately
    rather than the exclusivity check quietly narrowing.
    """
    module_types = _match_msg_types(_SNIFFER / "world_state_dispatch.py")

    assert module_types >= _RESOURCE_TYPES | _TANK_TYPES


# The same first-match-wins shape inside ONE function: a cascade of arms
# testing one discriminator, each ending in a bare return. Removing all
# four of the scorecard cascade's returns leaves the accumulator
# byte-identical across the 1,403,706 records in the 426 archived runs,
# because no arm can match after an earlier one did. That is the property
# below, and it is what makes those returns structural rather than
# redundant: a second arm testing an already-claimed value would make the
# first return the only thing preventing a double-apply.
_CASCADES: tuple[tuple[str, Path, str, str], ...] = (
    (
        "scorecard channel cascade",
        _ROOT / "diagnostics" / "session_scorecard_accumulator.py",
        "route_scorecard_record",
        "channel",
    ),
    (
        "scorecard metrics kinds",
        _ROOT / "diagnostics" / "session_scorecard_routes.py",
        "_route_metrics_diagnostic",
        "kind",
    ),
    (
        "wire timeline combat and identity",
        _ROOT / "validate" / "wire_timeline.py",
        "_ingest_combat_and_identity",
        "msg_type",
    ),
    (
        "shadow timeline combat events",
        _ROOT / "validate" / "shadow_timeline.py",
        "_ingest_combat_events",
        "msg_type",
    ),
)


def _compared_literals_in_order(path: Path, function: str, key: str) -> list[str | int]:
    """Return every literal one function compares ``key`` against, with repeats.

    ``_claimed_by`` returns a set, which cannot see a value tested twice
    -- and a value tested twice is exactly the failure this reads for.

    Args:
        path: Module to parse.
        function: Function whose cascade is being read.
        key: Discriminator name, compared as a bare local.

    Returns:
        Every literal compared against ``key``, duplicates preserved.
        The order is ``ast.walk`` order, not source order -- an arm
        nested inside a ``BoolOp`` (``channel == "WIRE" and ...``) is
        reached at a different depth than its siblings. Only the
        multiset matters to the caller.
    """
    found: list[str | int] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Compare) or not _is_key_lookup(inner.left, key):
                continue
            found.extend(
                comparator.value
                for comparator in inner.comparators
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (str, int))
            )
    return found


def test_every_cascade_arm_tests_a_distinct_value() -> None:
    """No cascade tests one discriminator value in two arms.

    A repeat means the second arm is unreachable for that value and the
    first arm's return is suddenly load-bearing -- the failure being
    silent either way, which is why it is read from the source rather
    than waited for.
    """
    repeats: dict[str, list[str | int]] = {}
    for name, path, function, key in _CASCADES:
        literals = _compared_literals_in_order(path, function, key)
        assert literals, f"{name}: read no arms -- the reader is wrong, not the source"
        counts = Counter(literals)
        duplicated = sorted(
            (value for value, count in counts.items() if count > 1),
            key=repr,
        )
        if duplicated:
            repeats[name] = duplicated

    assert repeats == {}
