"""Per-kind verification for a single wiki physics claim.

One checker per claim kind -- value, bytes, members, keys, probes, and
the weak ``law`` existence check -- plus the probe decoding they share.
The scanner that finds claims and dispatches to these is
:mod:`scripts.physics_claims`.
"""

from __future__ import annotations

from enum import IntEnum
from types import CodeType, ModuleType
from typing import Protocol

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
)

#: Element type of every container this rule can verify. The bound
#: modules hold enum members, ints and strings; ``object`` is banned in
#: annotations by the ``typing`` guard, so the union is spelled out.
_MemberItem = IntEnum | int | str

#: Container shapes a ``members`` claim can bind. The read site
#: annotates ``getattr`` with this and then re-checks by ``isinstance``,
#: the same narrow-then-verify idiom :func:`_check_value_claim` uses for
#: ``constant: int``.
_MemberSymbol = (
    type[IntEnum]
    | dict[_MemberItem, _MemberItem]
    | set[_MemberItem]
    | frozenset[_MemberItem]
    | tuple[_MemberItem, ...]
)


class _AnnotatedRecord(Protocol):
    """A symbol whose annotated fields a ``keys`` claim states.

    ``TypedDict`` classes carry their field names in ``__annotations__``.
    The read site still uses ``getattr`` with a default, because a claim
    may name a symbol that has no annotations at all — that is a
    reported violation, not a crash.
    """

    __annotations__: dict[str, str]


class _ProbeFn(Protocol):
    """Shape of a physics formula bound by a probe claim.

    The ``__code__`` attribute is part of the contract: probe claims
    bind plain module-level functions, whose arity the checker reads
    from ``__code__.co_argcount`` to verify probe args fit before
    calling.
    """

    __code__: CodeType

    def __call__(self, *args: int) -> int:
        """Evaluate the formula at one probe point.

        Args:
            *args: Integer probe arguments.

        Returns:
            The formula's integer value at the probe point.
        """
        ...


def _check_value_claim(
    claim: JSONObject,
    module: ModuleType,
    symbol_name: str,
    prefix: str,
) -> list[str]:
    """Verify an int-constant claim against the resolved symbol.

    Args:
        claim: Claim object carrying ``value``.
        module: Imported physics module.
        symbol_name: Symbol the claim binds.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when the constant matches).
    """
    expected = claim.get("value")
    if isinstance(expected, bool) or not isinstance(expected, int):
        return [f"{prefix}: 'value' must be an int"]
    constant: int = getattr(module, symbol_name)
    if isinstance(constant, bool) or not isinstance(constant, int):
        return [f"{prefix}: symbol is not an int constant"]
    if constant != expected:
        return [f"{prefix}: claim says {expected}, code has {constant}"]
    return []


def _check_bytes_claim(
    claim: JSONObject,
    module: ModuleType,
    symbol_name: str,
    prefix: str,
) -> list[str]:
    """Verify a bytes-constant claim against the resolved symbol.

    The wire vocabulary includes literal byte payloads that are not
    numbers — the plain-text commands ``PLAIN_QUIT`` (``b"-"``),
    ``PLAIN_AUTOSCROLL_ON`` (``b"A1"``) and friends. JSON has no bytes
    literal, so the claim carries a string and the comparison encodes
    it latin-1: every byte value 0-255 round-trips, and the common
    all-ASCII case stays readable in the wiki block.

    Args:
        claim: Claim object carrying ``bytes``.
        module: Imported module.
        symbol_name: Symbol the claim binds.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when the constant matches).
    """
    expected_text = claim.get("bytes")
    if not isinstance(expected_text, str):
        return [f"{prefix}: 'bytes' must be a latin-1 string"]
    constant: bytes = getattr(module, symbol_name)
    if not isinstance(constant, bytes):
        return [f"{prefix}: symbol is not a bytes constant"]
    expected = expected_text.encode("latin-1")
    if constant != expected:
        return [f"{prefix}: claim says {expected!r}, code has {constant!r}"]
    return []


def _unwrap(item: _MemberItem) -> JSONValue:
    """Reduce one container element to its JSON-comparable value.

    Args:
        item: A container element, possibly an ``IntEnum`` member.

    Returns:
        The element's int value when it is an enum member, else the
        element itself.
    """
    if isinstance(item, IntEnum):
        return item.value
    return item


def _normalize_members(value: _MemberSymbol) -> tuple[JSONValue | None, bool]:
    """Project a container symbol into a JSON-comparable shape.

    Four container shapes carry game facts the wiki states literally,
    and all four were previously only expressible as prose ``law``
    claims — which verify existence and nothing else:

    * ``IntEnum`` subclass -> ``{member name: int value}``
    * ``dict`` -> ``{key: value}``, keys taken from the enum member
      name when the key is an enum, else ``str(key)``
    * ``tuple`` / ``list`` -> array, ORDER-SENSITIVE: ``RANK_NAMES``
      is indexed by rank, so its order IS the fact being claimed
    * ``set`` / ``frozenset`` -> array, order-INSENSITIVE, since a
      set has no order to state

    Args:
        value: The resolved symbol.

    Returns:
        Pair of (projection, ordered). The projection is None when the
        symbol is not a supported container. ``ordered`` is False only
        for sets, whose claim may list members in any order.
    """
    # An IntEnum CLASS is detected by its ``__members__`` mapping rather
    # than ``isinstance(value, type)``: the bare ``type`` builtin reads as
    # ``type[type]`` under ``disallow_any_expr`` and fails mypy. The
    # mapping is also exactly the projection wanted here.
    enum_members: dict[str, IntEnum] | None = getattr(value, "__members__", None)
    if enum_members is not None:
        return {name: member.value for name, member in enum_members.items()}, True
    if isinstance(value, dict):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            name = key.name if isinstance(key, IntEnum) else str(key)
            out[name] = _unwrap(item)
        return out, True
    if isinstance(value, (set, frozenset)):
        return [_unwrap(item) for item in value], False
    if isinstance(value, tuple):
        return [_unwrap(item) for item in value], True
    return None, True


def _check_members_claim(
    claim: JSONObject,
    module: ModuleType,
    symbol_name: str,
    prefix: str,
) -> list[str]:
    """Verify a container claim against the symbol's full contents.

    Equality is TOTAL, not subset: a member the wiki omits fails as
    loudly as one it invents. That is deliberate — a partially stated
    table reads as complete to the next person, which is the failure
    this whole rule exists to prevent.

    Args:
        claim: Claim object carrying ``members``.
        module: Imported module.
        symbol_name: Symbol the claim binds.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when the contents match exactly).
    """
    expected = claim.get("members")
    if not isinstance(expected, (dict, list)):
        return [f"{prefix}: 'members' must be a JSON object or array"]
    symbol: _MemberSymbol = getattr(module, symbol_name)
    actual, ordered = _normalize_members(symbol)
    if actual is None:
        return [f"{prefix}: symbol is not an enum, mapping, sequence or set"]
    if not ordered and isinstance(actual, list) and isinstance(expected, list):
        actual = sorted(actual, key=repr)
        expected = sorted(expected, key=repr)
    if actual != expected:
        return [f"{prefix}: claim members {expected!r} != code {actual!r}"]
    return []


def _check_keys_claim(
    claim: JSONObject,
    module: ModuleType,
    symbol_name: str,
    prefix: str,
) -> list[str]:
    """Verify a record type's field names against the wiki.

    ``TypedDict`` records are the shape of most wire-adjacent
    bookkeeping — a fuel book, a damage book, an outcome record. Their
    FIELD SET is a real fact the wiki states, and without this kind the
    only available claim was ``law``: prose plus an existence check,
    which goes on passing when a field is added, renamed or dropped.

    The claim lists the field names; comparison is sorted and total,
    for the same reason :func:`_check_members_claim` is total.

    Args:
        claim: Claim object carrying ``keys``.
        module: Imported module.
        symbol_name: Symbol the claim binds.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when the field set matches exactly).
    """
    expected = claim.get("keys")
    if not isinstance(expected, list):
        return [f"{prefix}: 'keys' must be a JSON array of field names"]
    # Two steps, each with its own annotation: nesting the getattrs makes
    # the inner call an unannotated Any expression, which
    # ``disallow_any_expr`` rejects.
    record: _AnnotatedRecord = getattr(module, symbol_name)
    annotations: dict[str, str] = getattr(record, "__annotations__", {})
    if not annotations:
        return [f"{prefix}: symbol has no annotated fields to claim"]
    actual = sorted(annotations)
    wanted = sorted(str(name) for name in expected)
    if actual != wanted:
        return [f"{prefix}: claim keys {wanted!r} != code {actual!r}"]
    return []


def _decode_probe(probe: JSONObject, prefix: str) -> tuple[list[int], int, list[str]]:
    """Decode one probe entry into args and expected value.

    Args:
        probe: Probe object with ``args`` and ``expect``.
        prefix: ``page#id`` for violation messages.

    Returns:
        Tuple of (args, expected value, violations). Args and expected
        are empty/zero when violations are non-empty.
    """
    raw_args = probe.get("args")
    if not isinstance(raw_args, list):
        return [], 0, [f"{prefix}: probe 'args' must be a list of ints"]
    args: list[int] = []
    for arg in raw_args:
        if isinstance(arg, bool) or not isinstance(arg, int):
            return [], 0, [f"{prefix}: probe 'args' must be a list of ints"]
        args.append(arg)
    expected = probe.get("expect")
    if isinstance(expected, bool) or not isinstance(expected, int):
        return [], 0, [f"{prefix}: probe 'expect' must be an int"]
    return args, expected, []


def _run_probe(fn: _ProbeFn, args: list[int], expected: int, prefix: str) -> list[str]:
    """Evaluate one probe point against the bound formula.

    Args:
        fn: The bound physics formula.
        args: Integer probe arguments.
        expected: Expected formula value.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when the probe matches).
    """
    if fn.__code__.co_argcount != len(args):
        return [f"{prefix}: probe args {args} do not fit the signature"]
    result: int = fn(*args)
    if isinstance(result, bool) or not isinstance(result, int):
        return [f"{prefix}: probe {args} returned a non-int"]
    if result != expected:
        return [f"{prefix}: probe {args} expected {expected}, got {result}"]
    return []


def _check_probe_claim(
    claim: JSONObject,
    module: ModuleType,
    symbol_name: str,
    prefix: str,
) -> list[str]:
    """Verify a formula claim by evaluating its probe grid.

    Args:
        claim: Claim object carrying ``formula`` and ``probes``.
        module: Imported physics module.
        symbol_name: Symbol the claim binds.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when every probe matches).
    """
    if not isinstance(claim.get("formula"), str):
        return [f"{prefix}: 'formula' must be a string"]
    raw_probes = claim.get("probes")
    if not isinstance(raw_probes, list):
        return [f"{prefix}: 'probes' must be a list"]
    if not raw_probes:
        return [f"{prefix}: probe claim has an empty probe grid"]
    fn: _ProbeFn = getattr(module, symbol_name)
    if not callable(fn):
        return [f"{prefix}: symbol is not callable but claim has probes"]
    violations: list[str] = []
    for raw_probe in raw_probes:
        if not isinstance(raw_probe, dict):
            violations.append(f"{prefix}: probe entries must be JSON objects")
            continue
        args, expected, probe_violations = _decode_probe(raw_probe, prefix)
        if probe_violations:
            violations.extend(probe_violations)
            continue
        violations.extend(_run_probe(fn, args, expected, prefix))
    return violations


def _check_law_claim(claim: JSONObject, prefix: str) -> list[str]:
    """Verify a prose-law claim's shape.

    ``law`` claims bind physics symbols that CANNOT be verified on an
    int probe grid — predicates and geometry functions whose inputs
    include protocol objects (e.g. ``line_of_sight.is_shot_line_clear``
    takes a terrain view) or whose outputs are not scalars
    (``shot_line_tiles`` returns the Bresenham raster). The
    computational guarantee degrades, deliberately and visibly, to:
    the symbol EXISTS (checked by the caller) and the wiki states the
    game law it implements in prose. Scalar constants and int
    formulas must keep using ``value``/``probes`` — a law claim on a
    probe-able symbol is reviewer-rejected drift.

    Args:
        claim: Claim object carrying ``law``.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations (empty when the law text is a non-empty string).
    """
    law = claim.get("law")
    if not isinstance(law, str) or not law.strip():
        return [f"{prefix}: 'law' must be a non-empty prose string"]
    return []


def _run_kind_check(
    kind: str,
    claim: JSONObject,
    module: ModuleType,
    symbol_name: str,
    prefix: str,
) -> list[str]:
    """Dispatch one claim to the checker for its kind.

    Split out of :func:`_check_claim` to keep that function under the
    complexity bar as kinds are added.

    Args:
        kind: The single kind present on the claim.
        claim: Claim object.
        module: Imported module the claim binds into.
        symbol_name: Symbol the claim binds.
        prefix: ``page#id`` for violation messages.

    Returns:
        Violations from the kind's checker.
    """
    if kind == "value":
        return _check_value_claim(claim, module, symbol_name, prefix)
    if kind == "bytes":
        return _check_bytes_claim(claim, module, symbol_name, prefix)
    if kind == "members":
        return _check_members_claim(claim, module, symbol_name, prefix)
    if kind == "keys":
        return _check_keys_claim(claim, module, symbol_name, prefix)
    if kind == "probes":
        return _check_probe_claim(claim, module, symbol_name, prefix)
    return _check_law_claim(claim, prefix)


__all__ = []
