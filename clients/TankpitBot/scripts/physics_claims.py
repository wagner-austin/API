"""Guard rule: wiki claims must match the code they bind.

The wiki is the source of truth for the game's measured laws and the
protocol vocabulary; the code acts on them. This rule machine-checks
the binding in both directions on every ``make check`` (see
``wiki/pages/physics-module-roadmap.md`` Phase 1):

* Forward: every ``json claims`` fenced block in ``wiki/pages/*.md``
  names a symbol in one of :const:`CLAIM_TARGETS` and its expected
  value (an int or bytes constant) or probe grid (a formula checked
  at explicit points). The rule imports the symbol and verifies
  computationally.
* Reverse: every public symbol (``__all__``) of every bound target
  must be claimed by exactly one claim — a fact the wiki does not
  state is a violation too.

**Targets are packages OR single modules.** Reverse coverage is
all-or-nothing per target: binding a target obliges the wiki to claim
every one of its public symbols. That is the property worth having —
it is what makes "the wiki does not mention this" a build failure
rather than an omission nobody notices — but it means a target is
onboarded in one deliberate pass, not incrementally. Module-level
targets exist so a large package can be onboarded a module at a time
instead of in a 221-symbol big bang.

The module name is historical: this rule shipped 2026-07-21 binding
only :mod:`tankpit_bot.physics`, and the live references to
``physics_claims`` across the wiki, README and SCHEMA have not yet
been renamed. It binds :const:`CLAIM_TARGETS` now, not physics alone.

Claim blocks are validated by manual narrowing over ``JSONValue``
(no schema framework); every malformed shape produces a specific,
traceable violation message instead of an exception. A tree without
a ``wiki/pages`` directory is out of scope (mirrors
``contract_rules`` skipping absent packages), so guard runs against
synthetic test trees stay green.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from enum import IntEnum
from pathlib import Path
from types import CodeType, ModuleType
from typing import Protocol

from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONValue,
    load_json_str,
)
from platform_core.logging import get_logger

PHYSICS_PACKAGE = "tankpit_bot.physics"
COMMANDS_MODULE = "tankpit_bot.protocol.commands"
PROTOCOL_CONSTANTS_MODULE = "tankpit_bot.protocol.constants"

#: Every target the wiki must bind, in onboarding order. A package
#: binds every public symbol of every submodule; a bare module binds
#: only its own ``__all__``. Adding a target here is a commitment:
#: reverse coverage immediately requires a claim for each of its
#: public symbols, so the claims land in the same commit.
LEDGER_PACKAGE = "tankpit_bot.ledger"

CLAIM_TARGETS: tuple[str, ...] = (
    PHYSICS_PACKAGE,
    COMMANDS_MODULE,
    PROTOCOL_CONSTANTS_MODULE,
    LEDGER_PACKAGE,
)

CLAIM_FENCE_OPEN = "```json claims"
CLAIM_FENCE_CLOSE = "```"
#: Claim kinds. Exactly one must appear on every claim. ``law`` is the
#: weak one — existence plus prose — and exists only for symbols no
#: other kind can verify; prefer any of the others when they fit.
CLAIM_KINDS: tuple[str, ...] = ("value", "bytes", "members", "keys", "probes", "law")

_LOGGER = get_logger(__name__)


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


def _extract_claim_blocks(text: str, page: str) -> tuple[list[str], list[str]]:
    """Extract the raw text of every claim block in one wiki page.

    Args:
        text: Full markdown text of the page.
        page: Page name for violation messages.

    Returns:
        Pair of (raw JSON block texts, violations for unclosed fences).
    """
    blocks: list[str] = []
    violations: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped == CLAIM_FENCE_OPEN:
                current = []
        elif stripped == CLAIM_FENCE_CLOSE:
            blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)
    if current is not None:
        violations.append(f"{page}: unclosed '{CLAIM_FENCE_OPEN}' fence")
    return blocks, violations


def _parse_claim_block(raw: str, page: str) -> tuple[list[JSONObject], list[str]]:
    """Parse one claim block into claim objects.

    Args:
        raw: Raw JSON text of the block.
        page: Page name for violation messages.

    Returns:
        Pair of (claim objects, violations).
    """
    try:
        parsed = load_json_str(raw)
    except InvalidJsonError as error:
        _LOGGER.warning("physics_claim_block_unparseable page=%s error=%s", page, error)
        return [], [f"{page}: claim block is not valid JSON ({error})"]
    if not isinstance(parsed, dict):
        return [], [f"{page}: claim block must be a JSON object"]
    entries = parsed.get("claims")
    if not isinstance(entries, list):
        return [], [f"{page}: claim block lacks a 'claims' list"]
    claims: list[JSONObject] = []
    violations: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            claims.append(entry)
        else:
            violations.append(f"{page}: claim entries must be JSON objects")
    return claims, violations


def _import_claim_module(
    module_name: str,
    prefix: str,
) -> tuple[ModuleType | None, list[str]]:
    """Import the module a claim binds into.

    Args:
        module_name: Dotted module path from the claim's code address.
        prefix: ``page#id`` for violation messages.

    Returns:
        Pair of (module or None, violations).
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        _LOGGER.warning("physics_claim_module_unimportable module=%s error=%s", module_name, error)
        return None, [f"{prefix}: module '{module_name}' does not import"]
    return module, []


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


def _binds_into_target(module_name: str, target: str) -> bool:
    """Report whether a claim's module sits inside one bound target.

    A package target matches its submodules; a module target matches
    only itself. Without the exact-match arm a module target could
    never be satisfied, since ``"a.b.c".startswith("a.b.c.")`` is
    false.

    Args:
        module_name: Dotted module path from the claim's code address.
        target: One entry of :const:`CLAIM_TARGETS`.

    Returns:
        True when the claim binds into that target.
    """
    return module_name == target or module_name.startswith(f"{target}.")


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


def _check_claim(
    claim: JSONObject,
    page: str,
    targets: tuple[str, ...],
) -> tuple[str, str, list[str]]:
    """Fully verify one claim.

    Args:
        claim: Claim object.
        page: Page name for violation messages.
        targets: Bound targets; the claim must bind into one of them.

    Returns:
        Tuple of (claim id, bound code address, violations). The id and
        code are empty strings when the claim is too malformed to name.
    """
    claim_id = claim.get("id")
    code = claim.get("code")
    if not isinstance(claim_id, str) or not isinstance(code, str):
        return "", "", [f"{page}: claim needs string 'id' and 'code' fields"]
    prefix = f"{page}#{claim_id}"
    kinds_present = [kind for kind in CLAIM_KINDS if kind in claim]
    if len(kinds_present) != 1:
        joined_kinds = "/".join(CLAIM_KINDS)
        return claim_id, code, [f"{prefix}: claim needs exactly one of {joined_kinds}"]
    if ":" not in code:
        return claim_id, code, [f"{prefix}: code '{code}' is not 'module:symbol'"]
    module_name, _, symbol_name = code.partition(":")
    if not any(_binds_into_target(module_name, target) for target in targets):
        joined = ", ".join(targets)
        return claim_id, code, [f"{prefix}: '{module_name}' is outside {joined}"]
    module, import_violations = _import_claim_module(module_name, prefix)
    if module is None:
        return claim_id, code, import_violations
    if not hasattr(module, symbol_name):
        return claim_id, code, [f"{prefix}: '{symbol_name}' not found in {module_name}"]
    return claim_id, code, _run_kind_check(kinds_present[0], claim, module, symbol_name, prefix)


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


def _module_addresses(module_name: str) -> tuple[list[str], list[str]]:
    """Enumerate one module's public symbol addresses.

    Args:
        module_name: Dotted module to read ``__all__`` from.

    Returns:
        Pair of (addresses ``module:symbol``, violations).
    """
    module = importlib.import_module(module_name)
    exported: list[str] | None = getattr(module, "__all__", None)
    if exported is None:
        return [], [f"{module_name}: bound module lacks __all__"]
    return [f"{module_name}:{name}" for name in exported], []


def _public_symbol_addresses(target_name: str) -> tuple[list[str], list[str]]:
    """Enumerate every public symbol address of one bound target.

    A target that resolves to a package contributes every public
    symbol of every submodule; a target that resolves to a plain
    module contributes its own ``__all__`` only. The module arm is
    what lets a large package be onboarded a module at a time.

    Args:
        target_name: Dotted package or module to enumerate.

    Returns:
        Pair of (addresses ``module:symbol``, violations).
    """
    spec = importlib.util.find_spec(target_name)
    if spec is None:
        return [], [f"target '{target_name}' does not resolve"]
    if spec.submodule_search_locations is None:
        return _module_addresses(target_name)
    addresses: list[str] = []
    violations: list[str] = []
    for module_info in pkgutil.iter_modules(list(spec.submodule_search_locations)):
        module_addresses, module_violations = _module_addresses(f"{target_name}.{module_info.name}")
        addresses.extend(module_addresses)
        violations.extend(module_violations)
    return addresses, violations


def _scan_wiki_claims(
    pages_dir: Path,
    targets: tuple[str, ...],
) -> tuple[dict[str, str], list[str]]:
    """Check every claim block under a wiki pages directory.

    Args:
        pages_dir: Directory holding the wiki content pages.
        targets: Bound targets every claim must bind into.

    Returns:
        Pair of (claim id -> bound code address, violations).
    """
    violations: list[str] = []
    claim_codes: dict[str, str] = {}
    for page_path in sorted(pages_dir.glob("*.md")):
        page = page_path.name
        blocks, fence_violations = _extract_claim_blocks(
            page_path.read_text(encoding="utf-8"), page
        )
        violations.extend(fence_violations)
        for raw in blocks:
            claims, parse_violations = _parse_claim_block(raw, page)
            violations.extend(parse_violations)
            for claim in claims:
                claim_id, code, claim_violations = _check_claim(claim, page, targets)
                violations.extend(claim_violations)
                if claim_id and claim_id in claim_codes:
                    violations.append(f"{page}#{claim_id}: duplicate claim id")
                elif claim_id:
                    claim_codes[claim_id] = code
    return claim_codes, violations


def _reverse_coverage_violations(
    addresses: list[str],
    claim_codes: dict[str, str],
) -> list[str]:
    """Require every bound public symbol to be claimed exactly once.

    Args:
        addresses: Public symbol addresses of every bound target.
        claim_codes: Claim id -> bound code address.

    Returns:
        Violations for unclaimed or doubly-claimed symbols.
    """
    violations: list[str] = []
    bound_codes = sorted(claim_codes.values())
    for address in addresses:
        count = bound_codes.count(address)
        if count == 0:
            violations.append(f"{address}: public bound symbol has no wiki claim")
        elif count > 1:
            violations.append(f"{address}: bound by {count} claims, expected exactly 1")
    return violations


def run_physics_claim_rules(
    project_root: Path,
    *,
    package_name: str | None = None,
) -> int:
    """Run the wiki-claim binding rule over a project tree.

    Args:
        project_root: Project root containing ``wiki/pages``.
        package_name: Bind this single target instead of
            :const:`CLAIM_TARGETS`. Used by tests to drive the rule
            against a synthetic fixture package.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    pages_dir = project_root / "wiki" / "pages"
    if not pages_dir.is_dir():
        return 0
    targets = CLAIM_TARGETS if package_name is None else (package_name,)
    claim_codes, violations = _scan_wiki_claims(pages_dir, targets)
    addresses: list[str] = []
    for target in targets:
        target_addresses, target_violations = _public_symbol_addresses(target)
        addresses.extend(target_addresses)
        violations.extend(target_violations)
    violations.extend(_reverse_coverage_violations(addresses, claim_codes))
    for violation in violations:
        sys.stdout.write(f"physics_claim_violation {violation}\n")
    return len(violations)


__all__ = [
    "CLAIM_FENCE_OPEN",
    "CLAIM_KINDS",
    "CLAIM_TARGETS",
    "COMMANDS_MODULE",
    "PHYSICS_PACKAGE",
    "run_physics_claim_rules",
]
