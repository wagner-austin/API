"""Guard rule: wiki physics claims must match the physics package.

The wiki is the source of truth for game physics; the code acts on
it. This rule machine-checks the binding in both directions on every
``make check`` (see ``wiki/pages/physics-module-roadmap.md`` Phase 1):

* Forward: every ``json claims`` fenced block in ``wiki/pages/*.md``
  names a symbol in :mod:`tankpit_bot.physics` and its expected value
  (an int constant) or probe grid (a formula checked at explicit
  points). The rule imports the symbol and verifies computationally.
* Reverse: every public symbol (``__all__``) of every submodule of
  :mod:`tankpit_bot.physics` must be bound by exactly one claim —
  a physics fact the wiki does not claim is a violation too.

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
from pathlib import Path
from types import CodeType, ModuleType
from typing import Protocol

from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    load_json_str,
)
from platform_core.logging import get_logger

PHYSICS_PACKAGE = "tankpit_bot.physics"
CLAIM_FENCE_OPEN = "```json claims"
CLAIM_FENCE_CLOSE = "```"

_LOGGER = get_logger(__name__)


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
    package_name: str,
) -> tuple[str, str, list[str]]:
    """Fully verify one claim.

    Args:
        claim: Claim object.
        page: Page name for violation messages.
        package_name: Package every claim must bind into.

    Returns:
        Tuple of (claim id, bound code address, violations). The id and
        code are empty strings when the claim is too malformed to name.
    """
    claim_id = claim.get("id")
    code = claim.get("code")
    if not isinstance(claim_id, str) or not isinstance(code, str):
        return "", "", [f"{page}: claim needs string 'id' and 'code' fields"]
    prefix = f"{page}#{claim_id}"
    kinds_present = [kind for kind in ("value", "probes", "law") if kind in claim]
    if len(kinds_present) != 1:
        return claim_id, code, [f"{prefix}: claim needs exactly one of value/probes/law"]
    if ":" not in code:
        return claim_id, code, [f"{prefix}: code '{code}' is not 'module:symbol'"]
    module_name, _, symbol_name = code.partition(":")
    if not module_name.startswith(f"{package_name}."):
        return claim_id, code, [f"{prefix}: '{module_name}' is outside {package_name}"]
    module, import_violations = _import_claim_module(module_name, prefix)
    if module is None:
        return claim_id, code, import_violations
    if not hasattr(module, symbol_name):
        return claim_id, code, [f"{prefix}: '{symbol_name}' not found in {module_name}"]
    if kinds_present[0] == "value":
        return claim_id, code, _check_value_claim(claim, module, symbol_name, prefix)
    if kinds_present[0] == "probes":
        return claim_id, code, _check_probe_claim(claim, module, symbol_name, prefix)
    return claim_id, code, _check_law_claim(claim, prefix)


def _public_symbol_addresses(package_name: str) -> tuple[list[str], list[str]]:
    """Enumerate every public symbol address of the physics package.

    Args:
        package_name: Dotted package to enumerate submodules of.

    Returns:
        Pair of (addresses ``module:symbol``, violations).
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return [], [f"package '{package_name}' does not resolve to a package"]
    addresses: list[str] = []
    violations: list[str] = []
    for module_info in pkgutil.iter_modules(list(spec.submodule_search_locations)):
        module_name = f"{package_name}.{module_info.name}"
        module = importlib.import_module(module_name)
        exported: list[str] | None = getattr(module, "__all__", None)
        if exported is None:
            violations.append(f"{module_name}: physics module lacks __all__")
            continue
        addresses.extend(f"{module_name}:{name}" for name in exported)
    return addresses, violations


def _scan_wiki_claims(
    pages_dir: Path,
    package_name: str,
) -> tuple[dict[str, str], list[str]]:
    """Check every claim block under a wiki pages directory.

    Args:
        pages_dir: Directory holding the wiki content pages.
        package_name: Package every claim must bind into.

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
                claim_id, code, claim_violations = _check_claim(claim, page, package_name)
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
    """Require every public physics symbol to be claimed exactly once.

    Args:
        addresses: Public symbol addresses of the physics package.
        claim_codes: Claim id -> bound code address.

    Returns:
        Violations for unclaimed or doubly-claimed symbols.
    """
    violations: list[str] = []
    bound_codes = sorted(claim_codes.values())
    for address in addresses:
        count = bound_codes.count(address)
        if count == 0:
            violations.append(f"{address}: public physics symbol has no wiki claim")
        elif count > 1:
            violations.append(f"{address}: bound by {count} claims, expected exactly 1")
    return violations


def run_physics_claim_rules(
    project_root: Path,
    *,
    package_name: str = PHYSICS_PACKAGE,
) -> int:
    """Run the wiki-claim binding rule over a project tree.

    Args:
        project_root: Project root containing ``wiki/pages``.
        package_name: Physics package to bind (overridable for tests).

    Returns:
        Number of violations found (0 means the rule passes).
    """
    pages_dir = project_root / "wiki" / "pages"
    if not pages_dir.is_dir():
        return 0
    claim_codes, violations = _scan_wiki_claims(pages_dir, package_name)
    addresses, package_violations = _public_symbol_addresses(package_name)
    violations.extend(package_violations)
    violations.extend(_reverse_coverage_violations(addresses, claim_codes))
    for violation in violations:
        sys.stdout.write(f"physics_claim_violation {violation}\n")
    return len(violations)


__all__ = [
    "CLAIM_FENCE_OPEN",
    "PHYSICS_PACKAGE",
    "run_physics_claim_rules",
]
