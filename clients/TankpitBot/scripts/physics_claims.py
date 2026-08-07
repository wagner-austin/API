"""Guard rule: every wiki physics claim binds to real code.

Scans the wiki for ``json claims`` fences, resolves each claim's
``module:symbol`` against the claim targets, dispatches to the checker
for its kind, and enforces reverse coverage -- every public symbol of a
claim target must carry a claim. The per-kind checkers are
:mod:`scripts.physics_claim_checks`.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    load_json_str,
)
from platform_core.logging import get_logger

from scripts.physics_claim_checks import _run_kind_check

PHYSICS_PACKAGE = "tankpit_bot.physics"

COMMANDS_MODULE = "tankpit_bot.protocol.commands"

PROTOCOL_CONSTANTS_MODULE = "tankpit_bot.protocol.constants"

#: Every target the wiki must bind, in onboarding order. A package
#: binds every public symbol of every submodule; a bare module binds
#: only its own ``__all__``. Adding a target here is a commitment:
#: reverse coverage immediately requires a claim for each of its
#: public symbols, so the claims land in the same commit.
LEDGER_PACKAGE = "tankpit_bot.ledger"

#: Shot clearance is a physics LAW that reads terrain state, so the
#: module lives in ``state`` (a query) while the law it encodes is
#: claim-bound here. Bare module: only its own ``__all__``
#: (``is_shot_line_clear``, ``shot_line_tiles``) is bound.
LINE_OF_SIGHT_MODULE = "tankpit_bot.state.line_of_sight"

CLAIM_TARGETS: tuple[str, ...] = (
    PHYSICS_PACKAGE,
    COMMANDS_MODULE,
    PROTOCOL_CONSTANTS_MODULE,
    LEDGER_PACKAGE,
    LINE_OF_SIGHT_MODULE,
)

CLAIM_FENCE_OPEN = "```json claims"

CLAIM_FENCE_CLOSE = "```"

#: Claim kinds. Exactly one must appear on every claim. ``law`` is the
#: weak one — existence plus prose — and exists only for symbols no
#: other kind can verify; prefer any of the others when they fit.
CLAIM_KINDS: tuple[str, ...] = ("value", "bytes", "members", "keys", "probes", "law")

_LOGGER = get_logger(__name__)


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
    "CLAIM_FENCE_CLOSE",
    "CLAIM_FENCE_OPEN",
    "CLAIM_KINDS",
    "CLAIM_TARGETS",
    "run_physics_claim_rules",
]
