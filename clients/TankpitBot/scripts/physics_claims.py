"""Guard rule: every wiki physics claim binds to real code.

Scans the wiki for ``json claims`` fences, resolves each claim's
``module:symbol`` against the claim targets, dispatches to the checker
for its kind, and enforces reverse coverage -- every public symbol of a
claim target must carry a claim. The per-kind checkers are
:mod:`scripts.physics_claim_checks`.
"""

from __future__ import annotations

import ast
import importlib
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

#: The two halves that left ``commands.py`` when it was split by role
#: 2026-09-03. They are separate targets rather than one prefix because
#: ``_binds_into`` matches a module or its SUBmodules, and these are
#: siblings of ``commands``, not children of it.
COMMAND_FRAMES_MODULE = "tankpit_bot.protocol.command_frames"
COMMAND_BUILDERS_MODULE = "tankpit_bot.protocol.command_builders"

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
    COMMAND_FRAMES_MODULE,
    COMMAND_BUILDERS_MODULE,
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


def _symbol_is_exported(module_name: str, symbol_name: str, source_root: Path) -> bool:
    """Report whether a module in THIS tree exports the claimed symbol.

    The existence half of a claim, answered from source so that it
    describes the tree under check. ``_import_claim_module`` below
    still imports, because the per-kind checks compare a claimed value
    against the live object and no amount of parsing produces one --
    so a claim's VALUE is only ever verified against the installed
    tree, while its BINDING is verified against whichever tree this is
    pointed at. Committing a wiki page ahead of the code it describes
    fails the binding half, which is the case this separation exists
    for.

    Args:
        module_name: Dotted module from the claim's code address.
        symbol_name: The symbol the claim binds to.
        source_root: Directory the dotted name is rooted at.

    Returns:
        True when the module binds the name at module level. This is
        the source-level reading of ``hasattr``, which is what it
        replaces -- NOT ``__all__`` membership, which is a stricter
        question the reverse-coverage half asks separately. A claim may
        legitimately name a symbol the module does not re-export.
    """
    module_path = _module_source_path(module_name, source_root)
    if module_path is None:
        return False
    return symbol_name in _module_level_names(module_path.read_text(encoding="utf-8"))


def _module_source_path(module_name: str, source_root: Path) -> Path | None:
    """Locate the ``.py`` file a dotted module name names in a tree.

    Args:
        module_name: Dotted module path.
        source_root: Directory the dotted name is rooted at.

    Returns:
        The module file, a package's ``__init__.py``, or None when the
        tree holds neither.
    """
    module_path = _target_path(module_name, source_root).with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_init = _target_path(module_name, source_root) / "__init__.py"
    return package_init if package_init.is_file() else None


def _module_level_names(source: str) -> frozenset[str]:
    """Collect every name a module binds at module level.

    Args:
        source: A module's source text.

    Returns:
        Names an importer would find as attributes: assignments,
        annotated assignments, functions, classes, and imported names
        under whichever spelling they land as.
    """
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return frozenset(names)


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
    source_root: Path,
) -> tuple[str, str, list[str]]:
    """Fully verify one claim.

    Args:
        claim: Claim object.
        page: Page name for violation messages.
        targets: Bound targets; the claim must bind into one of them.
        source_root: Directory the claim's dotted module is rooted at.

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
    # Absent MODULE and absent SYMBOL are different repairs, so they stay
    # different messages -- and both are answered from the tree under
    # check, before the import that can only speak for the installed one.
    if _module_source_path(module_name, source_root) is None:
        return claim_id, code, [f"{prefix}: module '{module_name}' does not import"]
    if not _symbol_is_exported(module_name, symbol_name, source_root):
        return claim_id, code, [f"{prefix}: '{symbol_name}' not found in {module_name}"]
    module, import_violations = _import_claim_module(module_name, prefix)
    if module is None:
        return claim_id, code, import_violations
    if not hasattr(module, symbol_name):
        return claim_id, code, [f"{prefix}: '{symbol_name}' not found in {module_name}"]
    return claim_id, code, _run_kind_check(kinds_present[0], claim, module, symbol_name, prefix)


def _exported_names(source: str) -> list[str] | None:
    """Read a module's ``__all__`` out of its source text.

    Read rather than imported, and that is the whole point of this
    module's binding half. Importing resolves through ``sys.path`` to
    the INSTALLED package, which under an editable install is the
    working tree -- so a run that points the wiki half at one revision
    and the code half at another compares a pair that exists nowhere.
    Reading the source makes both halves functions of the same tree,
    which is what lets the rule be run against a committed revision.

    Args:
        source: A module's source text.

    Returns:
        The exported names in declaration order, or None when the
        module declares no ``__all__`` or declares one this cannot
        read. Every one of tankpit_bot's 495 declarations is a literal
        list of literal strings; a computed one returns None and is
        reported as missing rather than guessed at.
    """
    for node in ast.parse(source).body:
        # Both spellings occur here: a bare `__all__ = [...]` and an
        # annotated `__all__: tuple[str, ...] = ()`. Reading only the
        # first reported `ledger.outcome` as undeclared when it
        # declares an explicitly empty one.
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            named = any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            named = isinstance(node.target, ast.Name) and node.target.id == "__all__"
            value = node.value
        else:
            continue
        if not named or value is None:
            continue
        if not isinstance(value, ast.List | ast.Tuple):
            return None
        names: list[str] = []
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            names.append(element.value)
        return names
    return None


def _module_addresses(module_name: str, module_path: Path) -> tuple[list[str], list[str]]:
    """Enumerate one module's public symbol addresses.

    Args:
        module_name: Dotted module the addresses are reported under.
        module_path: The ``.py`` file to read ``__all__`` from.

    Returns:
        Pair of (addresses ``module:symbol``, violations).
    """
    exported = _exported_names(module_path.read_text(encoding="utf-8"))
    if exported is None:
        return [], [f"{module_name}: bound module lacks __all__"]
    return [f"{module_name}:{name}" for name in exported], []


def _target_path(target_name: str, source_root: Path) -> Path:
    """Locate a dotted target inside a source tree.

    Args:
        target_name: Dotted package or module.
        source_root: Directory the dotted name is rooted at.

    Returns:
        The package directory or the module file; the returned path
        may not exist, which the caller reports.
    """
    return source_root.joinpath(*target_name.split("."))


def _public_symbol_addresses(target_name: str, source_root: Path) -> tuple[list[str], list[str]]:
    """Enumerate every public symbol address of one bound target.

    A target that resolves to a package contributes every public
    symbol of every submodule; a target that resolves to a plain
    module contributes its own ``__all__`` only. The module arm is
    what lets a large package be onboarded a module at a time.

    Args:
        target_name: Dotted package or module to enumerate.
        source_root: Directory the dotted name is rooted at, so the
            target is read from the tree under check rather than from
            whichever copy happens to be installed.

    Returns:
        Pair of (addresses ``module:symbol``, violations).
    """
    target = _target_path(target_name, source_root)
    if target.is_dir() and (target / "__init__.py").is_file():
        addresses: list[str] = []
        violations: list[str] = []
        for child in sorted(target.iterdir()):
            if child.name == "__init__.py":
                continue
            if child.is_dir() and (child / "__init__.py").is_file():
                name, path = f"{target_name}.{child.name}", child / "__init__.py"
            elif child.suffix == ".py":
                name, path = f"{target_name}.{child.stem}", child
            else:
                continue
            child_addresses, child_violations = _module_addresses(name, path)
            addresses.extend(child_addresses)
            violations.extend(child_violations)
        return addresses, violations
    module = target.with_suffix(".py")
    if module.is_file():
        return _module_addresses(target_name, module)
    return [], [f"target '{target_name}' does not resolve"]


def _scan_wiki_claims(
    pages_dir: Path,
    targets: tuple[str, ...],
    source_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Check every claim block under a wiki pages directory.

    Args:
        pages_dir: Directory holding the wiki content pages.
        targets: Bound targets every claim must bind into.
        source_root: Directory the bound targets are rooted at.

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
                claim_id, code, claim_violations = _check_claim(claim, page, targets, source_root)
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
    source_root: Path | None = None,
) -> int:
    """Run the wiki-claim binding rule over a project tree.

    Both halves of the comparison -- the claims in ``wiki/pages`` and
    the public symbols they must bind to -- are read from the tree this
    is pointed at. That is what makes the rule answerable about a
    COMMITTED revision: extract one with ``git archive`` and pass its
    root, and the answer describes that revision rather than whatever
    is installed. Binding by import instead would read the wiki from
    the extracted tree and the symbols from the editable install, a
    pair that exists in no revision and is green by construction.

    Args:
        project_root: Project root containing ``wiki/pages``.
        package_name: Bind this single target instead of
            :const:`CLAIM_TARGETS`. Used by tests to drive the rule
            against a synthetic fixture package.
        source_root: Directory the target's dotted name is rooted at.
            Defaults to ``project_root / "src"``, the layout every
            package here uses.

    Returns:
        Number of violations found (0 means the rule passes).
    """
    pages_dir = project_root / "wiki" / "pages"
    if not pages_dir.is_dir():
        return 0
    roots = project_root / "src" if source_root is None else source_root
    targets = CLAIM_TARGETS if package_name is None else (package_name,)
    claim_codes, violations = _scan_wiki_claims(pages_dir, targets, roots)
    addresses: list[str] = []
    for target in targets:
        target_addresses, target_violations = _public_symbol_addresses(target, roots)
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
