"""An image spec CITES the repository, and until now nothing re-checked the citation.

WHAT THIS CAUGHT ON THE DAY IT WAS WRITTEN. ``model_trainer.cli._test_hooks``
passed the 600-line ceiling and gave up its measurement tables to a sibling
module. ``probed_shapes_hook`` moved with them. Five places in
``specs/abl-image.json`` still named the old home -- one ``required_symbols``
entry and four ``smoke_commands`` -- and every one of them runs INSIDE the
built image, at the end of a build that takes about twenty-five minutes. The
rename was correct, the spec was stale, and the only thing that would have
noticed was the image failing its own self-check after the expensive part was
already spent.

WHY A TEST AND NOT CARE. This is the same defect the wiki solved for itself
with ``source_git_blobs``: a citation that resolves today and is re-checked
by nothing is a claim about the future that nobody is keeping. A spec names
Python module paths in a repository that refactors, and the distance between
the two is a whole package boundary and a wheel build.

WHAT IT DELIBERATELY DOES NOT CHECK. ``smoke_commands`` are arbitrary Python
one-liners -- imports, asserts, comparisons against digests -- and static
analysis of them would either be a parser nobody trusts or a regex that
passes on the cases it was not written for. They stay unchecked here and the
image's own self-check remains the thing that runs them. What IS checked is
the half that can be: every symbol a spec declares must exist, by name, in
this monorepo's source.

A module whose root package this repository does not provide is skipped
rather than failed, and that is a fact rather than an exemption:
``turkic-lstm`` lives in ``~/PROJECTS/LSTM``, so ``char_lstm`` is not here to
check and asserting about it would be asserting about a tree this test cannot
see. What stops that from becoming a hole is the count assertion below --
a rule that silently scans nothing passes forever.
"""

from __future__ import annotations

import ast
import pathlib

from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from hpc3.contracts.image_spec import ImageSpec, SymbolCheck, decode_image_spec

_SPECS = pathlib.Path(__file__).parent.parent / "specs"

_MONOREPO = pathlib.Path(__file__).parents[3]
"""The repository root, three levels above ``tools/hpc3``.

Reached the same way :mod:`tests.test_committed_runs` reaches
``docs/RESEARCH.md``: the thing being checked is not inside this package, and
the check lives here because this package owns the specs.
"""

_SOURCE_ROOTS = "*/*/src"
"""Where every package in this monorepo puts its importable code."""


def _module_files() -> dict[str, pathlib.Path]:
    """Map every module this monorepo provides to the file defining it.

    Returns:
        Dotted module path to file. A package's ``__init__.py`` is recorded
        under the package's own name, so ``platform_core`` and
        ``platform_core.json_utils`` both resolve.
    """
    found: dict[str, pathlib.Path] = {}
    for root in sorted(_MONOREPO.glob(_SOURCE_ROOTS)):
        for path in sorted(root.rglob("*.py")):
            parts = path.relative_to(root).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                found[".".join(parts)] = path
    return found


def _imported_by(node: ast.stmt) -> set[str]:
    """Read the names one import statement binds.

    Args:
        node: A module-level statement.

    Returns:
        The names it binds, or nothing when it is not an import. A dotted
        ``import a.b`` binds ``a``, which is the name an attribute lookup
        would start from.
    """
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    return set()


def _bound_by(node: ast.stmt) -> set[str]:
    """Read the names one module-level statement binds.

    Args:
        node: A module-level statement.

    Returns:
        The names it binds -- a function, a class, an assignment target, an
        annotated target, or an import.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return _imported_by(node)


def _top_level_names(path: pathlib.Path) -> set[str]:
    """Read the names a module defines at its top level.

    Parsed rather than imported. Importing ``model_trainer.core.services``
    pulls torch into the test process to answer a question about a name, and
    a spec is checked most usefully in an environment that does not have the
    image's dependencies at all.

    Args:
        path: The module's file.

    Returns:
        Every name bound at module level.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        names |= _bound_by(node)
    return names


def _wheel_packages() -> dict[str, set[str]]:
    """Map each distribution this monorepo builds to the packages it ships.

    Derived from every ``pyproject.toml``, because a wheel's FILENAME carries
    the distribution name and a spec's symbols carry package names, and the
    two are related by a declaration nobody should retype.
    ``model-trainer-server`` ships ``model_trainer``; nothing about either
    name predicts the other, which is exactly why this cannot be a
    convention.

    Read through ``config_test_hooks.tomllib_loads`` rather than by importing
    ``tomllib``, which the env guard bans outright. The hook is the sanctioned
    seam for exactly this, and reaching for the allow-list instead would have
    been an exemption where a supported path already exists.

    Returns:
        Wheel-filename stem to the top-level packages that distribution
        installs. The key is normalised the way a wheel filename is, with
        hyphens as underscores, so it can be matched against
        :attr:`ImageSpec.wheels` directly.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(_MONOREPO.glob("*/*/pyproject.toml")):
        declared = config_test_hooks.tomllib_loads(path.read_text(encoding="utf-8"))
        if "tool" not in declared:
            continue
        tool = require_dict(declared, "tool")
        if "poetry" not in tool:
            continue
        poetry = require_dict(tool, "poetry")
        if "name" not in poetry or "packages" not in poetry:
            continue
        included: set[str] = set()
        for entry in require_list(poetry, "packages"):
            if isinstance(entry, dict) and "include" in entry:
                included.add(require_str(entry, "include"))
        if included:
            found[require_str(poetry, "name").replace("-", "_")] = included
    return found


def _specs() -> list[tuple[str, ImageSpec]]:
    """Decode every committed image spec.

    A spec is identified by carrying ``base_image``, which is the first field
    :func:`decode_image_spec` requires. That predicate is a SHAPE rather than
    a filename, for the reason :mod:`tests.test_committed_runs` gives for its
    own: ``specs/`` also holds ``retired-envs-abl.json``, a record of an
    environment that no longer exists, and it falls out of scope by being a
    different document rather than by being written into an exemption list
    somebody has to maintain.

    Returns:
        Each spec's filename and decoded body, in filename order.

    Raises:
        JSONTypeError: If a document carrying ``base_image`` cannot be
            decoded. A spec that no longer reads is a build nobody can
            repeat, which is exactly the claim ``specs/`` makes.
    """
    found: list[tuple[str, ImageSpec]] = []
    for path in sorted(_SPECS.glob("*.json")):
        document = narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
        if "base_image" not in document:
            continue
        found.append((path.name, decode_image_spec(document)))
    return found


def _checkable(modules: dict[str, pathlib.Path]) -> list[tuple[str, SymbolCheck]]:
    """Select the declared symbols whose package this repository provides.

    Args:
        modules: Every module this monorepo defines.

    Returns:
        Each checkable symbol with the spec filename that declares it.
    """
    roots = {name.split(".")[0] for name in modules}
    return [
        (filename, symbol)
        for filename, spec in _specs()
        for symbol in spec["required_symbols"]
        if symbol["module"].split(".")[0] in roots
    ]


def unresolved(
    symbols: list[tuple[str, SymbolCheck]], modules: dict[str, pathlib.Path]
) -> list[str]:
    """Name every declared symbol that this repository no longer defines.

    Factored out of the assertions so the rule can be pointed at a symbol
    that is KNOWN to be broken. A checker that has only ever been run against
    a passing tree has not been shown to check anything -- which is the same
    argument the smoke commands in these specs make about images.

    Args:
        symbols: Each symbol with the spec filename declaring it.
        modules: Every module this monorepo defines.

    Returns:
        One ``<spec>: <module>:<attribute>`` line per unresolved symbol, in
        the order given.
    """
    missing: list[str] = []
    for filename, symbol in symbols:
        location = f"{filename}: {symbol['module']}:{symbol['attribute']}"
        path = modules.get(symbol["module"])
        if path is None or symbol["attribute"] not in _top_level_names(path):
            missing.append(location)
    return missing


class TestEverySpecDecodes:
    """A spec that cannot be read builds nothing, and says so only when run."""

    def test_there_are_specs_to_check(self) -> None:
        # A rule that silently scans nothing passes forever.
        assert len(_specs()) >= 5

    def test_every_committed_spec_decodes(self) -> None:
        # Decoding IS the assertion: _specs() raises rather than returning a
        # spec it could not read.
        assert [filename for filename, _ in _specs()] != []


class TestEverySymbolAnImageAssertsStillExists:
    """The half of a spec that names this repository, re-checked."""

    def test_the_scan_reaches_this_monorepo(self) -> None:
        # If the glob missed, every assertion below would pass vacuously.
        modules = _module_files()

        assert "platform_core.json_utils" in modules
        assert "model_trainer.cli.continuations" in modules

    def test_some_declared_symbols_are_checkable(self) -> None:
        # `turkic-lstm` lives outside this repository, so its symbols are
        # legitimately unreachable. This pins that the OTHERS are reached.
        assert len(_checkable(_module_files())) >= 40

    def test_every_first_party_wheel_is_asserted_by_at_least_one_symbol(self) -> None:
        """A wheel nothing names cannot be caught being stale.

        This is the property the hand-transcribed list in
        :mod:`tests.test_committed_image_spec` was standing in for, and it is
        the reason that list said "a required symbol only detects a stale
        wheel if it names something the new code introduced". Stated as an
        invariant it can be derived; stated as a list of forty-six pairs it
        went stale fourteen times, and was red on ``main`` across nine
        consecutive commits.
        """
        packages = _wheel_packages()
        gaps: list[str] = []
        for filename, spec in _specs():
            asserted = {symbol["module"].split(".")[0] for symbol in spec["required_symbols"]}
            for wheel in spec["wheels"]:
                distribution = wheel.split("-")[0]
                shipped = packages.get(distribution)
                if shipped is not None and not (shipped & asserted):
                    gaps.append(f"{filename}: {wheel} ships {sorted(shipped)}, none asserted")

        assert gaps == []

    def test_the_wheel_map_reaches_this_monorepo(self) -> None:
        # Without this, the assertion above passes by mapping nothing.
        packages = _wheel_packages()

        assert packages["model_trainer_server"] == {"model_trainer"}
        assert packages["platform_core"] == {"platform_core"}

    def test_every_declared_symbol_still_resolves(self) -> None:
        modules = _module_files()

        assert unresolved(_checkable(modules), modules) == []

    def test_a_module_that_no_longer_exists_is_caught(self) -> None:
        # The negative control for the module half.
        modules = _module_files()
        moved = SymbolCheck(module="model_trainer.cli.gone_away", attribute="anything")

        assert unresolved([("synthetic.json", moved)], modules) == [
            "synthetic.json: model_trainer.cli.gone_away:anything"
        ]

    def test_an_attribute_that_moved_out_of_a_surviving_module_is_caught(self) -> None:
        # The case that actually happened, as a control rather than a story:
        # `_test_hooks` still exists and `probed_shapes_hook` is no longer in
        # it, so a module-only check would pass on a spec already broken.
        modules = _module_files()
        moved = SymbolCheck(module="model_trainer.cli._test_hooks", attribute="probed_shapes_hook")

        assert unresolved([("synthetic.json", moved)], modules) == [
            "synthetic.json: model_trainer.cli._test_hooks:probed_shapes_hook"
        ]

    def test_the_attribute_that_moved_resolves_in_its_new_home(self) -> None:
        # And the other half: the rename was correct, so the symbol the spec
        # names TODAY must be found.
        modules = _module_files()
        here = SymbolCheck(
            module="model_trainer.cli._measurement_hooks", attribute="probed_shapes_hook"
        )

        assert unresolved([("synthetic.json", here)], modules) == []
