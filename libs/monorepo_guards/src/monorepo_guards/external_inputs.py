"""Every path outside a checked package that the guard rules read.

WHY THIS EXISTS. Running the guards over one package is not a self-contained
act. Three rules resolve their declaring module from the monorepo root by
design -- :mod:`monorepo_guards.literal_set_rules` says so in as many words,
because looking only among the checked package's own files made those rules
inert for every set whose users live elsewhere, and they reported "0
violations" while checking nothing. :mod:`monorepo_guards.config_rules` scans
every package manifest under the category directories for the same reason.

So a caller that assembles a PARTIAL tree and runs ``make check`` in it -- the
fleet dispatcher does exactly this, sending one project and its dependencies
to another machine -- has to know which outside paths to carry. Measured
2026-09-04: a dispatch that carried the project, its path dependencies and the
shared launcher directories failed on ``corpus-format-declaration-unresolved``,
``risk-tier-declaration-unresolved`` and ``strategy-name-declaration-unresolved``
because the three declaring modules had not travelled with it.

That caller must not have to REDERIVE this list. A second copy of "what the
guards read" would drift towards carrying too little, and too little surfaces
as three guard failures on a remote node that read as the project's fault. So
the rules' own package answers the question, from the same constants the rules
themselves use.

WHAT IS DELIBERATELY NOT HERE. Paths a rule reads from INSIDE the package
under check, which the caller has by definition, and paths that are optional
in the sense that their absence narrows coverage without failing --
:func:`external_inputs` returns those too, because a check that silently
covers less than the local one is the failure mode this whole module is
about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from monorepo_guards.literal_set_rules import PACKAGE_SOURCE_GLOB, REGISTERED_SETS

#: The directories package manifests are looked for under.
#:
#: The single home for this tuple; :mod:`monorepo_guards.config_rules` reads it
#: from here rather than spelling it again, so a fourth category is added in
#: one place and both the scan and the fleet's staging pick it up.
CATEGORY_DIRECTORIES: Final[tuple[str, ...]] = ("services", "clients", "libs")

#: The document naming which rules run, read from the monorepo root.
GUARD_CONFIG_NAME: Final = "monorepo-guards.toml"


def external_inputs(monorepo_root: Path) -> tuple[Path, ...]:
    """Name every path outside a checked package that the rules read.

    Args:
        monorepo_root: Absolute path to the monorepo root.

    Returns:
        Absolute paths that exist, sorted and deduplicated: the guard config,
        every package manifest under :data:`CATEGORY_DIRECTORIES`, and each
        registered literal set's declaring module. Only existing paths are
        returned, because the caller's use for this is deciding what to COPY
        and a name that resolves to nothing cannot be copied -- a set whose
        declaring module is genuinely absent is a violation the rules
        themselves report, and reporting it twice in different words would
        send the reader to the wrong place.
    """
    found: set[Path] = set()
    config = monorepo_root / GUARD_CONFIG_NAME
    if config.is_file():
        found.add(config)
    found.update(package_manifests(monorepo_root))
    found.update(declaring_modules(monorepo_root))
    return tuple(sorted(found))


def package_manifests(monorepo_root: Path) -> tuple[Path, ...]:
    """Find every package manifest the config rule scans.

    Args:
        monorepo_root: Absolute path to the monorepo root.

    Returns:
        Each existing ``<category>/<package>/pyproject.toml``, sorted.
    """
    found: list[Path] = []
    for category in CATEGORY_DIRECTORIES:
        directory = monorepo_root / category
        if not directory.is_dir():
            continue
        for package in sorted(directory.iterdir()):
            manifest = package / "pyproject.toml"
            if manifest.is_file():
                found.append(manifest)
    return tuple(found)


def declaring_modules(monorepo_root: Path) -> tuple[Path, ...]:
    """Find the module that declares each registered literal set.

    Resolved with the same glob :class:`LiteralSetRule` uses, so the paths
    this returns are exactly the ones that rule will go looking for.

    Args:
        monorepo_root: Absolute path to the monorepo root.

    Returns:
        Each existing declaring module, sorted. A set whose module is absent
        contributes nothing rather than raising -- see
        :func:`external_inputs` for why that is not a silence.
    """
    found: list[Path] = []
    for source_root in sorted(monorepo_root.glob(PACKAGE_SOURCE_GLOB)):
        for declared in REGISTERED_SETS:
            module = source_root / declared.defining_module
            if module.is_file():
                found.append(module)
    return tuple(sorted(found))


__all__ = [
    "CATEGORY_DIRECTORIES",
    "GUARD_CONFIG_NAME",
    "declaring_modules",
    "external_inputs",
    "package_manifests",
]
