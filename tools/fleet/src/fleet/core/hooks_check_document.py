"""The hooks check pins as a JSON document, for readers outside this package.

:data:`fleet.contracts.tagged_tools.HOOKS_CHECK_PACKAGES` is the one
declaration of the tools MCPs ``packages/claude-hooks``'s make check runs on
the system interpreter, at their versions. Two readers outside this package
need it, and neither can import Python from here: MCPs
``scripts/host/session-host/provision.sh`` installs them on the session host
(until MCPs board task 88ca2402 it parsed ``tagged_tools.py`` with ``ast``),
and the fleet audit (MCPs ``fleet-mcp``) measures them on every machine that
registers the hooks, reading ``tools/fleet`` from the checkout beside its
workspace. So the tuple is rendered to ``rendered/`` as
:data:`HOOKS_CHECK_DOCUMENT`, committed, and held equal to the contract by
``tests/test_hooks_check_document.py``: ``make render-documents`` rewrites
it after a pin moves.

THE SHAPE: ``{"packages": [{"name": ..., "version": ...}, ...]}`` in the
contract's order, each pin split at its ``==``, so a reader neither parses a
requirement specifier nor learns which separator this package uses.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import dump_json_str

from fleet.contracts.tagged_tools import HOOKS_CHECK_PACKAGES

#: The committed copy's file name under ``rendered/``.
HOOKS_CHECK_DOCUMENT: Final = "hooks-check-packages.json"


def _pin(package: str) -> dict[str, str]:
    """One pin as the document carries it.

    Args:
        package: A ``name==version`` entry of :data:`HOOKS_CHECK_PACKAGES`.

    Returns:
        ``{"name": name, "version": version}``.
    """
    name, version = package.split("==")
    return {"name": name, "version": version}


def render_hooks_check_document() -> str:
    """The document, as committed.

    Returns:
        Every pin of :data:`HOOKS_CHECK_PACKAGES` in its order, as indented
        JSON ending in one newline.
    """
    pins = [_pin(package) for package in HOOKS_CHECK_PACKAGES]
    return dump_json_str({"packages": pins}, indent=2) + "\n"


__all__ = ["HOOKS_CHECK_DOCUMENT", "render_hooks_check_document"]
