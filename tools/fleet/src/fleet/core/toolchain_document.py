"""The toolchain's pinned versions as a JSON document, for readers outside this package.

:data:`fleet.contracts.toolchain.PINNED_POETRY` is the one declaration of the
Poetry every machine that commits or bakes a working tree runs (MCPs board
task ce698bfb). MCPs ``scripts/host/session-host/provision.sh`` and
``scripts/host/lib/fleet-exec-docker.sh`` install it on the session host and
on diphtheria and cannot import Python from here, so the pin is rendered to
``rendered/`` as :data:`TOOLCHAIN_DOCUMENT`, committed, and held equal to the
contract by ``tests/test_toolchain_document.py``, as the hooks check pins
are (:mod:`fleet.core.hooks_check_document`): ``make render-documents``
rewrites both after a pin moves.

THE SHAPE: ``{"poetry": {"version": ...}}``, one object per tool, so a
reader asks ``.poetry.version`` and a later pin is a new key, not a new
document.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import dump_json_str

from fleet.contracts.toolchain import PINNED_POETRY

#: The committed copy's file name under ``rendered/``.
TOOLCHAIN_DOCUMENT: Final = "toolchain-pins.json"


def render_toolchain_document() -> str:
    """The document, as committed.

    Returns:
        Every pinned tool's version, as indented JSON ending in one newline.
    """
    return dump_json_str({"poetry": {"version": PINNED_POETRY}}, indent=2) + "\n"


__all__ = ["TOOLCHAIN_DOCUMENT", "render_toolchain_document"]
