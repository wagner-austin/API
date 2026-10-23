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

The Rust toolchain a linux fleet worker is given
(:mod:`fleet.contracts.rust_toolchain`, MCPs board task 91bb5cda) is
rendered here beside it, for MCPs ``scripts/host/lib/rust-toolchain.sh``,
which every worker's provision runs.

THE SHAPE: one object per tool, keyed by the tool's own name, so a reader
asks ``.poetry.version`` or ``.["cargo-llvm-cov"].version`` and a later pin
is a new key, not a new document. ``rust`` carries the rustup components
beside its version, and ``rustup`` the one target its installer is fetched
for and that installer's SHA-256.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import dump_json_str

from fleet.contracts.rust_toolchain import (
    PINNED_CARGO_LLVM_COV,
    PINNED_RUST,
    PINNED_RUSTUP,
    PINNED_UV,
    RUST_COMPONENTS,
    RUSTUP_INIT_SHA256,
    RUSTUP_TARGET,
)
from fleet.contracts.toolchain import PINNED_POETRY

#: The committed copy's file name under ``rendered/``.
TOOLCHAIN_DOCUMENT: Final = "toolchain-pins.json"


def render_toolchain_document() -> str:
    """The document, as committed.

    Returns:
        Every pinned tool's version, as indented JSON ending in one newline.
    """
    document = {
        "poetry": {"version": PINNED_POETRY},
        "rust": {"version": PINNED_RUST, "components": list(RUST_COMPONENTS)},
        "rustup": {
            "version": PINNED_RUSTUP,
            "target": RUSTUP_TARGET,
            "sha256": RUSTUP_INIT_SHA256,
        },
        "cargo-llvm-cov": {"version": PINNED_CARGO_LLVM_COV},
        "uv": {"version": PINNED_UV},
    }
    return dump_json_str(document, indent=2) + "\n"


__all__ = ["TOOLCHAIN_DOCUMENT", "render_toolchain_document"]
