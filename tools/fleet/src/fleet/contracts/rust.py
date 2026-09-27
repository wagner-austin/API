"""The Rust toolchain a node carries, as declared and as its probe measured it.

WHY A NODE DECLARES IT AT ALL (MCPs board task 1e2da299). API
``services/covenant-radar-api`` depends on ``libs/cleargbm_rs``, a maturin
crate, as a path dependency without ``develop = true``, so a clean export
builds it from source through ``cargo``. Probed over ssh on 2026-09-26,
cargo was missing on diphtheria, sedona, serendipity and lavender: a job for
that project handed to any of them staged the whole tree and failed in
``poetry sync``. A project that builds a crate therefore requires the
``rust`` tag (:mod:`fleet.contracts.tags`), and a node carries that tag when
its declaration names a Rust toolchain.

WHY THE DECLARATION IS A VERSION AND IS RE-MEASURED. A boolean would be the
hand-written tag the tags module exists to avoid, true until somebody
remembered to change it. The declaration is instead the exact version
``cargo --version`` printed on that node, and the runner's toolchain probe
asks cargo again every tick; a node whose answer differs from its
declaration claims nothing (``NODE_RUST_MISMATCH``,
:func:`fleet.core.toolchain.readiness_gap`), so the tag is exactly as true as
the last measurement. A node that has cargo and declares none only withholds
the tag, which misroutes nothing, so that direction is reported by the ready
summary rather than refused.
"""

from __future__ import annotations

import re
from typing import Final

from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.toolchain import ToolReport

#: The executable the probe asks, and the one maturin builds a crate with.
CARGO: Final = "cargo"

#: What a declared version looks like: ``cargo --version``'s number, e.g.
#: ``1.98.1`` (measured on diphtheria 2026-09-27, which printed
#: ``cargo 1.98.1 (797e8a9bc 2026-08-05)``).
RUST_VERSION: Final = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def measured_rust(reports: tuple[ToolReport, ...]) -> str | None:
    """Read the Rust toolchain a node's probe reported.

    THE SECOND TOKEN, NOT THE LAST. :func:`fleet.contracts.toolchain.version_number`
    takes the trailing token, which for cargo is its build date. A present
    cargo whose answer does not read ``cargo <number> ...`` (rustup prints an
    error through the cargo shim when no default toolchain is set) is
    returned verbatim, so it can equal no declaration and the refusal quotes
    what the node said.

    Args:
        reports: What the node answered.

    Returns:
        The version cargo reported, the whole answer when it is not of that
        shape, or None when the probe reported no present cargo.
    """
    for report in reports:
        if report["name"] != CARGO or not report["present"]:
            continue
        words = report["version"].split()
        if len(words) >= 2 and words[0] == CARGO and RUST_VERSION.fullmatch(words[1]):
            return words[1]
        return report["version"]
    return None


def rust_gap(declared: str | None, reports: tuple[ToolReport, ...]) -> str | None:
    """Say how a node's declared Rust toolchain disagrees with its probe.

    Args:
        declared: The node's ``rust`` declaration.
        reports: What the node answered.

    Returns:
        None when the node declares no toolchain, or declares exactly the
        version its cargo reported. Otherwise the disagreement, naming both
        and the declaration that would match.
    """
    if declared is None:
        return None
    measured = measured_rust(reports)
    if measured == declared:
        return None
    found = "no cargo" if measured is None else f"cargo {measured!r}"
    fix = "null" if measured is None or not RUST_VERSION.fullmatch(measured) else repr(measured)
    return (
        f"declares rust {declared!r} but its probe reports {found}; the declaration is what "
        "gives a node the rust tag, so a crate build claimed on it would fail in poetry sync. "
        f"Set rust to {fix}, or install that toolchain"
    )


def decode_rust(value: JSONValue) -> str | None:
    """Read a node's ``rust`` declaration.

    Args:
        value: The declared value.

    Returns:
        The version, or None for a node with no Rust toolchain.

    Raises:
        JSONTypeError: If it is neither null nor a string of
            :data:`RUST_VERSION`'s shape.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not RUST_VERSION.fullmatch(value):
        raise JSONTypeError(
            f"rust must be null or the version cargo --version prints, e.g. '1.98.1', got "
            f"{value!r}; it is compared with the node's probe every tick"
        )
    return value


__all__ = ["CARGO", "RUST_VERSION", "decode_rust", "measured_rust", "rust_gap"]
