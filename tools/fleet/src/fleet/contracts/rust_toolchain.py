"""The Rust toolchain every linux fleet worker is given, at the versions the fleet pins.

MCPs board task 91bb5cda. The ``rust`` tag means the Rust check's whole
toolchain (:data:`fleet.contracts.capability.RUST_CHECK_SUBCOMMANDS` and
``RUST_CHECK_TOOLS``, MCPs board task b1089284), and on 2026-10-23 only
diphtheria carried it, installed by hand, while its budget checks only the
projects pinned to it: libs/covenant_ml's check (dispatch job f9921570) sat
queued from 04:33Z with every runner refusing it. A worker is provisioned
by MCPs ``scripts/host/fleet-node/provision.sh``, which cannot import Python
from here, so these pins are rendered beside Poetry's into
``rendered/toolchain-pins.json`` (:mod:`fleet.core.toolchain_document`) and
read there by MCPs ``scripts/host/lib/rust-toolchain.sh``.

THE VERSIONS ARE THE ONES DIPHTHERIA ANSWERED on 2026-10-23, over ssh as its
node account: ``cargo 1.98.1 (797e8a9bc 2026-08-05)``, ``rustup 1.29.1``,
``cargo-llvm-cov 0.9.1`` and ``uv 0.13.0``, the toolchain every passing Rust
check on linux has run on, so a provisioned worker answers the probe's
``cargo`` line with the version diphtheria's machine file declares.

THE INSTALLER IS PINNED BY ITS DIGEST. rustup-init is fetched from the
release's archive path, never ``sh.rustup.rs``'s newest, and refused unless
its SHA-256 is :data:`RUSTUP_INIT_SHA256`, the digest the archive published
beside it, read on 2026-10-23, so a later change to the file at that path is
a refusal rather than a different installer.
"""

from __future__ import annotations

from typing import Final

#: The Rust toolchain rustup installs and makes the account's default.
PINNED_RUST: Final = "1.98.1"

#: The rustup release whose installer is run.
PINNED_RUSTUP: Final = "1.29.1"

#: The one target the installer is fetched for: every linux worker is x86_64.
RUSTUP_TARGET: Final = "x86_64-unknown-linux-gnu"

#: The SHA-256 of that release's rustup-init for :data:`RUSTUP_TARGET`.
RUSTUP_INIT_SHA256: Final = "dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71"

#: The rustup components the Rust check's gate runs, beside the compiler:
#: ``cargo fmt``, ``cargo clippy``, and the LLVM tools ``cargo llvm-cov``
#: reads its coverage with.
RUST_COMPONENTS: Final[tuple[str, ...]] = ("rustfmt", "clippy", "llvm-tools-preview")

#: The cargo-llvm-cov ``cargo install --locked`` builds.
PINNED_CARGO_LLVM_COV: Final = "0.9.1"

#: The uv the Rust check's Python half creates its venv with.
PINNED_UV: Final = "0.13.0"


__all__ = [
    "PINNED_CARGO_LLVM_COV",
    "PINNED_RUST",
    "PINNED_RUSTUP",
    "PINNED_UV",
    "RUSTUP_INIT_SHA256",
    "RUSTUP_TARGET",
    "RUST_COMPONENTS",
]
