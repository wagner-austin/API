"""The toolchain pins' committed JSON: current and the contract's (MCPs board task ce698bfb).

MCPs' host provisioning installs the Poetry this copy names on the session
host and on diphtheria, and the Rust toolchain it names on every linux fleet
worker (MCPs board task 91bb5cda), so a pin moved in the contract without
``make render-documents`` would install the old version: the first class
fails until the copy is the render.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import load_json_str

from fleet.contracts.capability import RUST_CHECK_SUBCOMMANDS, RUST_CHECK_TOOLS
from fleet.contracts.rust_toolchain import (
    PINNED_CARGO_LLVM_COV,
    PINNED_RUST,
    PINNED_UV,
    RUST_COMPONENTS,
)
from fleet.contracts.toolchain import PINNED_POETRY
from fleet.core.rendered_powershell import RENDERED_DIRECTORY
from fleet.core.toolchain_document import TOOLCHAIN_DOCUMENT, render_toolchain_document

#: tools/fleet.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The committed copy.
_COMMITTED = _PACKAGE / RENDERED_DIRECTORY / TOOLCHAIN_DOCUMENT


class TestTheCommittedCopy:
    def test_the_committed_copy_is_the_render(self) -> None:
        assert _COMMITTED.read_text(encoding="utf-8") == render_toolchain_document(), (
            "run `make render-documents` in tools/fleet"
        )


class TestTheDocument:
    def test_it_names_every_pinned_tool_at_its_version(self) -> None:
        assert load_json_str(render_toolchain_document()) == {
            "poetry": {"version": "2.4.2"},
            "rust": {
                "version": "1.98.1",
                "components": ["rustfmt", "clippy", "llvm-tools-preview"],
            },
            "rustup": {
                "version": "1.29.1",
                "target": "x86_64-unknown-linux-gnu",
                "sha256": "dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71",
            },
            "cargo-llvm-cov": {"version": "0.9.1"},
            "uv": {"version": "0.13.0"},
        }
        assert PINNED_POETRY == "2.4.2"
        assert PINNED_RUST == "1.98.1"

    def test_the_rust_pins_give_every_tool_the_rust_probe_asks_for(self) -> None:
        # A worker given these pins must answer the probe's cargo line (MCPs
        # board task b1089284): each subcommand it asks is served by a pinned
        # component, llvm-cov's by the LLVM tools beside the cargo-llvm-cov
        # pin, and each tool it asks beside cargo has a pin of its own.
        serves = {"fmt": "rustfmt", "clippy": "clippy", "llvm-cov": "llvm-tools-preview"}
        assert sorted(serves) == sorted(RUST_CHECK_SUBCOMMANDS)
        assert [serves[subcommand] for subcommand in RUST_CHECK_SUBCOMMANDS] == list(
            RUST_COMPONENTS
        )
        assert PINNED_CARGO_LLVM_COV == "0.9.1"
        assert RUST_CHECK_TOOLS == ("uv",)
        assert PINNED_UV == "0.13.0"

    def test_it_ends_in_one_newline(self) -> None:
        text = render_toolchain_document()
        assert text.endswith("}\n")
        assert not text.endswith("\n\n")
