from __future__ import annotations

from collections.abc import Generator
from types import ModuleType

import pytest

import qr_api.generator as gen
from qr_api.generator import _load_segno_module, generate_png
from qr_api.types import QROptions


@pytest.fixture(autouse=True)
def _restore_import_hook() -> Generator[None, None, None]:
    """Restore the import hook after each test."""
    original = gen._import_module
    yield
    gen._import_module = original


def test_generate_png_with_three_digit_hex() -> None:
    opts = QROptions(
        url="https://example.com",
        ecc="M",
        box_size=5,
        border=2,
        fill_color="#0f0",
        back_color="#fff",
    )
    out = generate_png(opts)
    assert type(out) is bytes
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_load_segno_module_requires_make() -> None:
    module_stub = ModuleType("segno_stub")

    def _import_stub(name: str) -> ModuleType:
        return module_stub

    gen._import_module = _import_stub
    with pytest.raises(RuntimeError):
        _load_segno_module()
