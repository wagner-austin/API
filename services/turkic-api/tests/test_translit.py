import pathlib
import unicodedata

from turkic_api.core.translit import to_latin

ROOT = pathlib.Path(__file__).parent


def test_roundtrip_and_nfc() -> None:
    with open(ROOT / "sample_cy.txt", encoding="utf8") as f:
        src = f.read()
    out = "\n".join(to_latin(line, "kk") for line in src.splitlines())
    assert unicodedata.is_normalized("NFC", out)
    with open(ROOT / "expected_lat.txt", encoding="utf8") as f:
        exp = f.read()
    assert out.splitlines()[0] == exp.splitlines()[0]  # quick sanity


def test_byte_reduction() -> None:
    src = (ROOT / "sample_cy.txt").read_bytes()
    out = to_latin(src.decode("utf8"), "kk").encode("utf8")
    assert len(out) < len(src)
