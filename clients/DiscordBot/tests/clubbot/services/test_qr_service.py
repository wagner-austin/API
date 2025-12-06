from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageColor
from PIL.Image import Image as PILImage
from tests.conftest import _build_settings

from clubbot.config import DiscordbotSettings
from clubbot.services.qr.client import QRRequestPayload, QRService


def _make_png(box: int, border: int, back: str) -> bytes:
    modules = 10
    size = (modules + border * 2) * box
    img = Image.new("RGB", (size, size), color=ImageColor.getrgb(back))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class _FakeClient:
    def __init__(self, *, back: str, box: int, border: int) -> None:
        self._back = back
        self._box = box
        self._border = border

    def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes:
        return _make_png(self._box, self._border, self._back)


def _cfg() -> DiscordbotSettings:
    return _build_settings(
        qr_default_box_size=10,
        qr_default_border=4,
        qr_default_fill_color="#000000",
        qr_default_back_color="#FFFFFF",
        qr_public_responses=True,
        qr_api_url="http://localhost:8080",
    )


def test_generate_qr_png_basic() -> None:
    cfg = _cfg()
    svc = QRService(cfg, client=_FakeClient(back="#FFFFFF", box=10, border=4))
    out = svc.generate_qr("https://example.com")
    png = out.image_png
    assert type(png) is bytes
    img: PILImage
    with Image.open(BytesIO(png)) as img:
        assert img.format == "PNG"
        w, h = img.size
        assert w > 100 and h > 100
        pixel_value: int | tuple[int, ...] = img.getpixel((0, 0))
        expected_color: tuple[int, int, int] | tuple[int, int, int, int] = ImageColor.getrgb(
            "#FFFFFF"
        )
        assert pixel_value == expected_color


def test_generate_qr_png_background_and_grid_alignment() -> None:
    cfg = _cfg()
    svc = QRService(cfg, client=_FakeClient(back="#FEFEFE", box=8, border=3))
    payload: QRRequestPayload = {
        "url": "https://openai.com",
        "ecc": "H",
        "box_size": 8,
        "border": 3,
        "fill_color": "#112233",
        "back_color": "#FEFEFE",
    }
    out = svc.generate_qr_with_payload(payload)
    img: PILImage
    with Image.open(BytesIO(out.image_png)) as img:
        w, h = img.size
        assert w % 8 == 0 and h % 8 == 0
        pixel_value: int | tuple[int, ...] = img.getpixel((0, 0))
        expected_color: tuple[int, int, int] | tuple[int, int, int, int] = ImageColor.getrgb(
            "#FEFEFE"
        )
        assert pixel_value == expected_color
