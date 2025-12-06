from __future__ import annotations


def test_asgi_app_builds() -> None:
    from qr_api.asgi import app as asgi_app

    # Verify app has title attribute by accessing it directly
    assert asgi_app.title == "qr-api"
