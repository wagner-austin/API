"""Tests for classifier backend registry and capabilities.

Covers default registry composition and non-torch paths to keep strict typing
without importing heavy frameworks. Does not execute training loops here.
"""

from __future__ import annotations

from covenant_ml.backends import default_registry


def test_default_registry_lists_backends() -> None:
    """Default registry exposes both xgboost and mlp backends."""
    reg = default_registry()
    names = reg.list_backends()
    assert "xgboost" in names
    assert "mlp" in names


def test_capabilities_present_for_each_backend() -> None:
    """Capabilities are cached and include required keys."""
    reg = default_registry()
    for name in ("xgboost", "mlp"):
        caps = reg.get_capabilities(name)
        assert caps["supports_train"] is True
        assert caps["supports_early_stopping"] in (True, False)
        assert caps["supports_gpu"] in (True, False)
        assert type(caps["model_format"]) is str


def test_registry_get_returns_backend_instance() -> None:
    """Get method returns a backend instance with correct name."""
    reg = default_registry()
    xgb = reg.get("xgboost")
    assert xgb.backend_name() == "xgboost"
    mlp = reg.get("mlp")
    assert mlp.backend_name() == "mlp"


def test_backend_registration_factory_returns_callable() -> None:
    """BackendRegistration.factory() returns the factory callable."""
    from covenant_ml.backends.registry import BackendRegistration
    from covenant_ml.backends.xgboost_backend import create_xgboost_backend

    registration = BackendRegistration(create_xgboost_backend)
    factory = registration.factory()
    assert callable(factory)
    backend = factory()
    assert backend.backend_name() == "xgboost"


def test_capabilities_caching() -> None:
    """Capabilities are cached after first access."""
    from covenant_ml.backends.registry import BackendRegistration
    from covenant_ml.backends.xgboost_backend import create_xgboost_backend

    registration = BackendRegistration(create_xgboost_backend)
    # First access populates cache
    caps1 = registration.capabilities()
    # Second access returns cached
    caps2 = registration.capabilities()
    assert caps1 == caps2
    assert caps1["supports_train"] is True
