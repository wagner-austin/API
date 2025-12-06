"""Shared test fixtures and configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

# Fixtures directory location
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_D_DIR = FIXTURES_DIR / "sample.D"


def _get_sample_d_directory() -> Path:
    """Get path to sample.D fixture, skip if not found."""
    if not SAMPLE_D_DIR.exists():
        pytest.skip("sample.D test fixture not found")
    return SAMPLE_D_DIR


def _get_fixtures_dir() -> Path:
    """Get path to fixtures directory."""
    if not FIXTURES_DIR.exists():
        pytest.skip("fixtures directory not found")
    return FIXTURES_DIR


sample_d_directory = pytest.fixture(_get_sample_d_directory)
test_data_root = pytest.fixture(_get_fixtures_dir)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring external data",
    )
