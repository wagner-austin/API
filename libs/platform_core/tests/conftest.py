"""Shared test fixtures for platform_core tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from scripts import guard as guard_mod

from platform_core import json_utils as json_utils_mod
from platform_core import torch_types as torch_types_mod
from platform_core.config import _test_hooks


@pytest.fixture(autouse=True)
def _restore_config_hooks() -> Generator[None, None, None]:
    """Restore config hooks after each test."""
    original_get_env = _test_hooks.get_env
    original_tomllib_loads = _test_hooks.tomllib_loads
    yield
    _test_hooks.get_env = original_get_env
    _test_hooks.tomllib_loads = original_tomllib_loads


@pytest.fixture(autouse=True)
def _restore_json_utils_hooks() -> Generator[None, None, None]:
    """Restore json_utils hooks after each test."""
    original_json_loads = json_utils_mod._json_loads
    yield
    json_utils_mod._json_loads = original_json_loads


@pytest.fixture(autouse=True)
def _restore_guard_hooks() -> Generator[None, None, None]:
    """Restore guard hooks after each test."""
    original_is_dir = guard_mod._is_dir
    yield
    guard_mod._is_dir = original_is_dir


@pytest.fixture(autouse=True)
def _restore_torch_types_hooks() -> Generator[None, None, None]:
    """Restore torch_types hooks after each test."""
    original_import_torch = torch_types_mod._import_torch
    yield
    torch_types_mod._import_torch = original_import_torch
