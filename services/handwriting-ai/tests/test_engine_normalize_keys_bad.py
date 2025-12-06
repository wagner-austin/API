from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pytest
from platform_core.json_utils import JSONValue


class _NormalizeFn(Protocol):
    def __call__(self, keys: Sequence[JSONValue], label: str) -> tuple[str, ...]: ...


def test_normalize_keys_raises_on_non_string() -> None:
    mod = __import__("handwriting_ai.inference.engine", fromlist=["_normalize_keys"])
    normalize: _NormalizeFn = mod._normalize_keys
    with pytest.raises(RuntimeError, match="missing_keys entry is not a string"):
        _ = normalize(["ok", 1], "missing_keys")
