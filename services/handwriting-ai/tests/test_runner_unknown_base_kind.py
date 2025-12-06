from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pytest
from platform_core.json_utils import JSONValue


class _BuildFn(Protocol):
    def __call__(self, spec: Mapping[str, JSONValue]) -> JSONValue: ...


def test_build_dataset_from_spec_unknown_base_kind_branch() -> None:
    mod = __import__(
        "handwriting_ai.training.calibration.runner",
        fromlist=["_build_dataset_from_spec"],
    )
    fn: _BuildFn = mod._build_dataset_from_spec
    spec: dict[str, JSONValue] = {
        "base_kind": "weird",
        "mnist": None,
        "inline": None,
        "augment": {},
    }
    with pytest.raises(RuntimeError, match="unknown base_kind"):
        _ = fn(spec)
