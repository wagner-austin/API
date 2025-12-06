from __future__ import annotations

from types import MappingProxyType

import pytest
from platform_core.json_utils import JSONValue

import handwriting_ai.jobs.digits as dj

pytestmark = pytest.mark.usefixtures("digits_redis")


def test_process_train_job_invokes_decoder_with_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"ok": False, "is_dict": False}

    def _stub(p: dict[str, JSONValue]) -> None:
        called["ok"] = True
        called["is_dict"] = isinstance(p, dict)

    monkeypatch.setattr(dj, "_decode_and_process_train_job", _stub, raising=True)

    payload_map = MappingProxyType(
        {
            "type": "digits.train.v1",
            "request_id": "r",
            "user_id": 1,
            "model_id": "m",
            "epochs": 1,
            "batch_size": 1,
            "lr": 0.01,
            "seed": 1,
            "augment": False,
            "notes": None,
        }
    )

    dj.process_train_job(payload_map)

    assert called["ok"] is True and called["is_dict"] is True
