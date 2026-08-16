"""Tests for loss_mask_prefix_separator validation in runs.py.

The separator decides which tokens are excluded from the training loss, so a
value that reaches the worker unchecked would silently change what a run
optimises. Every rejection branch is exercised against the real decoder.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def _base_gpt2_payload() -> dict[str, JSONValue]:
    """Return base payload for gpt2 tests."""
    return {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "corpus_file_id": "cid",
        "tokenizer_id": "tok",
        "user_id": 0,
    }


class TestLossMaskPrefixSeparator:
    """Tests for the marker separator that splits a masked prefix from a body."""

    def test_absent_means_no_masking(self) -> None:
        """Omitting the field must leave every token a loss target."""
        out = _decode_train_request(_base_gpt2_payload())
        assert out["loss_mask_prefix_separator"] is None

    def test_explicit_null_means_no_masking(self) -> None:
        payload = _base_gpt2_payload()
        payload["loss_mask_prefix_separator"] = None
        out = _decode_train_request(payload)
        assert out["loss_mask_prefix_separator"] is None

    def test_separator_is_carried_through_verbatim(self) -> None:
        """Whitespace is significant: ' | ' and '|' split different spans."""
        payload = _base_gpt2_payload()
        payload["loss_mask_prefix_separator"] = " | "
        out = _decode_train_request(payload)
        assert out["loss_mask_prefix_separator"] == " | "

    def test_non_string_is_rejected(self) -> None:
        payload = _base_gpt2_payload()
        payload["loss_mask_prefix_separator"] = 7
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        err: AppError[ErrorCode] = excinfo.value
        assert err.code == ErrorCode.INVALID_INPUT
        assert "must be a string" in err.message

    def test_empty_string_is_rejected_rather_than_treated_as_absent(self) -> None:
        """An empty separator would mask nothing while the manifest recorded
        that masking was requested, so the arm would be mislabelled."""
        payload = _base_gpt2_payload()
        payload["loss_mask_prefix_separator"] = ""
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        err: AppError[ErrorCode] = excinfo.value
        assert err.code == ErrorCode.INVALID_INPUT
        assert err.http_status == 422
