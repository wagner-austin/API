from __future__ import annotations

from platform_core.errors import HandwritingErrorCode, handwriting_status_for


def test_status_for_preprocessing_failed() -> None:
    assert int(handwriting_status_for(HandwritingErrorCode.preprocessing_failed)) == 400


def test_status_for_invalid_model() -> None:
    assert int(handwriting_status_for(HandwritingErrorCode.invalid_model)) == 400
