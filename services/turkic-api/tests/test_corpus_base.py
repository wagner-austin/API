from __future__ import annotations

import pytest

from turkic_api.core.corpus import CorpusService
from turkic_api.core.models import ProcessSpec


def test_corpus_service_stream_not_implemented() -> None:
    spec: ProcessSpec = {
        "source": "oscar",
        "language": "kk",
        "max_sentences": 1,
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    svc = CorpusService()
    with pytest.raises(NotImplementedError):
        svc.stream(spec)
