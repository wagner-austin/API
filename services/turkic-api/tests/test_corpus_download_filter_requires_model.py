from __future__ import annotations

import pytest

from turkic_api.core.corpus_download import ensure_corpus_file
from turkic_api.core.models import Language, ProcessSpec, Source


def test_ensure_corpus_file_requires_model_when_filtering(tmp_path: str) -> None:
    spec = ProcessSpec(
        source=Source.OSCAR,
        language=Language.KK,
        max_sentences=1,
        transliterate=False,
        confidence_threshold=0.5,
    )
    with pytest.raises(ValueError, match="langid_model is required"):
        ensure_corpus_file(spec, tmp_path, script=None, langid_model=None)
