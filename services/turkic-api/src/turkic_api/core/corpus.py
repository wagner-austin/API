from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from turkic_api.core.models import ProcessSpec


class CorpusService:
    """Interface for fetching corpus sentences for processing."""

    def stream(self, spec: ProcessSpec) -> Generator[str, None, None]:
        raise NotImplementedError


class LocalCorpusService(CorpusService):
    """Reads corpus lines from data_dir/corpus/{source}_{language}.txt."""

    def __init__(self, data_dir: str) -> None:
        self._root = Path(data_dir) / "corpus"

    def stream(self, spec: ProcessSpec) -> Generator[str, None, None]:
        path = self._root / f"{spec['source']}_{spec['language']}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                yield s
                count += 1
                if count >= spec["max_sentences"]:
                    break
