from __future__ import annotations

from pathlib import Path

from model_trainer.core.services.training.dataset_builder import CausalLMDataset


class _Enc:
    def __init__(self: _Enc, ids: list[int]) -> None:
        self._ids = ids

    @property
    def ids(self: _Enc) -> list[int]:
        return self._ids


class _FakeTok:
    def encode(self: _FakeTok, text: str) -> _Enc:
        ids = [ord(c) % 20 for c in text]
        return _Enc(ids)

    def token_to_id(self: _FakeTok, token: str) -> int | None:
        return None

    def get_vocab_size(self: _FakeTok) -> int:
        return 256

    def decode(self: _FakeTok, ids: list[int]) -> str:
        # Simple round-trip for protocol completeness (not used by dataset)
        return "".join(chr(i) for i in ids)


def test_dataset_chunks_with_eos_and_pad(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abc\ndefg\n", encoding="utf-8")
    files = [str(corpus / "a.txt")]
    tok = _FakeTok()
    ds = CausalLMDataset(files=files, tokenizer=tok, max_len=5, eos_id=99, pad_id=0)
    assert len(ds) >= 1 and len(ds) == 2
    first = ds[0]
    assert first.shape[0] == 5
    # ensure EOS present or padding used to reach max_len (indexing to avoid untyped iteration)
    n = int(first.shape[0])
    vals: list[int] = []
    for i in range(n):
        # Each element is a scalar tensor; convert to Python int
        vals.append(int(first[i].item()))
    assert any(v == 99 or v == 0 for v in vals)
