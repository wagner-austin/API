from __future__ import annotations

from typing import Protocol

from model_trainer.core.encoding import HandleEncoder, ListEncoded


class _HandleProto(Protocol):
    def encode(self: _HandleProto, text: str) -> list[int]: ...
    def decode(self: _HandleProto, ids: list[int]) -> str: ...
    def token_to_id(self: _HandleProto, token: str) -> int | None: ...
    def get_vocab_size(self: _HandleProto) -> int: ...


class _FakeHandle:
    def __init__(self: _FakeHandle) -> None:
        self._vocab: dict[str, int] = {"[PAD]": 0, "A": 1, "B": 2}
        self._rev: dict[int, str] = {v: k for k, v in self._vocab.items()}

    def encode(self: _FakeHandle, text: str) -> list[int]:
        unk = -1
        return [self._vocab.get(tok, unk) for tok in text.split()]

    def decode(self: _FakeHandle, ids: list[int]) -> str:
        return " ".join(self._rev.get(i, "[UNK]") for i in ids)

    def token_to_id(self: _FakeHandle, token: str) -> int | None:
        return self._vocab.get(token)

    def get_vocab_size(self: _FakeHandle) -> int:
        return len(self._vocab)


def test_handle_encoder_encode_wraps_in_listencoded_and_copies() -> None:
    h = _FakeHandle()
    enc = HandleEncoder(h)
    res = enc.encode("A Z")
    assert isinstance(res, ListEncoded) and res.ids == [1, -1]
    # mutating returned ids should not affect subsequent encodes
    res.ids[0] = 99
    again = enc.encode("A Z")
    assert again.ids == [1, -1]


def test_handle_encoder_token_info_and_decode_and_vocab_size() -> None:
    h = _FakeHandle()
    enc = HandleEncoder(h)
    assert enc.token_to_id("A") == 1
    assert enc.get_vocab_size() == 3
    assert enc.decode([1, 2, 0, -1]) == "A B [PAD] [UNK]"
