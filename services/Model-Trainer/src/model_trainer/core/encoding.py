from __future__ import annotations

from typing import Protocol

from model_trainer.core.contracts.tokenizer import TokenizerHandle


class Encoded(Protocol):
    @property
    def ids(self: Encoded) -> list[int]: ...


class Encoder(Protocol):
    def encode(self: Encoder, text: str) -> Encoded: ...
    def token_to_id(self: Encoder, token: str) -> int | None: ...
    def get_vocab_size(self: Encoder) -> int: ...
    def decode(self: Encoder, ids: list[int]) -> str: ...


class ListEncoded:
    def __init__(self: ListEncoded, ids: list[int]) -> None:
        self._ids = ids

    @property
    def ids(self: ListEncoded) -> list[int]:
        return self._ids


class HandleEncoder:
    """Adapter from TokenizerHandle-like objects to Encoder.

    Expects an object with methods:
      - encode(text) -> list[int]
      - decode(ids) -> str
      - token_to_id(token) -> int | None
      - get_vocab_size() -> int
    """

    def __init__(self: HandleEncoder, handle: TokenizerHandle) -> None:
        self._h: TokenizerHandle = handle

    def encode(self: HandleEncoder, text: str) -> ListEncoded:
        ids = self._h.encode(text)
        return ListEncoded(list(ids))

    def decode(self: HandleEncoder, ids: list[int]) -> str:
        return self._h.decode(ids)

    def token_to_id(self: HandleEncoder, token: str) -> int | None:
        return self._h.token_to_id(token)

    def get_vocab_size(self: HandleEncoder) -> int:
        return int(self._h.get_vocab_size())


__all__ = [
    "Encoded",
    "Encoder",
    "HandleEncoder",
    "ListEncoded",
]
