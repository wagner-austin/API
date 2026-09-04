"""The boundary between a HuggingFace tokenizer and this codebase's Encoder.

Every method forwards, so what is worth asserting is that each forwards to the
RIGHT method: HuggingFace spells the vocabulary size ``__len__`` and the token
lookup ``convert_tokens_to_ids``, and the ``Encoder`` protocol spells them
``get_vocab_size`` and ``token_to_id``. A wrong pairing here type-checks and
returns a plausible number.
"""

from __future__ import annotations

from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder


class _Tokenizer:
    """A tokenizer with the surface HFTokenizerProto declares.

    Each method returns something identifiable rather than realistic, so a
    test can tell which one the adapter called.
    """

    @property
    def eos_token_id(self) -> int | None:
        return 50256

    @property
    def pad_token_id(self) -> int | None:
        return None

    def __len__(self) -> int:
        return 50257

    def encode(self, text: str) -> list[int]:
        return [len(text), 7]

    def decode(self, ids: list[int]) -> str:
        return "|".join(str(value) for value in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 1000 + len(token)


class TestHFTokenizerEncoder:
    def test_encode_returns_the_tokenizer_s_ids(self) -> None:
        assert HFTokenizerEncoder(_Tokenizer()).encode("abcd").ids == [4, 7]

    def test_encode_copies_rather_than_aliasing(self) -> None:
        """The scorer slices the ids; a shared list would let it edit the source."""
        tokenizer = _Tokenizer()
        encoder = HFTokenizerEncoder(tokenizer)

        first = encoder.encode("abcd").ids
        first.append(999)

        assert encoder.encode("abcd").ids == [4, 7]

    def test_decode_forwards_to_the_tokenizer(self) -> None:
        assert HFTokenizerEncoder(_Tokenizer()).decode([1, 2, 3]) == "1|2|3"

    def test_vocabulary_size_comes_from_len_not_a_method_named_for_it(self) -> None:
        """HuggingFace has no ``get_vocab_size``; it answers ``len``."""
        assert HFTokenizerEncoder(_Tokenizer()).get_vocab_size() == 50257

    def test_token_lookup_forwards_to_convert_tokens_to_ids(self) -> None:
        """And returns what the tokenizer said, including for absent tokens.

        The protocol allows None so an implementation can report a token it
        does not have. HuggingFace reports an absent token by returning the
        unknown-token id, which is a real id; manufacturing a None would mean
        guessing which id means "unknown". The adapter forwards instead, and
        its docstring says so.
        """
        assert HFTokenizerEncoder(_Tokenizer()).token_to_id("ab") == 1002

    def test_it_round_trips_through_the_encoder_protocol(self) -> None:
        encoder = HFTokenizerEncoder(_Tokenizer())

        assert encoder.decode(encoder.encode("xy").ids) == "2|7"
