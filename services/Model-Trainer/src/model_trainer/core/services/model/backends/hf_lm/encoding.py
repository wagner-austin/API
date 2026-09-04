"""Adapt a HuggingFace tokenizer to this codebase's :class:`Encoder`.

WHY AN ADAPTER RATHER THAN REUSE. ``HandleEncoder`` already turns a
:class:`~model_trainer.core.contracts.tokenizer.TokenizerHandle` into an
``Encoder``, and it is the right thing to use when there is a handle. There is
not one here: ``TokenizerHandle`` wants ``token_to_id`` and
``get_vocab_size``, a HuggingFace tokenizer spells those
``convert_tokens_to_ids`` and ``__len__``, and the loader that produces
handles reads a trained tokenizer's artifact DIRECTORY rather than a hub id.
Scoring gpt2 needs gpt2's own BPE, which only the hub path supplies.

So this is boundary translation between two type systems, which is the one
shape a wrapper is allowed to be. It adds no behaviour and hides no failure;
every method below forwards to the tokenizer's own.
"""

from __future__ import annotations

from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto


class HFTokenizerEncoder:
    """An :class:`Encoder` backed by a HuggingFace tokenizer.

    Attributes:
        _tokenizer: The tokenizer every method forwards to.
    """

    _tokenizer: HFTokenizerProto

    def __init__(self, tokenizer: HFTokenizerProto) -> None:
        """Wrap a loaded tokenizer.

        Args:
            tokenizer: The tokenizer to adapt.
        """
        self._tokenizer = tokenizer

    def encode(self, text: str) -> ListEncoded:
        """Encode text to token ids.

        Args:
            text: Text to encode.

        Returns:
            The ids, in the shape the scorer reads.
        """
        return ListEncoded(list(self._tokenizer.encode(text)))

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text.

        Args:
            ids: Ids to decode.

        Returns:
            The decoded text.
        """
        return self._tokenizer.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        """Look up one token's id.

        NEVER RETURNS NONE, AND THAT IS A PROPERTY OF THE TOKENIZER RATHER
        THAN A SHORTCUT HERE. ``Encoder`` declares ``int | None`` so an
        implementation can report a token it does not have;
        ``convert_tokens_to_ids`` reports an absent token by returning the
        vocabulary's unknown-token id, which is a real id and indistinguishable
        from a hit. Manufacturing a None would require guessing which id means
        "unknown", so the honest translation forwards what the tokenizer said.

        A caller that must distinguish absence cannot use this adapter. No
        caller in this codebase does: the cloze scorer and the answer-span
        measurement reach only for :meth:`encode` and :meth:`decode`.

        Args:
            token: The token to look up.

        Returns:
            The tokenizer's id for it.
        """
        return self._tokenizer.convert_tokens_to_ids(token)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size.

        Args:
            None.

        Returns:
            The number of tokens the tokenizer knows, which is what ``len``
            reports for a HuggingFace tokenizer.
        """
        return len(self._tokenizer)


__all__ = ["HFTokenizerEncoder"]
