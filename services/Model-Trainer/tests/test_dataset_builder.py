from __future__ import annotations

import torch

from model_trainer.core.services.training.dataset_builder import IGNORE_INDEX, CausalLMDataset


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


def test_dataset_chunks_with_eos_and_pad() -> None:
    ds = CausalLMDataset(
        lines=("abc", "defg"), tokenizer=_FakeTok(), max_len=5, eos_id=99, pad_id=0
    )
    assert len(ds) == 2
    first, labels = ds[0]
    assert first.shape[0] == 5
    assert labels.shape[0] == 5
    # ensure EOS present or padding used to reach max_len (indexing to avoid untyped iteration)
    n = int(first.shape[0])
    vals: list[int] = []
    for i in range(n):
        # Each element is a scalar tensor; convert to Python int
        vals.append(int(first[i].item()))
    assert any(v == 99 or v == 0 for v in vals)


def test_labels_match_inputs_when_no_separator_is_configured() -> None:
    """Without masking the dataset must behave exactly as before: every token
    is a loss target."""
    ds = CausalLMDataset(lines=("abcdefgh",), tokenizer=_FakeTok(), max_len=4, eos_id=99, pad_id=0)
    inputs, labels = ds[0]
    assert torch.equal(inputs, labels)


def test_prefix_tokens_are_excluded_from_the_loss() -> None:
    """The marker is fed as context but must not be a prediction target.

    The fake tokenizer emits one id per character, so the masked span is
    exactly ``len("hub | ")`` positions and the assertion can be exact.
    """
    ds = CausalLMDataset(
        lines=("hub | body text here",),
        tokenizer=_FakeTok(),
        max_len=64,
        eos_id=99,
        pad_id=0,
        loss_mask_prefix_separator=" | ",
    )
    inputs, labels = ds[0]
    prefix_len = len("hub | ")
    for i in range(prefix_len):
        assert int(labels[i].item()) == IGNORE_INDEX
    # The body is still a target, and the inputs kept the marker verbatim.
    assert int(labels[prefix_len].item()) == int(inputs[prefix_len].item())
    assert int(labels[prefix_len].item()) != IGNORE_INDEX


def test_a_line_without_the_separator_is_left_unmasked() -> None:
    """Diluted corpora mix marked wiki paragraphs with unmarked junk lines, so
    an absent separator must mean 'no prefix', not an error."""
    ds = CausalLMDataset(
        lines=("no marker on this line",),
        tokenizer=_FakeTok(),
        max_len=64,
        eos_id=99,
        pad_id=0,
        loss_mask_prefix_separator=" | ",
    )
    inputs, labels = ds[0]
    text_len = len("no marker on this line")
    for i in range(text_len):
        assert int(labels[i].item()) == int(inputs[i].item())


def test_padding_is_excluded_from_the_loss() -> None:
    """Only the final chunk is padded, and filler must not be a target."""
    ds = CausalLMDataset(lines=("ab",), tokenizer=_FakeTok(), max_len=8, eos_id=99, pad_id=0)
    inputs, labels = ds[0]
    # "ab" -> 2 ids + eos = 3 real tokens, so positions 3..7 are padding.
    for i in range(3, 8):
        assert int(inputs[i].item()) == 0
        assert int(labels[i].item()) == IGNORE_INDEX


def test_dataset_tokenization_progress_logs_periodically_not_per_line() -> None:
    """Progress logging fires on an interval, so both of its branches run.

    125 lines gives ``log_interval = max(1, 125 // 10) = 12``. Lines 12, 24 ...
    120 log and the rest do not, which exercises the interval check in both
    directions within one dataset.
    """
    lines = tuple(f"line {i}" for i in range(125))
    ds = CausalLMDataset(lines=lines, tokenizer=_FakeTok(), max_len=10, eos_id=99, pad_id=0)
    chunk, chunk_labels = ds[0]
    assert chunk.shape[0] == 10
    assert chunk_labels.shape[0] == 10
    assert len(ds) >= 10
