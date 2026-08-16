from __future__ import annotations

from pathlib import Path

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


def test_dataset_chunks_with_eos_and_pad(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abc\ndefg\n", encoding="utf-8")
    files = [str(corpus / "a.txt")]
    tok = _FakeTok()
    ds = CausalLMDataset(files=files, tokenizer=tok, max_len=5, eos_id=99, pad_id=0)
    assert len(ds) >= 1 and len(ds) == 2
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


def test_labels_match_inputs_when_no_separator_is_configured(tmp_path: Path) -> None:
    """Without masking the dataset must behave exactly as before: every token
    is a loss target."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("abcdefgh\n", encoding="utf-8")
    ds = CausalLMDataset(
        files=[str(corpus / "a.txt")], tokenizer=_FakeTok(), max_len=4, eos_id=99, pad_id=0
    )
    inputs, labels = ds[0]
    assert torch.equal(inputs, labels)


def test_prefix_tokens_are_excluded_from_the_loss(tmp_path: Path) -> None:
    """The marker is fed as context but must not be a prediction target.

    The fake tokenizer emits one id per character, so the masked span is
    exactly ``len("hub | ")`` positions and the assertion can be exact.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hub | body text here\n", encoding="utf-8")
    ds = CausalLMDataset(
        files=[str(corpus / "a.txt")],
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


def test_a_line_without_the_separator_is_left_unmasked(tmp_path: Path) -> None:
    """Diluted corpora mix marked wiki paragraphs with unmarked junk lines, so
    an absent separator must mean 'no prefix', not an error."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("no marker on this line\n", encoding="utf-8")
    ds = CausalLMDataset(
        files=[str(corpus / "a.txt")],
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


def test_padding_is_excluded_from_the_loss(tmp_path: Path) -> None:
    """Only the final chunk is padded, and filler must not be a target."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("ab\n", encoding="utf-8")
    ds = CausalLMDataset(
        files=[str(corpus / "a.txt")], tokenizer=_FakeTok(), max_len=8, eos_id=99, pad_id=0
    )
    inputs, labels = ds[0]
    # "ab" -> 2 ids + eos = 3 real tokens, so positions 3..7 are padding.
    for i in range(3, 8):
        assert int(inputs[i].item()) == 0
        assert int(labels[i].item()) == IGNORE_INDEX


def test_dataset_tokenization_progress_with_multiple_files(tmp_path: Path) -> None:
    """Test that tokenization progress is logged for multiple files.

    Uses 25 files so log_interval = max(1, min(10, 25//10)) = 2, which ensures
    some files skip logging (covering the False branch of the progress check).
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Create 25 files: log_interval = max(1, min(10, 25//10)) = 2
    # This ensures both True and False branches of progress logging are hit
    for i in range(25):
        (corpus / f"file_{i:02d}.txt").write_text(f"line {i}\n" * 5, encoding="utf-8")
    files = sorted([str(f) for f in corpus.iterdir()])
    tok = _FakeTok()
    ds = CausalLMDataset(files=files, tokenizer=tok, max_len=10, eos_id=99, pad_id=0)
    # Verify dataset was built correctly - accessing first chunk validates non-empty
    chunk, chunk_labels = ds[0]
    assert chunk.shape[0] == 10
    assert chunk_labels.shape[0] == 10
    # Dataset should have multiple chunks given 25 files with 5 lines each
    assert len(ds) >= 10
