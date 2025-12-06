from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from platform_core.errors import AppError

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import GenerateConfig, PreparedLMModel, ScoreConfig
from model_trainer.core.services.model.backends.gpt2.generate import (
    _check_stop_sequence,
    _read_prompt,
    generate_gpt2,
)
from model_trainer.core.services.model.backends.gpt2.score import (
    _compute_topk,
    _get_logits_and_loss,
    _read_text_or_path,
    score_gpt2,
)
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)


class _Enc:
    def __init__(self: _Enc, ids: list[int]) -> None:
        self._ids = ids

    @property
    def ids(self: _Enc) -> list[int]:
        return self._ids


class _Tok4DS:
    def encode(self: _Tok4DS, text: str) -> _Enc:
        return _Enc([ord(c) % 10 for c in text])

    def token_to_id(self: _Tok4DS, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _Tok4DS) -> int:
        return 10

    def decode(self: _Tok4DS, ids: list[int]) -> str:
        return "".join(chr(48 + i) for i in ids)


class _LMConfig(ConfigLike):
    n_positions = 16


class _Out(ForwardOutProto):
    def __init__(self: _Out, logits: torch.Tensor) -> None:
        self._logits = logits

    @property
    def loss(self: _Out) -> torch.Tensor:
        return torch.tensor(0.5, requires_grad=True)

    @property
    def logits(self: _Out) -> torch.Tensor:
        return self._logits


class _FakeLM(LMModelProto):
    def __init__(self: _FakeLM, vocab_size: int = 10) -> None:
        self._vocab_size = vocab_size
        self._param = torch.nn.Parameter(torch.zeros(1))

    def train(self: _FakeLM) -> None:
        return None

    def eval(self: _FakeLM) -> None:
        return None

    def forward(self: _FakeLM, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        batch_size = int(input_ids.size(0))
        seq_len = int(input_ids.size(1))
        logits = torch.randn(batch_size, seq_len, self._vocab_size)
        return _Out(logits)

    def parameters(self: _FakeLM) -> Sequence[ParameterLike]:
        return [self._param]

    def named_parameters(self: _FakeLM) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _FakeLM, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _FakeLM, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    @property
    def config(self: _FakeLM) -> ConfigLike:
        return _LMConfig()

    @classmethod
    def from_pretrained(cls: type[_FakeLM], path: str) -> LMModelProto:
        return cls()

    def generate(
        self: _FakeLM,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        batch = int(input_ids.size(0)) * num_return_sequences
        prompt_len = int(input_ids.size(1))
        total_len = prompt_len + max_new_tokens
        # Generate random tokens after prompt
        result = torch.zeros(batch, total_len, dtype=torch.long)
        for i in range(batch):
            result[i, :prompt_len] = input_ids[0]
            for j in range(prompt_len, total_len):
                result[i, j] = torch.randint(0, 10, (1,)).item()
        return result


def _make_prepared() -> PreparedLMModel:
    return PreparedLMModel(
        model=_FakeLM(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=16,
        tok_for_dataset=_Tok4DS(),
    )


# ===== Score tests =====


def test_read_text_or_path_text(settings_with_paths: Settings) -> None:
    cfg = ScoreConfig(text="hello", path=None, detail_level="summary", top_k=None, seed=None)
    result = _read_text_or_path(cfg, settings_with_paths)
    assert result == "hello"


def test_read_text_or_path_path(settings_with_paths: Settings) -> None:
    data_root = Path(settings_with_paths["app"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    test_file = data_root / "test_gpt2.txt"
    test_file.write_text("test content", encoding="utf-8")

    cfg = ScoreConfig(text=None, path=str(test_file), detail_level="summary", top_k=None, seed=None)
    result = _read_text_or_path(cfg, settings_with_paths)
    assert result == "test content"


def test_read_text_or_path_outside_data_root_raises(settings_with_paths: Settings) -> None:
    cfg = ScoreConfig(
        text=None, path="/tmp/outside.txt", detail_level="summary", top_k=None, seed=None
    )
    with pytest.raises(AppError, match="data_root"):
        _read_text_or_path(cfg, settings_with_paths)


def test_read_text_or_path_neither_raises(settings_with_paths: Settings) -> None:
    cfg = ScoreConfig(text=None, path=None, detail_level="summary", top_k=None, seed=None)
    with pytest.raises(AppError, match="either"):
        _read_text_or_path(cfg, settings_with_paths)


def test_get_logits_and_loss_short_sequence() -> None:
    prepared = _make_prepared()
    batch_ids: list[list[int]] = [[1]]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    logits, per_token_loss = _get_logits_and_loss(prepared, input_ids)
    assert logits.size(1) == 1
    assert per_token_loss.numel() == 0


def test_get_logits_and_loss_normal_sequence() -> None:
    prepared = _make_prepared()
    batch_ids: list[list[int]] = [[1, 2, 3]]
    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    logits, per_token_loss = _get_logits_and_loss(prepared, input_ids)
    assert logits.size(1) == 3
    assert per_token_loss.numel() == 2  # T-1 tokens


def test_compute_topk() -> None:
    prepared = _make_prepared()
    logits = torch.randn(1, 3, 10)
    result = _compute_topk(logits, prepared.tok_for_dataset, k=3)
    assert len(result) == 3
    for pos_topk in result:
        assert len(pos_topk) <= 3
        for token_str, prob in pos_topk:
            assert isinstance(token_str, str) and len(token_str) >= 0
            assert 0.0 <= prob <= 1.0


def test_score_gpt2_summary(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = ScoreConfig(text="abc", path=None, detail_level="summary", top_k=None, seed=42)
    result = score_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert math.isfinite(result["loss"])
    assert result["perplexity"] >= 1.0
    assert result["surprisal"] is None
    assert result["topk"] is None
    assert result["tokens"] is None


def test_score_gpt2_per_char(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = ScoreConfig(text="abc", path=None, detail_level="per_char", top_k=None, seed=42)
    result = score_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert result["surprisal"] is not None and len(result["surprisal"]) > 0
    assert len(result["surprisal"]) == 2
    assert result["tokens"] is not None and len(result["tokens"]) > 0


def test_score_gpt2_with_topk(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = ScoreConfig(text="abc", path=None, detail_level="summary", top_k=3, seed=42)
    result = score_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert result["topk"] is not None and len(result["topk"]) > 0


def test_score_gpt2_short_text(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = ScoreConfig(text="a", path=None, detail_level="summary", top_k=None, seed=42)
    result = score_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert result["loss"] == 0.0
    assert result["perplexity"] == 1.0


def test_score_gpt2_truncates_long_input(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    # Input longer than max_seq_len
    long_text = "a" * 100
    cfg = ScoreConfig(text=long_text, path=None, detail_level="summary", top_k=None, seed=42)
    result = score_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert math.isfinite(result["loss"])


# ===== Generate tests =====


def test_read_prompt_text(settings_with_paths: Settings) -> None:
    cfg = GenerateConfig(
        prompt_text="hello",
        prompt_path=None,
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    result = _read_prompt(cfg, settings_with_paths)
    assert result == "hello"


def test_read_prompt_path(settings_with_paths: Settings) -> None:
    data_root = Path(settings_with_paths["app"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    test_file = data_root / "prompt_gpt2.txt"
    test_file.write_text("test prompt", encoding="utf-8")

    cfg = GenerateConfig(
        prompt_text=None,
        prompt_path=str(test_file),
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    result = _read_prompt(cfg, settings_with_paths)
    assert result == "test prompt"


def test_read_prompt_outside_data_root_raises(settings_with_paths: Settings) -> None:
    cfg = GenerateConfig(
        prompt_text=None,
        prompt_path="/tmp/outside.txt",
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    with pytest.raises(AppError, match="data_root"):
        _read_prompt(cfg, settings_with_paths)


def test_read_prompt_neither_raises(settings_with_paths: Settings) -> None:
    cfg = GenerateConfig(
        prompt_text=None,
        prompt_path=None,
        max_new_tokens=10,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    with pytest.raises(AppError, match="either"):
        _read_prompt(cfg, settings_with_paths)


def test_check_stop_sequence_true() -> None:
    assert _check_stop_sequence("hello world", ["world"]) is True


def test_check_stop_sequence_false() -> None:
    assert _check_stop_sequence("hello world", ["foo"]) is False


def test_check_stop_sequence_empty() -> None:
    assert _check_stop_sequence("hello", []) is False


def test_generate_gpt2_basic(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=42,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1
    assert result["steps"] > 0
    assert len(result["eos_terminated"]) == 1


def test_generate_gpt2_multiple_sequences(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=42,
        num_return_sequences=3,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 3
    assert len(result["eos_terminated"]) == 3


def test_generate_gpt2_greedy(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=5,
        temperature=0.0,  # Greedy
        top_k=0,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=42,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1


def test_generate_gpt2_with_stop_sequences(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=False,
        stop_sequences=["0"],  # Stop when '0' is generated
        seed=42,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1


def test_generate_gpt2_truncates_long_prompt(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    # Prompt longer than max_seq_len - max_new_tokens
    long_prompt = "a" * 100
    cfg = GenerateConfig(
        prompt_text=long_prompt,
        prompt_path=None,
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=42,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1


def test_generate_gpt2_eos_termination(settings_with_paths: Settings) -> None:
    # Create a model that always generates EOS
    class _EosLM(_FakeLM):
        def generate(
            self: _EosLM,
            input_ids: torch.Tensor,
            *,
            max_new_tokens: int,
            do_sample: bool,
            temperature: float,
            top_k: int,
            top_p: float,
            num_return_sequences: int,
            eos_token_id: int,
            pad_token_id: int,
        ) -> torch.Tensor:
            batch = int(input_ids.size(0)) * num_return_sequences
            prompt_len = int(input_ids.size(1))
            total_len = prompt_len + max_new_tokens
            result = torch.zeros(batch, total_len, dtype=torch.long)
            for i in range(batch):
                result[i, :prompt_len] = input_ids[0]
                # Generate EOS immediately
                result[i, prompt_len] = eos_token_id
            return result

    prepared = PreparedLMModel(
        model=_EosLM(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=16,
        tok_for_dataset=_Tok4DS(),
    )
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=42,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert result["eos_terminated"][0] is True


def test_score_gpt2_no_seed(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = ScoreConfig(text="abc", path=None, detail_level="summary", top_k=None, seed=None)
    result = score_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert math.isfinite(result["loss"])


def test_generate_gpt2_no_seed(settings_with_paths: Settings) -> None:
    prepared = _make_prepared()
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=None,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1


def test_generate_gpt2_stop_sequence_not_found(settings_with_paths: Settings) -> None:
    """Test when stop sequences are specified but not found in generated text."""
    prepared = _make_prepared()
    # Use a stop sequence that won't appear in the output (numbers 0-9 as chars)
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=3,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=False,
        stop_sequences=["ZZZZZ", "YYYYY"],  # These won't appear in numeric output
        seed=42,
        num_return_sequences=1,
    )
    result = generate_gpt2(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1
    # Output should not be truncated since stop sequences weren't found
    assert len(result["outputs"][0]) > 0 and isinstance(result["outputs"][0], str)


# ===== Backend interface tests =====


def test_backend_score_valid_prepared(settings_with_paths: Settings) -> None:
    from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
    from model_trainer.core.services.model.backend_factory import create_gpt2_backend

    prepared = _make_prepared()
    backend = create_gpt2_backend(LocalTextDatasetBuilder())
    cfg = ScoreConfig(text="abc", path=None, detail_level="summary", top_k=None, seed=42)
    result = backend.score(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert result["loss"] >= 0.0
    assert result["perplexity"] >= 1.0


def test_backend_generate_valid_prepared(settings_with_paths: Settings) -> None:
    from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
    from model_trainer.core.services.model.backend_factory import create_gpt2_backend

    prepared = _make_prepared()
    backend = create_gpt2_backend(LocalTextDatasetBuilder())
    cfg = GenerateConfig(
        prompt_text="ab",
        prompt_path=None,
        max_new_tokens=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        stop_on_eos=True,
        stop_sequences=[],
        seed=42,
        num_return_sequences=1,
    )
    result = backend.generate(prepared=prepared, cfg=cfg, settings=settings_with_paths)
    assert len(result["outputs"]) == 1
    assert result["steps"] > 0
