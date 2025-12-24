"""Tests for inference request validation in runs.py (score, generate, chat)."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import (
    _decode_chat_request,
    _decode_generate_request,
    _decode_score_request,
)


class TestScoreRequest:
    """Tests for score request validation."""

    def test_score_with_text(self) -> None:
        """Test score request with text."""
        payload: dict[str, JSONValue] = {"text": "hello world"}
        out = _decode_score_request(payload)
        assert out["text"] == "hello world"
        assert out["path"] is None

    def test_score_with_path(self) -> None:
        """Test score request with path."""
        payload: dict[str, JSONValue] = {"path": "/tmp/file.txt"}
        out = _decode_score_request(payload)
        assert out["path"] == "/tmp/file.txt"
        assert out["text"] is None

    def test_score_mutual_exclusion(self) -> None:
        """Test score request rejects both text and path."""
        payload: dict[str, JSONValue] = {"text": "hello", "path": "/tmp/file"}
        with pytest.raises(AppError) as exc:
            _ = _decode_score_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "mutually exclusive" in str(err.message)

    def test_score_requires_text_or_path(self) -> None:
        """Test score request requires text or path."""
        payload: dict[str, JSONValue] = {}
        with pytest.raises(AppError) as exc:
            _ = _decode_score_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "either text or path" in str(err.message)

    def test_score_detail_level_summary(self) -> None:
        """Test score request detail_level=summary default."""
        payload: dict[str, JSONValue] = {"text": "hello"}
        out = _decode_score_request(payload)
        assert out["detail_level"] == "summary"

    def test_score_detail_level_per_char(self) -> None:
        """Test score request detail_level=per_char."""
        payload: dict[str, JSONValue] = {"text": "hello", "detail_level": "per_char"}
        out = _decode_score_request(payload)
        assert out["detail_level"] == "per_char"

    def test_score_top_k(self) -> None:
        """Test score request with top_k."""
        payload: dict[str, JSONValue] = {"text": "hello", "top_k": 5}
        out = _decode_score_request(payload)
        assert out["top_k"] == 5

    def test_score_top_k_none_default(self) -> None:
        """Test score request top_k=None default."""
        payload: dict[str, JSONValue] = {"text": "hello"}
        out = _decode_score_request(payload)
        assert out["top_k"] is None

    def test_score_seed(self) -> None:
        """Test score request with seed."""
        payload: dict[str, JSONValue] = {"text": "hello", "seed": 42}
        out = _decode_score_request(payload)
        assert out["seed"] == 42

    def test_score_seed_none_default(self) -> None:
        """Test score request seed=None default."""
        payload: dict[str, JSONValue] = {"text": "hello"}
        out = _decode_score_request(payload)
        assert out["seed"] is None


class TestGenerateRequest:
    """Tests for generate request validation."""

    def test_generate_with_prompt_text(self) -> None:
        """Test generate request with prompt_text."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello"}
        out = _decode_generate_request(payload)
        assert out["prompt_text"] == "hello"
        assert out["prompt_path"] is None

    def test_generate_with_prompt_path(self) -> None:
        """Test generate request with prompt_path."""
        payload: dict[str, JSONValue] = {"prompt_path": "/tmp/prompt.txt"}
        out = _decode_generate_request(payload)
        assert out["prompt_path"] == "/tmp/prompt.txt"
        assert out["prompt_text"] is None

    def test_generate_mutual_exclusion(self) -> None:
        """Test generate request rejects both prompt_text and prompt_path."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello", "prompt_path": "/tmp"}
        with pytest.raises(AppError) as exc:
            _ = _decode_generate_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "mutually exclusive" in str(err.message)

    def test_generate_requires_prompt(self) -> None:
        """Test generate request requires prompt."""
        payload: dict[str, JSONValue] = {}
        with pytest.raises(AppError) as exc:
            _ = _decode_generate_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "either prompt_text or prompt_path" in str(err.message)

    def test_generate_defaults(self) -> None:
        """Test generate request defaults."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello"}
        out = _decode_generate_request(payload)
        assert out["max_new_tokens"] == 64
        assert out["temperature"] == 1.0
        assert out["top_k"] == 50
        assert out["top_p"] == 1.0
        assert out["stop_on_eos"] is True
        assert out["stop_sequences"] == []
        assert out["seed"] is None
        assert out["num_return_sequences"] == 1

    def test_generate_stop_on_eos_false(self) -> None:
        """Test generate request stop_on_eos=False."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello", "stop_on_eos": False}
        out = _decode_generate_request(payload)
        assert out["stop_on_eos"] is False

    def test_generate_stop_on_eos_type_error(self) -> None:
        """Test generate request stop_on_eos type error."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello", "stop_on_eos": "yes"}
        with pytest.raises(AppError) as exc:
            _ = _decode_generate_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "stop_on_eos must be a boolean" in str(err.message)

    def test_generate_stop_sequences(self) -> None:
        """Test generate request with stop_sequences."""
        payload: dict[str, JSONValue] = {
            "prompt_text": "hello",
            "stop_sequences": ["\n", "END"],
        }
        out = _decode_generate_request(payload)
        assert out["stop_sequences"] == ["\n", "END"]

    def test_generate_stop_sequences_type_error(self) -> None:
        """Test generate request stop_sequences must be list."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello", "stop_sequences": "end"}
        with pytest.raises(AppError) as exc:
            _ = _decode_generate_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "stop_sequences must be a list" in str(err.message)

    def test_generate_stop_sequences_element_type_error(self) -> None:
        """Test generate request stop_sequences elements must be strings."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello", "stop_sequences": [123]}
        with pytest.raises(AppError) as exc:
            _ = _decode_generate_request(payload)
        err: AppError[ErrorCode] = exc.value
        assert "stop_sequences[0] must be a string" in str(err.message)

    def test_generate_seed(self) -> None:
        """Test generate request with seed."""
        payload: dict[str, JSONValue] = {"prompt_text": "hello", "seed": 42}
        out = _decode_generate_request(payload)
        assert out["seed"] == 42

    def test_generate_num_return_sequences(self) -> None:
        """Test generate request with num_return_sequences."""
        payload: dict[str, JSONValue] = {
            "prompt_text": "hello",
            "num_return_sequences": 4,
        }
        out = _decode_generate_request(payload)
        assert out["num_return_sequences"] == 4


class TestChatRequest:
    """Tests for chat request validation."""

    def test_chat_basic(self) -> None:
        """Test basic chat request."""
        payload: dict[str, JSONValue] = {"message": "hello"}
        out = _decode_chat_request(payload)
        assert out["message"] == "hello"
        assert out["session_id"] is None

    def test_chat_with_session_id(self) -> None:
        """Test chat request with session_id."""
        payload: dict[str, JSONValue] = {"message": "hello", "session_id": "sess-123"}
        out = _decode_chat_request(payload)
        assert out["session_id"] == "sess-123"

    def test_chat_defaults(self) -> None:
        """Test chat request defaults."""
        payload: dict[str, JSONValue] = {"message": "hello"}
        out = _decode_chat_request(payload)
        assert out["max_new_tokens"] == 128
        assert out["temperature"] == 0.8
        assert out["top_k"] == 50
        assert out["top_p"] == 0.95
