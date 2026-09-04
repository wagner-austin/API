"""Tests for HuggingFace LM IO module."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str

from model_trainer.core.contracts.model import PreparedLMModel, QuantizationConfig
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.finetuning.strategies._test_hooks import (
    reset_hooks as reset_ft_hooks,
)
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    HFTokenizerProto,
    Hooks,
    reset_hooks,
)
from model_trainer.core.services.model.backends.hf_lm.io import (
    HFLMMetadata,
    _decode_metadata,
    _encode_metadata,
    _get_model_max_seq_len,
    _require_strategy_name,
    load_base_of_prepared_hf_lm,
    load_prepared_hf_lm_from_handle,
    read_hf_lm_metadata,
    save_prepared_hf_lm,
)
from model_trainer.core.types import ConfigLike, LMModelProto

from .testing import FakeHFModel, FakeHFTokenizer


class _FakeModelLoader:
    """Fake model loader for testing."""

    def __init__(self, name_prefix: str = "") -> None:
        self._name_prefix = name_prefix

    def __call__(
        self, model_id_or_path: str, quantization: QuantizationConfig | None
    ) -> LMModelProto:
        return FakeHFModel(f"{self._name_prefix}{model_id_or_path}")


class _FakeFullModelLoader:
    """Fake full model loader for finetuning strategy hook."""

    def __init__(self, name_prefix: str = "") -> None:
        self._name_prefix = name_prefix

    def __call__(self, model_path: str) -> LMModelProto:
        return FakeHFModel(f"{self._name_prefix}{model_path}")


class _FakeTokenizerLoader:
    """Fake tokenizer loader for testing."""

    def __call__(self, model_id_or_path: str) -> HFTokenizerProto:
        return FakeHFTokenizer()


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    reset_ft_hooks()
    yield
    reset_hooks()
    reset_ft_hooks()


class TestRequireStrategyName:
    """Tests for _require_strategy_name function."""

    def test_extracts_full_strategy(self) -> None:
        """Test extraction of 'full' strategy name from JSON object."""
        obj: JSONObject = {"strategy_name": "full"}
        result = _require_strategy_name(obj, "strategy_name")
        assert result == "full"

    def test_extracts_lora_strategy(self) -> None:
        """Test extraction of 'lora' strategy name from JSON object."""
        obj: JSONObject = {"strategy_name": "lora"}
        result = _require_strategy_name(obj, "strategy_name")
        assert result == "lora"

    def test_extracts_qlora_strategy(self) -> None:
        """Test extraction of 'qlora' strategy name from JSON object."""
        obj: JSONObject = {"strategy_name": "qlora"}
        result = _require_strategy_name(obj, "strategy_name")
        assert result == "qlora"

    def test_raises_for_missing_field(self) -> None:
        """Test that JSONTypeError is raised for missing field."""
        obj: JSONObject = {}
        with pytest.raises(JSONTypeError, match="Missing required field"):
            _require_strategy_name(obj, "strategy_name")

    def test_a_wellformed_string_naming_no_strategy_is_a_value_fault(self) -> None:
        """The shape is fine, so the failure is about the value, not the file.

        Distinct from the missing-field case above, which stays a
        ``JSONTypeError``: that one says the metadata is malformed, this one
        says the metadata is well-formed and names something that does not
        exist.
        """
        obj: JSONObject = {"strategy_name": "invalid"}
        with pytest.raises(AppError) as excinfo:
            _require_strategy_name(obj, "strategy_name")
        assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN

    def test_a_non_string_field_is_a_shape_fault(self) -> None:
        """A number where a name belongs never reaches the value check."""
        obj: JSONObject = {"strategy_name": 3}
        with pytest.raises(JSONTypeError):
            _require_strategy_name(obj, "strategy_name")


class TestEncodeMetadata:
    """Tests for _encode_metadata function."""

    def test_encodes_metadata_to_json(self) -> None:
        """Test encoding of HFLMMetadata to JSON object."""
        metadata = HFLMMetadata(
            strategy_name="full",
            hub_model_id="test/model",
            tokenizer_id="test-tok",
            is_peft=False,
            quantization=None,
        )
        result = _encode_metadata(metadata)
        assert result["strategy_name"] == "full"
        assert result["hub_model_id"] == "test/model"
        assert result["tokenizer_id"] == "test-tok"
        assert result["is_peft"] is False


class TestDecodeMetadata:
    """Tests for _decode_metadata function."""

    def test_decodes_valid_metadata(self) -> None:
        """Test decoding of valid JSON object to HFLMMetadata."""
        obj: JSONObject = {
            "strategy_name": "lora",
            "hub_model_id": "test/model",
            "tokenizer_id": "test-tok",
            "is_peft": True,
            "quantization": None,
        }
        result = _decode_metadata(obj)
        assert result["strategy_name"] == "lora"
        assert result["hub_model_id"] == "test/model"
        assert result["tokenizer_id"] == "test-tok"
        assert result["is_peft"] is True

    def test_raises_for_missing_hub_model_id(self) -> None:
        """Test that JSONTypeError is raised for missing hub_model_id."""
        obj: JSONObject = {"strategy_name": "full", "tokenizer_id": "tok", "is_peft": False}
        with pytest.raises(JSONTypeError, match="Missing required field 'hub_model_id'"):
            _decode_metadata(obj)

    def test_missing_tokenizer_id_returns_none(self) -> None:
        """Test that missing tokenizer_id returns None (optional for hf_lm)."""
        obj: JSONObject = {"strategy_name": "full", "hub_model_id": "model", "is_peft": False}
        result = _decode_metadata(obj)
        assert result["tokenizer_id"] is None

    def test_raises_for_missing_is_peft(self) -> None:
        """Test that JSONTypeError is raised for missing is_peft."""
        obj: JSONObject = {"strategy_name": "full", "hub_model_id": "model", "tokenizer_id": "tok"}
        with pytest.raises(JSONTypeError, match="Missing required field 'is_peft'"):
            _decode_metadata(obj)


class TestGetModelMaxSeqLen:
    """Tests for _get_model_max_seq_len function."""

    def test_returns_max_position_embeddings(self) -> None:
        """Test extraction of max_position_embeddings from config."""

        class _ConfigMPE(ConfigLike):
            max_position_embeddings = 1024

        class _Model(FakeHFModel):
            @property
            def config(self) -> ConfigLike:
                return _ConfigMPE()

        model = _Model()
        result = _get_model_max_seq_len(model)
        assert result == 1024

    def test_returns_n_positions_when_no_max_pos(self) -> None:
        """Test fallback to n_positions when max_position_embeddings missing."""

        class _ConfigNPos(ConfigLike):
            n_positions = 512

        class _Model(FakeHFModel):
            @property
            def config(self) -> ConfigLike:
                return _ConfigNPos()

        model = _Model()
        result = _get_model_max_seq_len(model)
        assert result == 512

    def test_returns_max_seq_length_when_no_other_attrs(self) -> None:
        """Test fallback to max_seq_length when other attrs missing."""

        class _ConfigMSL(ConfigLike):
            max_seq_length = 4096

        class _Model(FakeHFModel):
            @property
            def config(self) -> ConfigLike:
                return _ConfigMSL()

        model = _Model()
        result = _get_model_max_seq_len(model)
        assert result == 4096

    def test_returns_default_when_no_attrs(self) -> None:
        """Test that 2048 is returned when no sequence length attrs exist."""

        class _EmptyConfig(ConfigLike):
            pass

        class _Model(FakeHFModel):
            @property
            def config(self) -> ConfigLike:
                return _EmptyConfig()

        model = _Model()
        result = _get_model_max_seq_len(model)
        assert result == 2048


class TestSavePreparedHFLM:
    """Tests for save_prepared_hf_lm function."""

    def test_raises_when_strategy_name_is_none(self) -> None:
        """Test that ValueError is raised when strategy_name is None."""
        from model_trainer.core.services.model.backends.hf_lm.prepare import (
            HFTokenizerEncoder,
        )

        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=512,
            tok_for_dataset=HFTokenizerEncoder(FakeHFTokenizer()),
            strategy_name=None,
            hub_model_id="test/model",
            is_peft=False,
            quantization=None,
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="strategy_name is required"),
        ):
            save_prepared_hf_lm(prepared, tmpdir)

    def test_raises_when_hub_model_id_is_none(self) -> None:
        """Test that ValueError is raised when hub_model_id is None."""
        from model_trainer.core.services.model.backends.hf_lm.prepare import (
            HFTokenizerEncoder,
        )

        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=512,
            tok_for_dataset=HFTokenizerEncoder(FakeHFTokenizer()),
            strategy_name="full",
            hub_model_id=None,
            is_peft=False,
            quantization=None,
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="hub_model_id is required"),
        ):
            save_prepared_hf_lm(prepared, tmpdir)

    def test_raises_for_invalid_strategy_name(self) -> None:
        """A prepared model carrying an undeclared strategy name cannot be saved.

        ``PreparedLMModel.strategy_name`` is a bare ``str`` because it is
        reconstructed from disk, so this is the point where an unknown name is
        caught, with the same code every other entry point raises.
        """
        from model_trainer.core.services.model.backends.hf_lm.prepare import (
            HFTokenizerEncoder,
        )

        # strategy_name is str | None, so we can pass "invalid" directly
        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=512,
            tok_for_dataset=HFTokenizerEncoder(FakeHFTokenizer()),
            strategy_name="invalid",
            hub_model_id="test/model",
            is_peft=False,
            quantization=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AppError) as excinfo:
                save_prepared_hf_lm(prepared, tmpdir)
            assert excinfo.value.code is ModelTrainerErrorCode.STRATEGY_NAME_UNKNOWN

    def test_saves_prepared_model_with_full_strategy(self) -> None:
        """Test successful save of prepared model with full strategy."""
        from model_trainer.core.services.model.backends.hf_lm.prepare import (
            HFTokenizerEncoder,
        )

        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=512,
            tok_for_dataset=HFTokenizerEncoder(FakeHFTokenizer()),
            strategy_name="full",
            hub_model_id="test/base-model",
            is_peft=False,
            quantization=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_prepared_hf_lm(prepared, tmpdir)

            # Verify metadata was written
            metadata_path = Path(tmpdir) / "hf_lm_metadata.json"
            assert metadata_path.exists()

            # Verify metadata content
            from platform_core.json_utils import load_json_str, narrow_json_to_dict

            metadata_json = load_json_str(metadata_path.read_text(encoding="utf-8"))
            metadata_obj = narrow_json_to_dict(metadata_json)
            assert metadata_obj["strategy_name"] == "full"
            assert metadata_obj["hub_model_id"] == "test/base-model"
            assert metadata_obj["tokenizer_id"] == "test-tok"
            assert metadata_obj["is_peft"] is False


class TestLoadPreparedHFLMFromHandle:
    """Tests for load_prepared_hf_lm_from_handle function."""

    def test_raises_when_metadata_not_found(self) -> None:
        """Test that FileNotFoundError is raised when metadata is missing."""

        class _FakeTokHandle(TokenizerHandle):
            def encode(self, text: str) -> list[int]:
                return []

            def decode(self, ids: list[int]) -> str:
                return ""

            def token_to_id(self, token: str) -> int | None:
                return 0

            def get_vocab_size(self) -> int:
                return 100

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError, match="Metadata not found"),
        ):
            load_prepared_hf_lm_from_handle(tmpdir, _FakeTokHandle())

    def test_loads_prepared_model_successfully(self) -> None:
        """Test successful loading of prepared model."""
        from model_trainer.core.services.finetuning.strategies._test_hooks import (
            Hooks as FtHooks,
        )

        class _FakeTokHandle(TokenizerHandle):
            def encode(self, text: str) -> list[int]:
                return []

            def decode(self, ids: list[int]) -> str:
                return ""

            def token_to_id(self, token: str) -> int | None:
                return 0

            def get_vocab_size(self) -> int:
                return 100

        Hooks.load_hf_model = _FakeModelLoader()
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()
        # Also set the full strategy's load hook
        FtHooks.load_full_model = _FakeFullModelLoader(name_prefix="loaded-")

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata: JSONObject = {
                "strategy_name": "full",
                "hub_model_id": "test/base-model",
                "tokenizer_id": "test-tok",
                "is_peft": False,
                "quantization": None,
            }
            (Path(tmpdir) / "hf_lm_metadata.json").write_text(
                dump_json_str(metadata), encoding="utf-8"
            )

            result = load_prepared_hf_lm_from_handle(tmpdir, _FakeTokHandle())

            assert result.tokenizer_id == "test-tok"
            assert result.strategy_name == "full"
            assert result.hub_model_id == "test/base-model"
            assert result.is_peft is False


class TestLoadingTheBaseOfASavedRun:
    """The paired control for an adapter, which is not the same as a baseline.

    :func:`load_prepared_hf_lm_from_hub` loads UNQUANTIZED weights, because a
    baseline exists to be compared against and must carry no arm. The control
    for one specific adapter is a different object: it deliberately carries
    that adapter's quantization, because the question being asked is what the
    adapter did, and an adapter trained against NF4 weights compared against
    bfloat16 ones would be a comparison of two changes at once.
    """

    _QUANTIZED: ClassVar[QuantizationConfig] = {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }

    def _artifact(self, tmp_path: Path, *, quantized: bool) -> str:
        """Write a saved run's metadata.

        Args:
            tmp_path: Directory to write into.
            quantized: Whether the run recorded a quantization.

        Returns:
            The artifact directory, as a string.
        """
        from model_trainer.core.services.model.backends.hf_lm.io import _encode_metadata

        metadata = HFLMMetadata(
            strategy_name="qlora",
            hub_model_id="Qwen/Qwen2.5-Coder-1.5B",
            tokenizer_id=None,
            is_peft=True,
            quantization=self._QUANTIZED if quantized else None,
        )
        (tmp_path / "hf_lm_metadata.json").write_text(
            dump_json_str(_encode_metadata(metadata)), encoding="utf-8"
        )
        return str(tmp_path)

    def test_it_refuses_a_directory_that_is_not_a_saved_run(self, tmp_path: Path) -> None:
        """Guessing the strategy would reconstruct a different model."""
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            _ = load_base_of_prepared_hf_lm(str(tmp_path))

    def test_it_carries_the_runs_own_quantization(self, tmp_path: Path) -> None:
        """This is the whole difference from the hub baseline."""
        Hooks.load_hf_model = _FakeModelLoader()
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()

        result = load_base_of_prepared_hf_lm(self._artifact(tmp_path, quantized=True))

        assert result.quantization == self._QUANTIZED

    def test_it_attaches_no_adapter(self, tmp_path: Path) -> None:
        """Recorded rather than implied, so a record cannot mistake it for the arm."""
        Hooks.load_hf_model = _FakeModelLoader()
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()

        result = load_base_of_prepared_hf_lm(self._artifact(tmp_path, quantized=True))

        assert result.is_peft is False
        assert result.strategy_name is None

    def test_it_names_the_base_the_adapter_was_trained_against(self, tmp_path: Path) -> None:
        Hooks.load_hf_model = _FakeModelLoader()
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()

        result = load_base_of_prepared_hf_lm(self._artifact(tmp_path, quantized=True))

        assert result.hub_model_id == "Qwen/Qwen2.5-Coder-1.5B"

    def test_an_unquantized_run_reloads_unquantized(self, tmp_path: Path) -> None:
        """The control follows the run rather than assuming four-bit."""
        Hooks.load_hf_model = _FakeModelLoader()
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()

        result = load_base_of_prepared_hf_lm(self._artifact(tmp_path, quantized=False))

        assert result.quantization is None

    def test_metadata_reads_back_what_was_written(self, tmp_path: Path) -> None:
        """Three readers now open this file; one decoder answers all of them."""
        directory = self._artifact(tmp_path, quantized=True)

        assert read_hf_lm_metadata(directory)["strategy_name"] == "qlora"

    def test_the_control_and_the_arm_share_their_tokenizer_and_token_ids(
        self, tmp_path: Path
    ) -> None:
        """Two spellings of these is how the pair silently stops being paired."""
        from model_trainer.core.services.finetuning.strategies._test_hooks import (
            Hooks as FtHooks,
        )

        Hooks.load_hf_model = _FakeModelLoader()
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()
        FtHooks.load_full_model = _FakeFullModelLoader(name_prefix="loaded-")

        full = tmp_path / "full"
        full.mkdir()
        (full / "hf_lm_metadata.json").write_text(
            dump_json_str(
                {
                    "strategy_name": "full",
                    "hub_model_id": "test/base-model",
                    "tokenizer_id": "test-tok",
                    "is_peft": False,
                    "quantization": None,
                }
            ),
            encoding="utf-8",
        )

        arm = load_prepared_hf_lm_from_handle(str(full), None)
        control = load_base_of_prepared_hf_lm(str(full))

        assert (arm.eos_id, arm.pad_id) == (control.eos_id, control.pad_id)
        assert arm.max_seq_len == control.max_seq_len
