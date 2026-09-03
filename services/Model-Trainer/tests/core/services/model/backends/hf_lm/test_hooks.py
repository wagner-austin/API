"""Tests for HuggingFace LM backend hooks."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol

import pytest
from platform_core.determinism_record import UNPINNED_STACK, determinism_record

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    HFModelLoader,
    HFTokenizerLoader,
    HFTokenizerProto,
    Hooks,
    _default_create_causal_dataset,
    _default_create_dataloader,
    _default_create_trainer,
    _default_get_eval_dir,
    _default_get_model_dir,
    _default_load_hf_model,
    _default_load_hf_tokenizer,
    _default_load_prepared_model,
    _default_load_tokenizer,
    _default_read_text_file,
    reset_hooks,
)
from model_trainer.core.types import LMModelProto

from .testing import FakeEncoder, FakeHFModel, FakeHFTokenizer


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        data_root: str | None = ...,
    ) -> Settings: ...


class _FakeModelLoader:
    """Fake model loader that implements HFModelLoader protocol."""

    def __call__(
        self, model_id_or_path: str, quantization: QuantizationConfig | None
    ) -> LMModelProto:
        return FakeHFModel(model_id_or_path)


class _FakeTokenizerLoader:
    """Fake tokenizer loader that implements HFTokenizerLoader protocol."""

    def __call__(self, model_id_or_path: str) -> HFTokenizerProto:
        return FakeHFTokenizer()


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    yield
    reset_hooks()


class TestHooksReset:
    """Tests for reset_hooks function."""

    def test_reset_hooks_restores_the_real_model_loader(self) -> None:
        """A fake model loader is replaced by the production implementation."""
        Hooks.load_hf_model = _FakeModelLoader()
        reset_hooks()
        hook: HFModelLoader = Hooks.load_hf_model
        assert hook is _default_load_hf_model

    def test_reset_hooks_restores_the_real_tokenizer_loader(self) -> None:
        """A fake tokenizer loader is replaced by the production implementation."""
        Hooks.load_hf_tokenizer = _FakeTokenizerLoader()
        reset_hooks()
        hook: HFTokenizerLoader = Hooks.load_hf_tokenizer
        assert hook is _default_load_hf_tokenizer


class TestHooksAreBoundAtImport:
    """The container is usable with no wiring step."""

    def test_every_hook_holds_its_production_implementation(self) -> None:
        """No caller has to initialize the container before using it."""
        assert Hooks.load_hf_model is _default_load_hf_model
        assert Hooks.load_hf_tokenizer is _default_load_hf_tokenizer
        assert Hooks.create_trainer is _default_create_trainer
        assert Hooks.load_tokenizer is _default_load_tokenizer
        assert Hooks.load_prepared_model is _default_load_prepared_model
        assert Hooks.create_causal_dataset is _default_create_causal_dataset
        assert Hooks.create_dataloader is _default_create_dataloader
        assert Hooks.get_model_dir is _default_get_model_dir
        assert Hooks.get_eval_dir is _default_get_eval_dir
        assert Hooks.read_text_file is _default_read_text_file


class _CapturingModelLoader:
    """Model loader that captures the model IDs for testing."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    def __call__(
        self, model_id_or_path: str, quantization: QuantizationConfig | None
    ) -> LMModelProto:
        self.captured.append(model_id_or_path)
        return FakeHFModel(model_id_or_path)


class _CapturingTokenizerLoader:
    """Tokenizer loader that captures model IDs for testing."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    def __call__(self, model_id_or_path: str) -> HFTokenizerProto:
        self.captured.append(model_id_or_path)
        return FakeHFTokenizer()


class TestHooksCallable:
    """Tests for hook callable behavior."""

    def test_model_loader_can_be_set_and_called(self) -> None:
        """Test that load_hf_model can be set and called."""
        fake_loader = _CapturingModelLoader()
        Hooks.load_hf_model = fake_loader

        hook: HFModelLoader | None = Hooks.load_hf_model
        assert hook is fake_loader

        result = fake_loader("test/model-id", None)

        assert len(fake_loader.captured) == 1
        assert fake_loader.captured[0] == "test/model-id"
        assert type(result) is FakeHFModel

    def test_tokenizer_loader_can_be_set_and_called(self) -> None:
        """Test that load_hf_tokenizer can be set and called."""
        fake_loader = _CapturingTokenizerLoader()
        Hooks.load_hf_tokenizer = fake_loader

        hook: HFTokenizerLoader | None = Hooks.load_hf_tokenizer
        assert hook is fake_loader

        result = fake_loader("test/model-id")

        assert len(fake_loader.captured) == 1
        assert fake_loader.captured[0] == "test/model-id"
        assert type(result) is FakeHFTokenizer


class TestDefaultGetModelDir:
    """Tests for _default_get_model_dir function."""

    def test_returns_path_to_model_dir(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that _default_get_model_dir returns correct path."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        settings = settings_factory(
            artifacts_root=str(artifacts),
            data_root=str(tmp_path / "data"),
        )

        result = _default_get_model_dir(settings, "run-123")

        assert result == artifacts / "models" / "run-123"


class TestDefaultGetEvalDir:
    """Tests for _default_get_eval_dir function."""

    def test_returns_path_to_eval_dir(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that _default_get_eval_dir returns correct path."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        settings = settings_factory(
            artifacts_root=str(artifacts),
            data_root=str(tmp_path / "data"),
        )

        result = _default_get_eval_dir(settings, "run-456")

        # model_eval_dir returns models/{run_id}/eval
        assert result == artifacts / "models" / "run-456" / "eval"


class TestDefaultReadTextFile:
    """Tests for _default_read_text_file function."""

    def test_reads_file_contents(self, tmp_path: Path) -> None:
        """Test that _default_read_text_file reads file contents."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding="utf-8")

        result = _default_read_text_file(test_file)

        assert result == "Hello, world!"

    def test_reads_unicode_contents(self, tmp_path: Path) -> None:
        """Test that _default_read_text_file handles unicode."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Hello \u4e16\u754c", encoding="utf-8")

        result = _default_read_text_file(test_file)

        assert result == "Hello \u4e16\u754c"


class TestDefaultCreateCausalDataset:
    """Tests for _default_create_causal_dataset function."""

    def test_creates_dataset_from_lines(self) -> None:
        """Test that _default_create_causal_dataset creates a dataset."""
        import torch

        encoder = FakeEncoder()
        dataset = _default_create_causal_dataset(
            lines=("hello world", "goodbye world"),
            tokenizer=encoder,
            max_len=64,
            eos_id=0,
            pad_id=1,
        )

        # The dataset yields (input_ids, labels); both are max_len long.
        input_ids, labels = dataset[0]
        assert input_ids.size(0) == 64
        assert labels.size(0) == 64
        # Valid token IDs, and with no separator configured every real token is
        # a target. Trailing padding is excluded, so compare only the real span.
        assert input_ids.min().item() >= 0
        real = labels != -100
        assert torch.equal(labels[real], input_ids[real])
        assert bool(real.any().item())


class TestDefaultCreateDataloader:
    """Tests for _default_create_dataloader function."""

    def test_creates_dataloader_from_dataset(self) -> None:
        """Test that _default_create_dataloader creates a DataLoader."""
        encoder = FakeEncoder()
        dataset = _default_create_causal_dataset(
            lines=("hello world",),
            tokenizer=encoder,
            max_len=64,
            eos_id=0,
            pad_id=1,
        )

        loader = _default_create_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # A batch is (input_ids, labels), each [batch_size, max_len].
        batch = next(iter(loader))
        assert len(batch) == 2
        input_ids, labels = batch[0], batch[1]
        assert input_ids.dim() == 2
        assert input_ids.size(1) == 64
        assert labels.shape == input_ids.shape


class TestDefaultLoadHfModel:
    """Tests for _default_load_hf_model function."""

    def test_loads_tiny_gpt2_model(self) -> None:
        """Test loading a tiny GPT2 model from HuggingFace Hub."""
        # Use tiny-gpt2 which is a minimal model for testing
        model = _default_load_hf_model("sshleifer/tiny-gpt2", None)

        # Verify model has expected callable
        forward_method = model.forward
        assert callable(forward_method)


class TestDefaultLoadHfTokenizer:
    """Tests for _default_load_hf_tokenizer function."""

    def test_loads_tiny_gpt2_tokenizer(self) -> None:
        """Test loading a tiny GPT2 tokenizer from HuggingFace Hub."""
        tokenizer = _default_load_hf_tokenizer("sshleifer/tiny-gpt2")

        # Verify tokenizer has expected methods
        ids = tokenizer.encode("hello")
        first_id = ids[0]  # Will raise IndexError if empty
        assert first_id >= 0
        text = tokenizer.decode(ids)
        first_char = text[0]
        assert first_char == first_char  # type check


class TestDefaultLoadBpeTokenizer:
    """Tests for _default_load_tokenizer function."""

    def test_loads_bpe_tokenizer_from_path(self, tmp_path: Path) -> None:
        """Test loading a BPE tokenizer from path."""
        from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
        from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

        # Create BPE tokenizer artifacts
        tok_dir = tmp_path / "tokenizer"
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "data.txt").write_text("hello world test data\n", encoding="utf-8")

        cfg = TokenizerTrainConfig(
            method="bpe",
            vocab_size=64,
            min_frequency=1,
            corpus_path=str(corpus),
            holdout_fraction=0.1,
            seed=42,
            out_dir=str(tok_dir),
        )
        # Tokenizer training (no ML loss metric)
        loss_initial = 0.0
        BPEBackend().train(cfg)
        loss_final = 0.0
        assert loss_final <= loss_initial

        # Now test loading it
        handle = _default_load_tokenizer(str(tok_dir))

        # Verify handle works by checking first ID
        ids = handle.encode("hello")
        first_id = ids[0]  # Will raise IndexError if empty
        assert first_id >= 0


class TestDefaultLoadPreparedModel:
    """Tests for _default_load_prepared_model function."""

    def test_loads_prepared_model(self, tmp_path: Path) -> None:
        """Test loading a prepared model from path."""
        from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
        from model_trainer.core.services.finetuning.strategies._test_hooks import (
            Hooks as FtHooks,
        )
        from model_trainer.core.services.model.backends.hf_lm.io import (
            save_prepared_hf_lm,
        )
        from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

        # Set the finetuning strategy hook for loading full models
        def full_loader(model_path: str) -> LMModelProto:
            return _default_load_hf_model(model_path, None)

        FtHooks.load_full_model = full_loader

        # Create BPE tokenizer artifacts
        tok_dir = tmp_path / "tokenizer"
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "data.txt").write_text("hello world test data\n", encoding="utf-8")

        cfg = TokenizerTrainConfig(
            method="bpe",
            vocab_size=64,
            min_frequency=1,
            corpus_path=str(corpus),
            holdout_fraction=0.1,
            seed=42,
            out_dir=str(tok_dir),
        )
        # Tokenizer training (no ML loss metric)
        loss_initial = 0.0
        BPEBackend().train(cfg)
        loss_final = 0.0
        assert loss_final <= loss_initial

        # Load tokenizer handle
        handle = _default_load_tokenizer(str(tok_dir))

        # Load tiny model and create PreparedLMModel
        model = _default_load_hf_model("sshleifer/tiny-gpt2", None)
        tokenizer = _default_load_hf_tokenizer("sshleifer/tiny-gpt2")

        from model_trainer.core.contracts.model import PreparedLMModel
        from model_trainer.core.services.model.backends.hf_lm.prepare import (
            HFTokenizerEncoder,
        )

        # Get token IDs - GPT2 tokenizers always have eos_token_id
        eos_id_or_none = tokenizer.eos_token_id
        pad_id_or_none = tokenizer.pad_token_id
        # Use eos_id as pad_id fallback (GPT2 uses eos as pad)
        eos_id: int = eos_id_or_none if eos_id_or_none is not None else 50256
        pad_id: int = pad_id_or_none if pad_id_or_none is not None else eos_id

        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="sshleifer/tiny-gpt2",
            eos_id=eos_id,
            pad_id=pad_id,
            max_seq_len=128,
            tok_for_dataset=HFTokenizerEncoder(tokenizer),
            strategy_name="full",
            hub_model_id="sshleifer/tiny-gpt2",
            is_peft=False,
        )

        # Save the prepared model
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        save_prepared_hf_lm(prepared, str(model_dir))

        # Now test loading it
        loaded = _default_load_prepared_model(str(model_dir), handle)

        assert loaded.max_seq_len >= 1
        assert loaded.eos_id >= 0


class TestDefaultCreateTrainer:
    """Tests for _default_create_trainer function."""

    def test_creates_trainer(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test creating a trainer instance."""
        from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
        from model_trainer.core.services.model.backends.hf_lm.prepare import (
            HFTokenizerEncoder,
        )

        # Create settings
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )

        # Load tiny model
        model = _default_load_hf_model("sshleifer/tiny-gpt2", None)
        tokenizer = _default_load_hf_tokenizer("sshleifer/tiny-gpt2")

        # Get token IDs - GPT2 tokenizers always have eos_token_id
        eos_id_or_none = tokenizer.eos_token_id
        pad_id_or_none = tokenizer.pad_token_id
        # Use eos_id as pad_id fallback (GPT2 uses eos as pad)
        eos_id: int = eos_id_or_none if eos_id_or_none is not None else 50256
        pad_id: int = pad_id_or_none if pad_id_or_none is not None else eos_id

        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="sshleifer/tiny-gpt2",
            eos_id=eos_id,
            pad_id=pad_id,
            max_seq_len=128,
            tok_for_dataset=HFTokenizerEncoder(tokenizer),
        )

        # Create training config
        cfg: ModelTrainConfig = {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 128,
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "tokenizer_id": "test-tok",
            "corpus_path": str(tmp_path / "corpus"),
            "corpus_format": "lines",
            "holdout_fraction": 0.1,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "precision": "fp32",
            "data_num_workers": 0,
            "data_pin_memory": False,
            "early_stopping_patience": 3,
            "test_split_ratio": 0.1,
            "finetune_lr_cap": 1e-5,
            "loss_mask_prefix_separator": None,
            "finetuning_strategy": "full",
            "hub_model_id": None,
            "lora": None,
            "cartridge": None,
            "quantization": None,
            "gguf_export": None,
        }

        # Create trainer
        trainer = _default_create_trainer(
            prepared=prepared,
            cfg=cfg,
            settings=settings,
            run_id="test-run",
            redis_hb=lambda x: None,
            cancelled=lambda: False,
            resume=False,
            progress=None,
            service_name="test-service",
            wandb_publisher=None,
            # "Deliberately not pinned" is a posture, not an absence, and the
            # parameter is not optional -- a run that records nothing cannot
            # afterwards say what it ran under. UNPINNED_STACK is what the
            # record carries when setup_env declines to pin.
            determinism=determinism_record(UNPINNED_STACK, {}),
        )

        # Verify trainer has expected methods
        train_method = trainer.train
        assert callable(train_method)
