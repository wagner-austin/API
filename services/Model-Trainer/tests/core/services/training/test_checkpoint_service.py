"""Tests for training-checkpoint persistence, RNG capture and validation."""

from __future__ import annotations

import random

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingCheckpointMeta,
    encode_training_checkpoint_meta,
)
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.infra.paths import checkpoint_path, checkpoints_dir
from model_trainer.core.services.training.checkpoint import (
    RngStates,
    TrainingCheckpoint,
    capture_rng_states,
    checkpoint_exists,
    delete_training_checkpoint,
    load_training_checkpoint,
    restore_rng_states,
    save_training_checkpoint,
)
from model_trainer.core.types import TorchStateValue

RUN_ID = "run-ckpt-test"


def _make_cfg() -> ModelTrainConfig:
    """Build a resolved CPU training config for checkpoint tests."""
    return {
        "model_family": "char_lstm",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.001,
        "tokenizer_id": "tok",
        "corpus_path": "/data/corpus_cache/fid.txt",
        "holdout_fraction": 0.0,
        "seed": 7,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0001,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }


def _make_meta(run_id: str = RUN_ID) -> TrainingCheckpointMeta:
    """Build checkpoint metadata for the given run."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "epochs_completed": 2,
        "global_step": 10,
        "last_loss": 0.5,
        "best_val_loss": None,
        "epochs_no_improve": 0,
        "best_saved": False,
        "total_samples_processed": 20,
        "total_tokens_processed": 320,
        "elapsed_seconds": 12.5,
        "started_at_iso": "2026-08-18T00:00:00",
        "epoch_summaries": [],
        "config": _make_cfg(),
    }


def _make_checkpoint(run_id: str = RUN_ID) -> TrainingCheckpoint:
    """Build a checkpoint with small real tensors."""
    return TrainingCheckpoint(
        meta=_make_meta(run_id),
        model_state={"weight": torch.ones(2, 2), "bias": torch.zeros(2)},
        optimizer_state={"state": {}, "param_groups": []},
        rng=capture_rng_states(),
    )


class TestSaveLoadRoundTrip:
    """Persistence round-trips through one atomic file."""

    def test_round_trip_preserves_meta_and_tensors(self, settings_with_paths: Settings) -> None:
        saved_path = save_training_checkpoint(settings_with_paths, _make_checkpoint())
        assert saved_path == checkpoint_path(settings_with_paths, RUN_ID)
        assert checkpoint_exists(settings_with_paths, RUN_ID)

        loaded = load_training_checkpoint(settings_with_paths, RUN_ID)
        assert loaded.meta == _make_meta()
        assert torch.equal(loaded.model_state["weight"], torch.ones(2, 2))
        assert torch.equal(loaded.model_state["bias"], torch.zeros(2))
        assert loaded.optimizer_state == {"state": {}, "param_groups": []}

    def test_save_replaces_previous_checkpoint(self, settings_with_paths: Settings) -> None:
        _ = save_training_checkpoint(settings_with_paths, _make_checkpoint())
        second = _make_checkpoint()
        second.meta["epochs_completed"] = 3
        _ = save_training_checkpoint(settings_with_paths, second)

        loaded = load_training_checkpoint(settings_with_paths, RUN_ID)
        assert loaded.meta["epochs_completed"] == 3
        leftovers = list(checkpoints_dir(settings_with_paths).glob("*.tmp"))
        assert leftovers == []

    def test_delete_removes_file_and_reports(self, settings_with_paths: Settings) -> None:
        _ = save_training_checkpoint(settings_with_paths, _make_checkpoint())
        assert delete_training_checkpoint(settings_with_paths, RUN_ID) is True
        assert checkpoint_exists(settings_with_paths, RUN_ID) is False
        assert delete_training_checkpoint(settings_with_paths, RUN_ID) is False


class TestLoadValidation:
    """Loading refuses every malformed state with a typed error."""

    def _write_payload(self, settings: Settings, payload: TorchStateValue) -> None:
        checkpoints_dir(settings).mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(checkpoint_path(settings, RUN_ID)))

    def _assert_load_fails(
        self, settings: Settings, code: ModelTrainerErrorCode, match: str
    ) -> None:
        with pytest.raises(AppError) as excinfo:
            _ = load_training_checkpoint(settings, RUN_ID)
        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == code
        assert match in str(exc)

    def test_missing_file_raises_not_found(self, settings_with_paths: Settings) -> None:
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND, "no checkpoint"
        )

    def test_non_mapping_payload_is_corrupt(self, settings_with_paths: Settings) -> None:
        self._write_payload(settings_with_paths, [1, 2, 3])
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "mapping"
        )

    def test_non_string_payload_key_is_corrupt(self, settings_with_paths: Settings) -> None:
        self._write_payload(settings_with_paths, {1: "x"})
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "non-string key"
        )

    def test_missing_entry_is_corrupt(self, settings_with_paths: Settings) -> None:
        self._write_payload(settings_with_paths, {"model_state": {}})
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "meta_json"
        )

    def test_non_string_meta_json_is_corrupt(self, settings_with_paths: Settings) -> None:
        self._write_payload(settings_with_paths, {"meta_json": 7})
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "string"
        )

    def test_meta_json_non_object_is_corrupt(self, settings_with_paths: Settings) -> None:
        self._write_payload(settings_with_paths, {"meta_json": dump_json_str([1])})
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "decode to an object"
        )

    def test_meta_failing_validation_is_corrupt(self, settings_with_paths: Settings) -> None:
        self._write_payload(settings_with_paths, {"meta_json": dump_json_str({"run_id": RUN_ID})})
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "failed validation"
        )

    def test_schema_version_mismatch_is_refused(self, settings_with_paths: Settings) -> None:
        meta = _make_meta()
        meta["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
        self._write_payload(
            settings_with_paths,
            {"meta_json": dump_json_str(encode_training_checkpoint_meta(meta))},
        )
        self._assert_load_fails(
            settings_with_paths,
            ModelTrainerErrorCode.CHECKPOINT_SCHEMA_UNSUPPORTED,
            "schema version",
        )

    def test_run_id_mismatch_is_corrupt(self, settings_with_paths: Settings) -> None:
        meta = _make_meta(run_id="some-other-run")
        self._write_payload(
            settings_with_paths,
            {"meta_json": dump_json_str(encode_training_checkpoint_meta(meta))},
        )
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "some-other-run"
        )

    def _valid_payload(self) -> dict[str, TorchStateValue]:
        checkpoint = _make_checkpoint()
        return {
            "meta_json": dump_json_str(encode_training_checkpoint_meta(checkpoint.meta)),
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "rng_torch": checkpoint.rng.torch_state,
            "rng_cuda": checkpoint.rng.cuda_states,
            "rng_python_json": checkpoint.rng.python_state_json,
        }

    def test_non_mapping_model_state_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["model_state"] = [1]
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "model_state"
        )

    def test_non_string_model_state_key_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["model_state"] = {1: torch.ones(1)}
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "non-string key"
        )

    def test_non_tensor_model_state_value_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["model_state"] = {"weight": 5}
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "weight"
        )

    def test_non_mapping_optimizer_state_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["optimizer_state"] = 3
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "optimizer_state"
        )

    def test_non_string_optimizer_key_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["optimizer_state"] = {2: []}
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "non-string key"
        )

    def test_non_tensor_rng_torch_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["rng_torch"] = "x"
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "rng_torch"
        )

    def test_non_list_rng_cuda_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["rng_cuda"] = "x"
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "rng_cuda"
        )

    def test_non_tensor_rng_cuda_entry_is_corrupt(self, settings_with_paths: Settings) -> None:
        payload = self._valid_payload()
        payload["rng_cuda"] = ["x"]
        self._write_payload(settings_with_paths, payload)
        self._assert_load_fails(
            settings_with_paths, ModelTrainerErrorCode.CHECKPOINT_CORRUPT, "rng_cuda[0]"
        )


class TestRngStates:
    """RNG capture and restore reproduce sampling exactly."""

    def test_round_trip_reproduces_python_and_torch_sequences(self) -> None:
        random.seed(123)
        torch.manual_seed(123)
        _ = random.random()
        _ = torch.rand(3)

        states = capture_rng_states()
        expected_python = [random.random() for _ in range(5)]
        expected_torch = torch.rand(5)

        for _ in range(11):
            _ = random.random()
        _ = torch.rand(7)

        restore_rng_states(states)
        assert [random.random() for _ in range(5)] == expected_python
        assert torch.equal(torch.rand(5), expected_torch)

    def test_restore_skips_cuda_when_no_states_captured(self) -> None:
        states = RngStates(
            torch_state=torch.get_rng_state(),
            cuda_states=[],
            python_state_json=capture_rng_states().python_state_json,
        )
        restore_rng_states(states)

    def test_restore_rejects_invalid_python_state_json(self) -> None:
        states = RngStates(
            torch_state=torch.get_rng_state(),
            cuda_states=[],
            python_state_json=dump_json_str([1, 2]),
        )
        with pytest.raises(JSONTypeError, match="object"):
            restore_rng_states(states)

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"version": 3, "internal_state": "x", "gauss_next": None}, "internal_state"),
            (
                {"version": 3, "internal_state": [1, "x"], "gauss_next": None},
                r"internal_state\[1\]",
            ),
            (
                {"version": 3, "internal_state": [1, True], "gauss_next": None},
                r"internal_state\[1\]",
            ),
            ({"version": 3, "internal_state": [1, 2], "gauss_next": "x"}, "gauss_next"),
            ({"version": 3, "internal_state": [1, 2], "gauss_next": True}, "gauss_next"),
        ],
    )
    def test_restore_rejects_malformed_python_state(self, payload: JSONObject, match: str) -> None:
        states = RngStates(
            torch_state=torch.get_rng_state(),
            cuda_states=[],
            python_state_json=dump_json_str(payload),
        )
        with pytest.raises(JSONTypeError, match=match):
            restore_rng_states(states)

    def test_restore_accepts_integer_gauss_carry(self) -> None:
        captured = narrow_json_to_dict(load_json_str(capture_rng_states().python_state_json))
        edited: JSONObject = dict(captured)
        edited["gauss_next"] = 1
        states = RngStates(
            torch_state=torch.get_rng_state(),
            cuda_states=[],
            python_state_json=dump_json_str(edited),
        )
        restore_rng_states(states)


class TestPythonRngCaptureGuards:
    """Capture refuses malformed states from the hooked random module."""

    def _with_fake_getstate(self, value: tuple[TorchStateValue, ...], match: str) -> None:
        original = _test_hooks.random_getstate

        def _fake() -> tuple[TorchStateValue, ...]:
            return value

        _test_hooks.random_getstate = _fake
        try:
            with pytest.raises(RuntimeError, match=match):
                _ = capture_rng_states()
        finally:
            _test_hooks.random_getstate = original

    def test_wrong_arity_rejected(self) -> None:
        self._with_fake_getstate((3, (1, 2)), "expected 3")

    def test_non_int_version_rejected(self) -> None:
        self._with_fake_getstate(("3", (1, 2), None), "version")

    def test_bool_version_rejected(self) -> None:
        self._with_fake_getstate((True, (1, 2), None), "version")

    def test_non_tuple_internal_state_rejected(self) -> None:
        self._with_fake_getstate((3, [1, 2], None), "not tuple")

    def test_non_int_internal_value_rejected(self) -> None:
        self._with_fake_getstate((3, (1, "x"), None), "not int")

    def test_bool_internal_value_rejected(self) -> None:
        self._with_fake_getstate((3, (1, True), None), "not int")

    def test_non_float_gauss_rejected(self) -> None:
        self._with_fake_getstate((3, (1, 2), "x"), "gauss")

    def test_float_gauss_accepted(self) -> None:
        original = _test_hooks.random_getstate

        def _fake() -> tuple[TorchStateValue, ...]:
            return (3, (1, 2), 0.5)

        _test_hooks.random_getstate = _fake
        try:
            states = capture_rng_states()
        finally:
            _test_hooks.random_getstate = original
        assert '"gauss_next":0.5' in states.python_state_json


class TestCudaRngBranches:
    """CUDA generator states route through hooks so both branches test."""

    def test_capture_includes_cuda_states_when_available(self) -> None:
        original_avail = _test_hooks.cuda_is_available
        original_get = _test_hooks.torch_cuda_get_rng_state_all
        fake_states = [torch.zeros(4, dtype=torch.uint8)]

        def _fake_avail() -> bool:
            return True

        def _fake_get() -> list[torch.Tensor]:
            return fake_states

        _test_hooks.cuda_is_available = _fake_avail
        _test_hooks.torch_cuda_get_rng_state_all = _fake_get
        try:
            states = capture_rng_states()
        finally:
            _test_hooks.cuda_is_available = original_avail
            _test_hooks.torch_cuda_get_rng_state_all = original_get
        assert states.cuda_states == fake_states

    def test_restore_applies_cuda_states_when_present(self) -> None:
        original_set = _test_hooks.torch_cuda_set_rng_state_all
        applied: list[list[torch.Tensor]] = []

        def _fake_set(states: list[torch.Tensor]) -> None:
            applied.append(states)

        _test_hooks.torch_cuda_set_rng_state_all = _fake_set
        cuda_states = [torch.ones(4, dtype=torch.uint8)]
        try:
            restore_rng_states(
                RngStates(
                    torch_state=torch.get_rng_state(),
                    cuda_states=cuda_states,
                    python_state_json=capture_rng_states().python_state_json,
                )
            )
        finally:
            _test_hooks.torch_cuda_set_rng_state_all = original_set
        assert applied == [cuda_states]
