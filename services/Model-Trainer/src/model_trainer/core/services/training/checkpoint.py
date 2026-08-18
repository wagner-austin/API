"""Persistence for training checkpoints.

One rolling checkpoint file per run, written after every completed epoch
and replaced atomically, so an interrupted run can be resumed from its
last epoch boundary instead of from scratch. The file is a single torch
payload holding the JSON-encoded :class:`TrainingCheckpointMeta` beside
the tensor state (model, optimizer, RNG), which
makes the write atomic as a unit: ``os.replace`` either publishes a
complete checkpoint or leaves the previous one untouched.

The file lives under ``checkpoints_dir`` rather than the run's model
directory, so the artifact uploader and the post-completion cleanup
service never see it. A successful run deletes its checkpoint at the
end; a failed or cancelled run leaves it in place, and that file is
exactly what ``POST /runs/{run_id}/resume`` continues from.

Loading is strict: a missing file, an unreadable payload, a schema
version this code does not understand, or a file whose recorded run id
disagrees with its name each raise a distinct, typed error. Nothing is
recovered from silently; a corrupt checkpoint means the operator reruns
from scratch, explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeGuard

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_int,
    require_list,
)
from platform_core.logging import get_logger

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingCheckpointMeta,
    decode_training_checkpoint_meta,
    encode_training_checkpoint_meta,
)
from model_trainer.core.infra.paths import checkpoint_path, checkpoints_dir
from model_trainer.core.types import TorchStateValue

_logger = get_logger(__name__)

#: torch.Tensor laundered through an annotated assignment: the bare class
#: expression carries Any under disallow_any_expr, the annotated constant
#: does not.
_TENSOR_CLASS: type[torch.Tensor] = torch.Tensor


def _is_tensor_value(value: TorchStateValue) -> TypeGuard[torch.Tensor]:
    """Report whether a loaded payload value is a torch tensor.

    Args:
        value: Value taken from a loaded checkpoint payload.

    Returns:
        True when the value is a tensor, narrowing its static type.
    """
    return isinstance(value, _TENSOR_CLASS)


def _corrupt(run_id: str, reason: str) -> AppError[ModelTrainerErrorCode]:
    """Build the typed error for an unusable checkpoint payload.

    Args:
        run_id: Run whose checkpoint failed to load.
        reason: What was wrong with the payload.

    Returns:
        AppError carrying ``CHECKPOINT_CORRUPT``.
    """
    return AppError(
        ModelTrainerErrorCode.CHECKPOINT_CORRUPT,
        f"checkpoint for run '{run_id}' is unusable: {reason}",
        model_trainer_status_for(ModelTrainerErrorCode.CHECKPOINT_CORRUPT),
    )


class RngStates:
    """Captured random-number-generator states for exact continuation.

    Attributes:
        torch_state: CPU generator state from ``torch.get_rng_state``.
        cuda_states: Per-device generator states from
            ``torch.cuda.get_rng_state_all``; empty on CPU-only hosts.
        python_state_json: ``random.getstate`` encoded as a JSON string.
    """

    __slots__ = ("cuda_states", "python_state_json", "torch_state")

    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor]
    python_state_json: str

    def __init__(
        self: RngStates,
        *,
        torch_state: torch.Tensor,
        cuda_states: list[torch.Tensor],
        python_state_json: str,
    ) -> None:
        """Initialise captured RNG states.

        Args:
            torch_state: CPU generator state.
            cuda_states: Per-device CUDA generator states.
            python_state_json: Encoded ``random.getstate`` result.
        """
        self.torch_state = torch_state
        self.cuda_states = cuda_states
        self.python_state_json = python_state_json


def _encode_python_rng_state() -> str:
    """Encode the current ``random`` module state as a JSON string.

    The state tuple's shape is validated rather than assumed, because a
    checkpoint that persists a malformed state would corrupt the resumed
    run's sampling silently.

    Returns:
        JSON string carrying version, internal state and gauss carry.

    Raises:
        RuntimeError: If the state tuple does not have the documented
            CPython shape.
    """
    state = _test_hooks.random_getstate()
    if len(state) != 3:
        raise RuntimeError(f"random.getstate returned {len(state)} entries; expected 3")
    version_raw = state[0]
    if isinstance(version_raw, bool) or not isinstance(version_raw, int):
        raise RuntimeError(f"random.getstate version is {type(version_raw).__name__}, not int")
    internal_raw = state[1]
    if not isinstance(internal_raw, tuple):
        raise RuntimeError(
            f"random.getstate internal state is {type(internal_raw).__name__}, not tuple"
        )
    internal_values: list[JSONValue] = []
    for value in internal_raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"random.getstate internal state holds {type(value).__name__}, not int"
            )
        internal_values.append(value)
    gauss_raw = state[2]
    gauss_next: float | None
    if gauss_raw is None:
        gauss_next = None
    elif isinstance(gauss_raw, float):
        gauss_next = gauss_raw
    else:
        raise RuntimeError(
            f"random.getstate gauss value is {type(gauss_raw).__name__}, not float or None"
        )
    payload: JSONObject = {
        "version": version_raw,
        "internal_state": internal_values,
        "gauss_next": gauss_next,
    }
    return dump_json_str(payload)


def _decode_python_rng_state(raw: str) -> tuple[TorchStateValue, ...]:
    """Decode a JSON string back into a ``random.setstate`` tuple.

    Args:
        raw: JSON string produced by :func:`_encode_python_rng_state`.

    Returns:
        The state tuple ``random.setstate`` accepts.

    Raises:
        JSONTypeError: If the string does not carry a valid state.
    """
    loaded = load_json_str(raw)
    if not isinstance(loaded, dict):
        raise JSONTypeError(f"python RNG state must be an object, got {type(loaded).__name__}")
    version = require_int(loaded, "version")
    internal_raw = require_list(loaded, "internal_state")
    internal: list[int] = []
    for index, item in enumerate(internal_raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise JSONTypeError(
                f"Field 'internal_state[{index}]' must be an integer, got {type(item).__name__}"
            )
        internal.append(item)
    gauss_raw: JSONValue = loaded.get("gauss_next")
    gauss_next: float | None
    if gauss_raw is None:
        gauss_next = None
    elif isinstance(gauss_raw, bool) or not isinstance(gauss_raw, int | float):
        raise JSONTypeError(
            f"Field 'gauss_next' must be a number or null, got {type(gauss_raw).__name__}"
        )
    else:
        gauss_next = float(gauss_raw)
    return (version, tuple(internal), gauss_next)


def capture_rng_states() -> RngStates:
    """Capture torch CPU, CUDA and python RNG states.

    Returns:
        The captured states, ready for checkpoint persistence.
    """
    cuda_states: list[torch.Tensor] = (
        _test_hooks.torch_cuda_get_rng_state_all() if _test_hooks.cuda_is_available() else []
    )
    return RngStates(
        torch_state=torch.get_rng_state(),
        cuda_states=cuda_states,
        python_state_json=_encode_python_rng_state(),
    )


def restore_rng_states(states: RngStates) -> None:
    """Restore torch CPU, CUDA and python RNG states.

    CUDA states are restored only when the checkpoint carries them; a
    checkpoint written on a CPU-only host carries an empty list, and the
    config fingerprint guarantees the resumed run targets the same
    device kind as the original.

    Args:
        states: States captured by :func:`capture_rng_states`.

    Raises:
        JSONTypeError: If the python RNG state string is invalid.
    """
    torch.set_rng_state(states.torch_state)
    if states.cuda_states:
        _test_hooks.torch_cuda_set_rng_state_all(states.cuda_states)
    _test_hooks.random_setstate(_decode_python_rng_state(states.python_state_json))


class TrainingCheckpoint:
    """A complete checkpoint: typed metadata plus tensor state.

    Attributes:
        meta: JSON-encodable progress and fingerprint metadata.
        model_state: Model parameters and buffers by name.
        optimizer_state: Optimizer state as returned by ``state_dict``.
        rng: Captured RNG states.

    The fp16 GradScaler is deliberately absent: the training loop creates
    a fresh scaler at each epoch start, so a resumed epoch begins exactly
    as an uninterrupted next epoch would.
    """

    __slots__ = ("meta", "model_state", "optimizer_state", "rng")

    meta: TrainingCheckpointMeta
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, TorchStateValue]
    rng: RngStates

    def __init__(
        self: TrainingCheckpoint,
        *,
        meta: TrainingCheckpointMeta,
        model_state: dict[str, torch.Tensor],
        optimizer_state: dict[str, TorchStateValue],
        rng: RngStates,
    ) -> None:
        """Initialise a checkpoint.

        Args:
            meta: Progress and fingerprint metadata.
            model_state: Model parameters and buffers by name.
            optimizer_state: Optimizer ``state_dict`` result.
            rng: Captured RNG states.
        """
        self.meta = meta
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.rng = rng


def checkpoint_exists(settings: Settings, run_id: str) -> bool:
    """Report whether a checkpoint file exists for the run.

    Args:
        settings: Application settings locating the artifacts root.
        run_id: Run to look up.

    Returns:
        True when the checkpoint file is present.
    """
    return checkpoint_path(settings, run_id).is_file()


def save_training_checkpoint(settings: Settings, checkpoint: TrainingCheckpoint) -> Path:
    """Write a checkpoint atomically, replacing any previous one.

    The payload is written to a sibling temporary file and published
    with ``os.replace``, so a crash mid-write leaves the previous
    checkpoint intact and readable.

    Args:
        settings: Application settings locating the artifacts root.
        checkpoint: The checkpoint to persist.

    Returns:
        Path of the published checkpoint file.
    """
    run_id = checkpoint.meta["run_id"]
    target = checkpoint_path(settings, run_id)
    checkpoints_dir(settings).mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload: dict[str, TorchStateValue] = {
        "meta_json": dump_json_str(encode_training_checkpoint_meta(checkpoint.meta)),
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "rng_torch": checkpoint.rng.torch_state,
        "rng_cuda": checkpoint.rng.cuda_states,
        "rng_python_json": checkpoint.rng.python_state_json,
    }
    torch.save(payload, str(temporary))
    os.replace(temporary, target)
    _logger.info(
        "Checkpoint saved",
        extra={
            "category": "training",
            "event": "training_checkpoint_saved",
            "run_id": run_id,
            "epochs_completed": checkpoint.meta["epochs_completed"],
            "global_step": checkpoint.meta["global_step"],
            "path": str(target),
        },
    )
    return target


def _payload_entry(payload: dict[str, TorchStateValue], key: str, run_id: str) -> TorchStateValue:
    """Fetch a required entry from a loaded checkpoint payload.

    Args:
        payload: The loaded torch payload.
        key: Entry name.
        run_id: Run for error attribution.

    Returns:
        The entry value.

    Raises:
        AppError: With ``CHECKPOINT_CORRUPT`` when the entry is absent.
    """
    if key not in payload:
        raise _corrupt(run_id, f"payload entry '{key}' is missing")
    return payload[key]


def _payload_str(payload: dict[str, TorchStateValue], key: str, run_id: str) -> str:
    """Fetch a required string entry from a loaded checkpoint payload.

    Args:
        payload: The loaded torch payload.
        key: Entry name.
        run_id: Run for error attribution.

    Returns:
        The string value.

    Raises:
        AppError: With ``CHECKPOINT_CORRUPT`` when the entry is absent or
            not a string.
    """
    value = _payload_entry(payload, key, run_id)
    if not isinstance(value, str):
        raise _corrupt(
            run_id, f"payload entry '{key}' must be a string, got {type(value).__name__}"
        )
    return value


def _payload_tensor(payload: dict[str, TorchStateValue], key: str, run_id: str) -> torch.Tensor:
    """Fetch a required tensor entry from a loaded checkpoint payload.

    Args:
        payload: The loaded torch payload.
        key: Entry name.
        run_id: Run for error attribution.

    Returns:
        The tensor value.

    Raises:
        AppError: With ``CHECKPOINT_CORRUPT`` when the entry is absent or
            not a tensor.
    """
    value = _payload_entry(payload, key, run_id)
    if not _is_tensor_value(value):
        raise _corrupt(
            run_id, f"payload entry '{key}' must be a tensor, got {type(value).__name__}"
        )
    return value


def _payload_tensor_dict(
    payload: dict[str, TorchStateValue], key: str, run_id: str
) -> dict[str, torch.Tensor]:
    """Fetch a required name-to-tensor mapping from a loaded payload.

    Args:
        payload: The loaded torch payload.
        key: Entry name.
        run_id: Run for error attribution.

    Returns:
        The validated mapping.

    Raises:
        AppError: With ``CHECKPOINT_CORRUPT`` when the entry is absent,
            not a mapping, or holds a non-tensor value.
    """
    value = _payload_entry(payload, key, run_id)
    if not isinstance(value, dict):
        raise _corrupt(
            run_id, f"payload entry '{key}' must be a mapping, got {type(value).__name__}"
        )
    out: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str):
            raise _corrupt(run_id, f"payload entry '{key}' holds a non-string key")
        if not _is_tensor_value(tensor):
            raise _corrupt(
                run_id,
                f"payload entry '{key}[{name}]' must be a tensor, got {type(tensor).__name__}",
            )
        out[name] = tensor
    return out


def _payload_object_dict(
    payload: dict[str, TorchStateValue], key: str, run_id: str
) -> dict[str, TorchStateValue]:
    """Fetch a required string-keyed mapping from a loaded payload.

    Args:
        payload: The loaded torch payload.
        key: Entry name.
        run_id: Run for error attribution.

    Returns:
        The validated mapping.

    Raises:
        AppError: With ``CHECKPOINT_CORRUPT`` when the entry is absent,
            not a mapping, or holds a non-string key.
    """
    value = _payload_entry(payload, key, run_id)
    if not isinstance(value, dict):
        raise _corrupt(
            run_id, f"payload entry '{key}' must be a mapping, got {type(value).__name__}"
        )
    out: dict[str, TorchStateValue] = {}
    for name, item in value.items():
        if not isinstance(name, str):
            raise _corrupt(run_id, f"payload entry '{key}' holds a non-string key")
        out[name] = item
    return out


def _payload_tensor_list(
    payload: dict[str, TorchStateValue], key: str, run_id: str
) -> list[torch.Tensor]:
    """Fetch a required tensor list from a loaded payload.

    Args:
        payload: The loaded torch payload.
        key: Entry name.
        run_id: Run for error attribution.

    Returns:
        The validated list.

    Raises:
        AppError: With ``CHECKPOINT_CORRUPT`` when the entry is absent,
            not a list, or holds a non-tensor.
    """
    value = _payload_entry(payload, key, run_id)
    if not isinstance(value, list):
        raise _corrupt(run_id, f"payload entry '{key}' must be a list, got {type(value).__name__}")
    entries: list[TorchStateValue] = list(value)
    out: list[torch.Tensor] = []
    for index, item in enumerate(entries):
        if not _is_tensor_value(item):
            raise _corrupt(
                run_id,
                f"payload entry '{key}[{index}]' must be a tensor, got {type(item).__name__}",
            )
        out.append(item)
    return out


def load_training_checkpoint(settings: Settings, run_id: str) -> TrainingCheckpoint:
    """Load and validate the checkpoint for a run.

    Args:
        settings: Application settings locating the artifacts root.
        run_id: Run whose checkpoint to load.

    Returns:
        The validated checkpoint.

    Raises:
        AppError: With ``CHECKPOINT_NOT_FOUND`` when no checkpoint file
            exists; ``CHECKPOINT_CORRUPT`` when the payload shape or the
            metadata is invalid or the recorded run id disagrees with the
            requested one; ``CHECKPOINT_SCHEMA_UNSUPPORTED`` when the
            file was written by a different schema version.
    """
    path = checkpoint_path(settings, run_id)
    if not path.is_file():
        raise AppError(
            ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND,
            f"no checkpoint exists for run '{run_id}' at {path}",
            model_trainer_status_for(ModelTrainerErrorCode.CHECKPOINT_NOT_FOUND),
        )

    loaded: TorchStateValue = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise _corrupt(run_id, f"payload must be a mapping, got {type(loaded).__name__}")
    payload: dict[str, TorchStateValue] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise _corrupt(run_id, "payload holds a non-string key")
        payload[key] = value

    meta_json = _payload_str(payload, "meta_json", run_id)
    meta_raw = load_json_str(meta_json)
    if not isinstance(meta_raw, dict):
        raise _corrupt(run_id, f"meta_json must decode to an object, got {type(meta_raw).__name__}")
    try:
        meta = decode_training_checkpoint_meta(meta_raw)
    except JSONTypeError as error:
        raise _corrupt(run_id, f"metadata failed validation: {error}") from error

    if meta["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise AppError(
            ModelTrainerErrorCode.CHECKPOINT_SCHEMA_UNSUPPORTED,
            (
                f"checkpoint for run '{run_id}' has schema version "
                f"{meta['schema_version']}; this build reads version "
                f"{CHECKPOINT_SCHEMA_VERSION}"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CHECKPOINT_SCHEMA_UNSUPPORTED),
        )
    if meta["run_id"] != run_id:
        raise _corrupt(run_id, f"metadata records run id '{meta['run_id']}'")

    checkpoint = TrainingCheckpoint(
        meta=meta,
        model_state=_payload_tensor_dict(payload, "model_state", run_id),
        optimizer_state=_payload_object_dict(payload, "optimizer_state", run_id),
        rng=RngStates(
            torch_state=_payload_tensor(payload, "rng_torch", run_id),
            cuda_states=_payload_tensor_list(payload, "rng_cuda", run_id),
            python_state_json=_payload_str(payload, "rng_python_json", run_id),
        ),
    )
    _logger.info(
        "Checkpoint loaded",
        extra={
            "category": "training",
            "event": "training_checkpoint_loaded",
            "run_id": run_id,
            "epochs_completed": meta["epochs_completed"],
            "global_step": meta["global_step"],
            "path": str(path),
        },
    )
    return checkpoint


def delete_training_checkpoint(settings: Settings, run_id: str) -> bool:
    """Delete the run's checkpoint file if one exists.

    Called when a run completes: the final model and manifest supersede
    the resume state. A run that never reached its first epoch boundary
    has no checkpoint, which is a legitimate state rather than an error.

    Args:
        settings: Application settings locating the artifacts root.
        run_id: Run whose checkpoint to delete.

    Returns:
        True when a file was deleted, False when none existed.
    """
    path = checkpoint_path(settings, run_id)
    if not path.is_file():
        return False
    path.unlink()
    _logger.info(
        "Checkpoint deleted",
        extra={
            "category": "training",
            "event": "training_checkpoint_deleted",
            "run_id": run_id,
            "path": str(path),
        },
    )
    return True


__all__ = [
    "RngStates",
    "TrainingCheckpoint",
    "capture_rng_states",
    "checkpoint_exists",
    "delete_training_checkpoint",
    "load_training_checkpoint",
    "restore_rng_states",
    "save_training_checkpoint",
]
