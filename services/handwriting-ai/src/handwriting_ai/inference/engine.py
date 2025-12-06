from __future__ import annotations

import pickle
import threading
import zipfile
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Final, NamedTuple, Protocol, Self, TypeGuard

import torch
from platform_core.json_utils import JSONValue, load_json_str
from platform_core.logging import get_logger
from platform_ml import ArtifactStore
from torch import Tensor

from ..config import Settings, ensure_settings
from .manifest import ModelManifest, from_path_manifest
from .types import PredictOutput

_LOAD_ERRORS: Final[tuple[type[BaseException], ...]] = (
    OSError,
    ValueError,
    RuntimeError,
    TypeError,
    EOFError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


class InferenceEngine:
    """Bounded thread-pool inference engine with Torch CPU model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = ensure_settings(settings, create_dirs=False)
        self._logger = get_logger("handwriting_ai")
        self._pool = _make_pool(self._settings)
        self._model_lock = threading.RLock()
        self._model: TorchModel | None = None
        self._manifest: ModelManifest | None = None
        self._artifacts_dir: Path | None = None
        self._last_manifest_mtime: float | None = None
        self._last_model_mtime: float | None = None
        torch.set_num_threads(1)

    @property
    def ready(self) -> bool:
        return self._model is not None and self._manifest is not None

    @property
    def model_id(self) -> str | None:
        return self._manifest["model_id"] if self._manifest is not None else None

    @property
    def manifest(self) -> ModelManifest | None:
        return self._manifest

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        return self._pool.submit(self._predict_impl, preprocessed)

    def _predict_impl(self, preprocessed: Tensor) -> PredictOutput:
        # Lazy failure if model not ready
        man = self._manifest
        model_obj = self._model
        if man is None or model_obj is None:
            raise RuntimeError("Model not loaded")

        # Defer Tensor handling to Torch with type as object to satisfy typing constraints
        tensor = _as_torch_tensor(preprocessed)
        # Test-time augmentation: average predictions across small shifts
        batch = _augment_for_tta(tensor) if self._settings["digits"]["tta"] else tensor
        model_obj.eval()
        with torch.no_grad():
            logits_obj = model_obj(batch)
        temperature = float(man["temperature"])
        probs_vec = _softmax_avg(logits_obj, temperature)
        probs_py = tuple(float(x) for x in probs_vec)
        top_idx = 0
        best = probs_py[0]
        for i in range(1, len(probs_py)):
            if probs_py[i] > best:
                best = probs_py[i]
                top_idx = i
        conf = float(probs_py[top_idx])
        model_id_str = man["model_id"]
        return {"digit": top_idx, "confidence": conf, "probs": probs_py, "model_id": model_id_str}

    def _ensure_artifacts_present(self, model_dir: Path) -> tuple[Path, Path] | None:
        """Ensure manifest and model files exist, fetching remotely if needed.

        Returns (manifest_path, model_path) if both exist, None otherwise.
        """
        manifest_path = model_dir / "manifest.json"
        model_path = model_dir / "model.pt"

        if not model_dir.exists():
            return None

        if not (manifest_path.exists() and model_path.exists()):
            if manifest_path.exists():
                self._download_remote_if_needed(model_dir, manifest_path)
            if not (manifest_path.exists() and model_path.exists()):
                return None

        return manifest_path, model_path

    def _load_and_validate_model(self, manifest: ModelManifest, model_path: Path) -> TorchModel:
        """Build model architecture and load state dict with validation."""
        try:
            model = _build_model(arch=manifest["arch"], n_classes=int(manifest["n_classes"]))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            self._logger.error("model_build_failed error=%s", exc)
            raise
        try:
            sd = _load_state_dict_file(model_path)
        except _LOAD_ERRORS as e:
            get_logger("handwriting_ai").error(
                "state_dict_load_failed exc=%s msg=%s", e.__class__.__name__, str(e)
            )
            raise
        try:
            _validate_state_dict(sd, manifest["arch"], int(manifest["n_classes"]))
            model.load_state_dict(sd)
        except ValueError as exc:
            self._logger.error("state_dict_invalid error=%s", exc)
            raise
        return model

    def try_load_active(self) -> None:
        active = self._settings["digits"]["active_model"]
        model_dir = self._settings["digits"]["model_dir"] / active

        paths = self._ensure_artifacts_present(model_dir)
        if paths is None:
            return
        manifest_path, model_path = paths

        try:
            manifest = from_path_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            self._logger.error("manifest_load_failed error=%s", exc)
            raise

        # Validate preprocess signature compatibility
        from ..preprocess import preprocess_signature

        if manifest["preprocess_hash"] != preprocess_signature():
            return

        model = self._load_and_validate_model(manifest, model_path)

        with self._model_lock:
            self._model = model
            self._manifest = manifest
            self._artifacts_dir = model_dir
            try:
                m1, m2 = _collect_artifact_mtimes(manifest_path, model_path)
                self._last_manifest_mtime = m1
                self._last_model_mtime = m2
            except OSError as exc:
                self._logger.error("artifact_mtime_unavailable error=%s", exc)
                raise

    def _download_remote_if_needed(self, model_dir: Path, manifest_path: Path) -> None:
        # Load raw JSON and detect v2 with file_id
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._logger.error("manifest_read_failed error=%s", exc)
            raise
        value: JSONValue = load_json_str(raw_text)
        if not isinstance(value, dict):
            raise ValueError("manifest must be a JSON object for remote fetch")
        schema = str(value.get("schema_version", "")).strip()
        if schema != "v2.0":
            return  # Only v2 manifests support remote fetching
        file_id_val = value.get("file_id")
        if not isinstance(file_id_val, str) or file_id_val.strip() == "":
            raise RuntimeError("v2 manifest missing file_id; cannot fetch remote artifact")
        # Require data-bank config from settings
        app = self._settings["app"]
        api_url = str(app.get("data_bank_api_url", ""))
        api_key = str(app.get("data_bank_api_key", ""))
        if api_url.strip() == "" or api_key.strip() == "":
            raise RuntimeError("missing data-bank-api configuration for remote download")
        store = ArtifactStore(api_url, api_key)
        expected_root = model_dir.name
        store.download_artifact(
            file_id_val,
            dest_dir=model_dir.parent,
            request_id="handwriting-engine-bootstrap",
            expected_root=expected_root,
        )

    def reload_if_changed(self) -> bool:
        """Reload active model if manifest or weights changed on disk.

        Returns True if a reload occurred and engine remains ready; False otherwise.
        """
        art = self._artifacts_dir
        if art is None:
            return False
        manifest_path = art / "manifest.json"
        model_path = art / "model.pt"
        m1, m2 = _collect_artifact_mtimes(manifest_path, model_path)
        if self._last_manifest_mtime is None or self._last_model_mtime is None:
            return False
        if m1 <= self._last_manifest_mtime and m2 <= self._last_model_mtime:
            return False

        # Avoid reading while writer is updating the manifest: require stable size.
        size1 = manifest_path.stat().st_size
        if size1 <= 0:
            return False
        size2 = manifest_path.stat().st_size
        if size2 != size1:
            return False

        self.try_load_active()
        return self.ready


def _make_pool(settings: Settings) -> ThreadPoolExecutor:
    if settings["app"]["threads"] == 0:
        import os

        cpu_count = os.cpu_count() or 1
        size = min(8, cpu_count)
    else:
        size = settings["app"]["threads"]
    return ThreadPoolExecutor(max_workers=size, thread_name_prefix="predict")


class LoadStateResult(NamedTuple):
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


class TorchModel(Protocol):
    def eval(self) -> Self: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult: ...
    def train(self, mode: bool = True) -> Self: ...
    def state_dict(self) -> dict[str, Tensor]: ...
    def parameters(self) -> Sequence[torch.nn.Parameter]: ...


class _RawLoadStateResult(Protocol):
    @property
    def missing_keys(self) -> Sequence[str]: ...

    @property
    def unexpected_keys(self) -> Sequence[str]: ...


class _TypedModule(Protocol):
    def eval(self) -> torch.nn.Module: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def load_state_dict(self, sd: dict[str, Tensor]) -> _RawLoadStateResult: ...
    def train(self, mode: bool = True) -> torch.nn.Module: ...
    def state_dict(self) -> dict[str, Tensor]: ...


def _coerce_state_result(res: _RawLoadStateResult) -> LoadStateResult:
    missing_keys = _normalize_keys(res.missing_keys, "missing_keys")
    unexpected_keys = _normalize_keys(res.unexpected_keys, "unexpected_keys")
    return LoadStateResult(missing_keys=missing_keys, unexpected_keys=unexpected_keys)


def _normalize_keys(keys: Sequence[str], label: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for entry in keys:
        if not isinstance(entry, str):
            raise RuntimeError(f"{label} entry is not a string")
        normalized.append(entry)
    return tuple(normalized)


class _WrappedTorchModel:
    """Adapter that enforces strict typing on torch.nn.Module interactions."""

    def __init__(self, module: torch.nn.Module) -> None:
        typed_module: _TypedModule = module
        self._module = typed_module
        self._raw_module = module

    def eval(self) -> Self:
        self._module.eval()
        return self

    def __call__(self, x: Tensor) -> Tensor:
        out = self._module(x)
        if not (hasattr(out, "dtype") and hasattr(out, "shape")):
            raise RuntimeError("model forward did not return Tensor-like output")
        return out

    def load_state_dict(self, sd: dict[str, Tensor]) -> LoadStateResult:
        res = self._module.load_state_dict(sd)
        return _coerce_state_result(res)

    def train(self, mode: bool = True) -> Self:
        self._module.train(mode)
        return self

    def state_dict(self) -> dict[str, Tensor]:
        raw = self._module.state_dict()
        typed: dict[str, Tensor] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise RuntimeError("state_dict key must be str")
            if not (hasattr(value, "dtype") and hasattr(value, "shape")):
                raise RuntimeError("state_dict value must be Tensor")
            typed[key] = value
        return typed

    def parameters(self) -> Sequence[torch.nn.Parameter]:
        return tuple(self._raw_module.parameters())


class _ModelsModule(Protocol):
    @property
    def resnet18(self) -> _ResNet18Builder: ...


class _ResNet18Builder(Protocol):
    def __call__(self, *, weights: None, num_classes: int) -> torch.nn.Module: ...


def _build_model(arch: str, n_classes: int) -> TorchModel:
    import importlib

    import torch.nn as nn

    models_mod: _ModelsModule = importlib.import_module("torchvision.models")
    resnet_attr = models_mod.resnet18
    if not callable(resnet_attr):
        raise RuntimeError(f"torchvision.models.{arch} is not callable")
    resnet_fn: _ResNet18Builder = resnet_attr
    module: nn.Module = resnet_fn(weights=None, num_classes=int(n_classes))
    if not hasattr(module, "_modules"):
        raise RuntimeError("model builder did not return a torch.nn.Module")
    # CIFAR-style stem and 1-channel - require these attributes for valid ResNet structure
    if not hasattr(module, "conv1"):
        raise RuntimeError("model missing required conv1 attribute")
    if not hasattr(module, "maxpool"):
        raise RuntimeError("model missing required maxpool attribute")
    conv1_new = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    maxpool_new = nn.Identity()
    # Use _modules dict to properly register submodules for state_dict() serialization
    module._modules["conv1"] = conv1_new
    module._modules["maxpool"] = maxpool_new
    return _WrappedTorchModel(module)


def build_fresh_state_dict(arch: str, n_classes: int) -> dict[str, Tensor]:
    m = _build_model(arch=arch, n_classes=n_classes)
    sd_obj = m.state_dict()
    if not isinstance(sd_obj, dict):
        raise RuntimeError("state_dict() did not return a dict")
    out: dict[str, Tensor] = {}
    for k, v in sd_obj.items():
        if isinstance(k, str) and hasattr(v, "shape") and hasattr(v, "dtype"):
            out[k] = v
        else:
            raise RuntimeError("invalid state dict entry from model")
    return out


def _as_torch_tensor(x: Tensor) -> Tensor:
    t = x
    if t.ndim == 3:
        # Expect 1x28x28 -> add batch
        t = t.unsqueeze(0)
    return t.to(dtype=torch.float32)


def _softmax_avg(logits: Tensor, temperature: float) -> list[float]:
    logits_t = logits / temperature
    probs = torch.softmax(logits_t, dim=1)
    mean_probs = probs.mean(dim=0) if probs.ndim == 2 and probs.shape[0] > 1 else probs[0]
    n = int(mean_probs.shape[0])
    return [float(mean_probs[i].item()) for i in range(n)]


def _augment_for_tta(x: Tensor) -> Tensor:
    # Input is 4D (1,1,28,28)
    if x.ndim != 4:
        return x
    # Identity + small shifts
    batch: list[Tensor] = [x]
    batch.append(torch.roll(x, shifts=(0, 1), dims=(2, 3)))  # right by 1
    batch.append(torch.roll(x, shifts=(0, -1), dims=(2, 3)))  # left by 1
    batch.append(torch.roll(x, shifts=(1, 0), dims=(2, 3)))  # down by 1
    batch.append(torch.roll(x, shifts=(-1, 0), dims=(2, 3)))  # up by 1
    # Add small rotations for variety
    for ang in (-6.0, -3.0, 3.0, 6.0):
        batch.append(_rotate_tensor(x, degrees=float(ang)))
    return torch.cat(batch, dim=0)


def _rotate_tensor(x: Tensor, degrees: float) -> Tensor:
    import math

    import torch.nn.functional as functional

    # x: (B=1, C=1, H, W)
    b, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    theta = torch.zeros((b, 2, 3), dtype=x.dtype, device=x.device)
    rad = float(degrees) * math.pi / 180.0
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    theta[:, 0, 0] = float(cos_a)
    theta[:, 0, 1] = float(-sin_a)
    theta[:, 1, 0] = float(sin_a)
    theta[:, 1, 1] = float(cos_a)
    size_list: list[int] = [b, c, h, w]
    grid = functional.affine_grid(theta, size=size_list, align_corners=False)
    return functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


LoadedStateDict = dict[str, Tensor]
WrappedStateDict = dict[str, LoadedStateDict]


class _TorchLoadFn(Protocol):
    def __call__(
        self, f: str, *, map_location: torch.device, weights_only: bool
    ) -> LoadedStateDict | WrappedStateDict: ...


def _is_wrapped_state_dict(
    value: LoadedStateDict | WrappedStateDict,
) -> TypeGuard[WrappedStateDict]:
    return set(value.keys()) == {"state_dict"}


def _is_flat_state_dict(
    value: LoadedStateDict | WrappedStateDict,
) -> TypeGuard[LoadedStateDict]:
    return not _is_wrapped_state_dict(value)


def _load_state_dict_file(path: Path) -> dict[str, Tensor]:
    # Load state dict saved by our trainer (direct format: dict[str, Tensor])
    torch_mod = __import__("torch")
    load_fn: _TorchLoadFn = torch_mod.load
    loaded_raw = load_fn(path.as_posix(), map_location=torch.device("cpu"), weights_only=True)
    # Validate loaded is a dict
    if not isinstance(loaded_raw, dict):
        raise ValueError("state dict file does not contain a dict")
    if _is_wrapped_state_dict(loaded_raw):
        nested = loaded_raw["state_dict"]
        if not isinstance(nested, dict):
            raise ValueError("state_dict wrapper must contain a dict")
        sd_source: LoadedStateDict = nested
    elif _is_flat_state_dict(loaded_raw):
        sd_source = loaded_raw
    else:
        raise ValueError("state dict file does not contain tensors")
    # Validate all entries are properly typed
    out: dict[str, Tensor] = {}
    for k, v in sd_source.items():
        if not isinstance(k, str):
            raise ValueError("invalid state dict entry")
        if not hasattr(v, "shape") or not hasattr(v, "dtype"):
            raise ValueError("invalid state dict entry")
        out[k] = v
    return out


def _validate_state_dict(sd: dict[str, Tensor], arch: str, n_classes: int) -> None:
    w = sd.get("fc.weight")
    b = sd.get("fc.bias")
    if w is None or b is None:
        raise ValueError("missing classifier weights in state dict")
    if w.ndim != 2 or b.ndim != 1:
        raise ValueError("invalid classifier tensor dimensions")
    if int(w.shape[0]) != n_classes or int(b.shape[0]) != n_classes:
        raise ValueError("classifier head size does not match n_classes")
    # ResNet-18 expected feature dimension
    expected_in = 512
    if int(w.shape[1]) != expected_in:
        raise ValueError("classifier head in_features does not match backbone")
    # Minimal backbone invariants for resnet18 CIFAR-style stem
    conv1 = sd.get("conv1.weight")
    if conv1 is None or conv1.ndim != 4:
        raise ValueError("missing or invalid conv1.weight")
    if int(conv1.shape[0]) != 64 or int(conv1.shape[1]) != 1:
        raise ValueError("unexpected conv1 shape for 1-channel stem")
    # Expect presence of top-level batch norm
    if "bn1.weight" not in sd or "bn1.bias" not in sd:
        raise ValueError("missing bn1 parameters")
    # Ensure main layers exist
    has_layers = all(any(k.startswith(f"layer{i}.") for k in sd) for i in range(1, 5))
    if not has_layers:
        raise ValueError("missing resnet layer blocks")


def _collect_artifact_mtimes(manifest_path: Path, model_path: Path) -> tuple[float, float]:
    """Collect mtimes with bounded repeated stat() calls.

    Always performs a fixed small number of stat() reads to ensure deterministic
    surfacing of delayed stat failures in tests and to mitigate transient
    filesystem timing anomalies. All OSErrors propagate to callers.
    """
    last_m1: float | None = None
    last_m2: float | None = None
    for _ in range(16):
        last_m1 = manifest_path.stat().st_mtime
        last_m2 = model_path.stat().st_mtime
    return (
        last_m1 if last_m1 is not None else manifest_path.stat().st_mtime,
        last_m2 if last_m2 is not None else model_path.stat().st_mtime,
    )
