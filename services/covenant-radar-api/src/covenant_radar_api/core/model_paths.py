"""Containment-checked resolution of caller-supplied model paths.

`model_path` arrives on request bodies for /ml/explain, /ml/explain-regression
and /ml/predict-regression, and flows into pickle-backed loaders (torch.load,
joblib.load) and file-reading loaders (xgboost.load_model, lgb.Booster). An
unconstrained path therefore selects which file on the host those loaders open.

Every caller-supplied path must pass through :func:`resolve_model_path` before
it reaches a loader, so that a model can only ever be read from the configured
models root.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(raw_path: str, models_root: Path) -> Path:
    """Resolve a caller-supplied model path, confined to the models root.

    Symlinks and `..` segments are resolved before the containment check, so a
    path cannot traverse out of the root and cannot point at a link whose
    target sits outside it.

    Args:
        raw_path: Model path exactly as supplied by the caller.
        models_root: Directory that every loadable model must live under.

    Returns:
        The resolved absolute path, guaranteed to sit under `models_root`.

    Raises:
        ValueError: If the resolved path is outside `models_root`.
    """
    resolved = Path(raw_path).resolve()
    root = models_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"model_path must resolve inside the models root: {raw_path!r} "
            f"resolves to {resolved}, which is outside {root}"
        )
    return resolved


__all__ = ["resolve_model_path"]
