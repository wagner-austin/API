"""Grouped k-fold cross-validation for external datasets.

A single 70/15/15 split of a grouped dataset answers with one number and a
shrug: at 75 matches the test fold holds ~11 of them, and the measured AUC
swung 0.79 -> 0.63 between two exports for no modelling reason at all. The
honest instrument at low group counts is k-fold over WHOLE GROUPS -- the
splitter has existed in ``covenant_ml.validation`` since the AMEX work; this
script is the runner that finally points it at a registered dataset.

Every fold trains a fresh model on k-1 folds of groups (the backend's inner
early-stopping split is grouped too) and scores AUC on the held-out fold's
rows, which no training in that fold has seen. The report is per-fold AUC
plus mean and standard deviation -- the spread IS the finding, not noise to
hide.

Run as ``python -m scripts.cv_external <dataset> <backend> [folds] [seed]``
with backend ``cleargbm`` or ``lightgbm``.
"""

from __future__ import annotations

import math
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.metrics import compute_auc
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    ClearGBMConfig,
    LightGBMConfig,
)
from covenant_ml.validation.splitter import group_stratified_kfold_split
from numpy.typing import NDArray

from covenant_radar_api.worker import _test_hooks as hooks

EXIT_OK = 0
EXIT_BAD_USAGE = 2

DEFAULT_FOLDS = 5
DEFAULT_SEED = 42

#: Inner split for the backend's own early stopping, applied to each fold's
#: training groups. A zero test share would hand the backend an empty test
#: array to compute metrics on; 15% is the price of not special-casing the
#: backends, and the held-out fold is what this script actually scores.
INNER_RATIOS = (0.70, 0.15, 0.15)


def _cleargbm_config(seed: int) -> ClearGBMConfig:
    """Fixed ClearGBM hyperparameters for the evaluation protocol.

    Deliberately not tunable from the command line: this script measures the
    evaluation spread, and a per-run hyperparameter surface would turn every
    comparison into a two-variable experiment.

    Args:
        seed: Random seed for the fold's training.

    Returns:
        The training configuration.
    """
    return ClearGBMConfig(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=None,
        max_bins=64,
        subsample=0.8,
        random_state=seed,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=-1,
        train_ratio=INNER_RATIOS[0],
        val_ratio=INNER_RATIOS[1],
        test_ratio=INNER_RATIOS[2],
        early_stopping_rounds=10,
    )


def _lightgbm_config(seed: int) -> LightGBMConfig:
    """Fixed LightGBM hyperparameters, mirroring the ClearGBM protocol.

    Args:
        seed: Random seed for the fold's training.

    Returns:
        The training configuration.
    """
    return LightGBMConfig(
        device="cpu",
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        train_ratio=INNER_RATIOS[0],
        val_ratio=INNER_RATIOS[1],
        test_ratio=INNER_RATIOS[2],
        random_state=seed,
        early_stopping_rounds=10,
    )


def _config_for(backend: str, seed: int) -> tuple[BackendName, ClassifierTrainConfig] | None:
    """Resolve a supported backend name and its fixed config.

    Args:
        backend: The requested backend.
        seed: Random seed for training.

    Returns:
        The typed backend name and configuration, or None for a backend
        this protocol does not cover.
    """
    if backend == "cleargbm":
        return "cleargbm", _cleargbm_config(seed)
    if backend == "lightgbm":
        return "lightgbm", _lightgbm_config(seed)
    return None


def main(
    argv: Sequence[str] | None = None,
    external_dir: Path = Path("data/external"),
) -> int:
    """Run grouped k-fold CV for the dataset and backend named on the CLI.

    Args:
        argv: ``<dataset> <backend> [folds] [seed]``. ``None`` reads the
            process arguments.
        external_dir: Root directory for datasets, a parameter so a test can
            point it at a scratch tree.

    Returns:
        ``EXIT_OK`` on a completed evaluation, ``EXIT_BAD_USAGE`` on a bad
        argument shape, an unsupported backend, or a dataset without groups.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (2, 3, 4):
        sys.stdout.write("usage: cv_external <dataset> <backend> [folds] [seed]\n")
        return EXIT_BAD_USAGE
    dataset_name, backend = args[0], args[1]
    n_folds = int(args[2]) if len(args) >= 3 else DEFAULT_FOLDS
    seed = int(args[3]) if len(args) == 4 else DEFAULT_SEED

    registry = hooks.dataset_registry_factory()
    if dataset_name not in registry:
        available = ", ".join(registry.list_names())
        sys.stdout.write(f"dataset must be one of: {available} (got {dataset_name})\n")
        return EXIT_BAD_USAGE
    resolved = _config_for(backend, seed)
    if resolved is None:
        sys.stdout.write(f"backend must be cleargbm or lightgbm (got {backend})\n")
        return EXIT_BAD_USAGE
    backend_name, config = resolved

    dataset = hooks.dataset_loader(registry.get(dataset_name), external_dir)
    groups = dataset["groups"]
    if groups is None:
        sys.stdout.write(
            f"{dataset_name} has no group column; grouped CV needs one -- a plain "
            "k-fold of correlated rows would score memorization as skill\n"
        )
        return EXIT_BAD_USAGE

    x, y = dataset["x"], dataset["y"]
    feature_names = list(dataset["meta"]["feature_names"])
    splits = group_stratified_kfold_split(y, groups, n_folds, seed)
    trainer = BaseTabularTrainer(hooks.registry_factory())

    sys.stdout.write(
        f"{dataset_name} via {backend}: {len(y)} rows, "
        f"{len(np.unique(groups))} groups, {n_folds} folds, seed {seed}\n"
    )
    aucs: list[float] = []
    for split in splits["folds"]:
        train_idx = split["train_indices"]
        val_idx = split["val_indices"]
        with tempfile.TemporaryDirectory(prefix="cv_external_") as scratch:
            outcome = trainer.train(
                backend=backend_name,
                x_features=x[train_idx],
                y_labels=y[train_idx],
                feature_names=feature_names,
                config=config,
                output_dir=Path(scratch),
                progress=None,
                groups=groups[train_idx],
            )
            model = hooks.registry_factory().get(backend_name).load(path=outcome["model_path"])
            proba: NDArray[np.float64] = model.predict_proba(x[val_idx])
        positive: NDArray[np.float64] = proba[:, 1]
        fold_auc = compute_auc(y[val_idx], positive)
        aucs.append(fold_auc)
        held_out = len(np.unique(groups[val_idx]))
        sys.stdout.write(
            f"fold {split['fold_number']}: auc {fold_auc:.4f} "
            f"({held_out} held-out groups, {len(val_idx)} rows)\n"
        )

    mean = sum(aucs) / len(aucs)
    deviations = [(auc - mean) ** 2 for auc in aucs]
    std = math.sqrt(sum(deviations) / len(deviations))
    sys.stdout.write(f"mean auc {mean:.4f} +/- {std:.4f} over {n_folds} folds\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
