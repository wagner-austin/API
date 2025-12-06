from __future__ import annotations

from collections.abc import Callable

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from ...config.settings import Settings
from ...contracts.model import (
    BackendCapabilities,
    EvalOutcome,
    GenerateConfig,
    GenerateOutcome,
    ModelArtifact,
    ModelBackend,
    ModelTrainConfig,
    PreparedLMModel,
    ScoreConfig,
    ScoreOutcome,
    TrainOutcome,
)
from ...contracts.tokenizer import TokenizerHandle

UNAVAILABLE_CAPABILITIES: BackendCapabilities = {
    "supports_train": False,
    "supports_evaluate": False,
    "supports_score": False,
    "supports_generate": False,
    "supports_distributed": False,
    "supported_sizes": (),
}


class UnavailableBackend(ModelBackend):
    def __init__(self: UnavailableBackend, name: str) -> None:
        self._name = name

    def name(self: UnavailableBackend) -> str:
        return self._name

    def capabilities(self: UnavailableBackend) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self: UnavailableBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )

    def save(self: UnavailableBackend, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )

    def load(
        self: UnavailableBackend,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )

    def train(
        self: UnavailableBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedLMModel,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
    ) -> TrainOutcome:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )

    def evaluate(
        self: UnavailableBackend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )

    def score(
        self: UnavailableBackend,
        *,
        prepared: PreparedLMModel,
        cfg: ScoreConfig,
        settings: Settings,
    ) -> ScoreOutcome:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )

    def generate(
        self: UnavailableBackend,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        raise AppError(
            ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
            f"model backend unavailable: {self._name}",
            model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
        )
