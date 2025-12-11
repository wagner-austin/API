"""Generic backend factory for creating ModelBackend implementations.

Replaces the duplicate GPT2BackendImpl and CharLSTMBackendImpl classes
with a single factory function that creates backends from function definitions.

All typing is explicit and strict - no Any, cast, or type: ignore.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypedDict

from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetBuilder
from model_trainer.core.contracts.model import (
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
from model_trainer.core.contracts.tokenizer import TokenizerHandle


class EvalResultProto(Protocol):
    """Protocol for evaluation result from backend evaluate functions."""

    loss: float
    perplexity: float


class PrepareFn(Protocol):
    """Protocol for backend prepare functions."""

    def __call__(self, tokenizer: TokenizerHandle, cfg: ModelTrainConfig) -> PreparedLMModel: ...


class SaveFn(Protocol):
    """Protocol for backend save functions."""

    def __call__(self, prepared: PreparedLMModel, out_dir: str) -> None: ...


class LoadFn(Protocol):
    """Protocol for backend load functions."""

    def __call__(self, artifact_path: str, tokenizer: TokenizerHandle) -> PreparedLMModel: ...


class TrainFn(Protocol):
    """Protocol for backend train functions."""

    def __call__(
        self,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome: ...


class EvaluateFn(Protocol):
    """Protocol for backend evaluate functions."""

    def __call__(
        self,
        *,
        run_id: str,
        cfg: ModelTrainConfig,
        settings: Settings,
        dataset_builder: DatasetBuilder,
    ) -> EvalResultProto: ...


class ScoreFn(Protocol):
    """Protocol for backend score functions."""

    def __call__(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: ScoreConfig,
        settings: Settings,
    ) -> ScoreOutcome: ...


class GenerateFn(Protocol):
    """Protocol for backend generate functions."""

    def __call__(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome: ...


class BackendFuncs(TypedDict):
    """Function references defining a backend implementation."""

    name: str
    prepare: PrepareFn
    save: SaveFn
    load: LoadFn
    train: TrainFn
    evaluate: EvaluateFn
    score: ScoreFn
    generate: GenerateFn


class _FactoryBackend:
    """ModelBackend implementation created by the factory."""

    _funcs: BackendFuncs
    _ds: DatasetBuilder
    _caps: BackendCapabilities

    def __init__(
        self,
        funcs: BackendFuncs,
        dataset_builder: DatasetBuilder,
        caps: BackendCapabilities,
    ) -> None:
        self._funcs = funcs
        self._ds = dataset_builder
        self._caps = caps

    def name(self) -> str:
        return self._funcs["name"]

    def capabilities(self) -> BackendCapabilities:
        return self._caps

    def prepare(
        self,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return self._funcs["prepare"](tokenizer, cfg)

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        self._funcs["save"](prepared, out_dir)
        return ModelArtifact(out_dir=out_dir)

    def load(
        self,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return self._funcs["load"](artifact_path, tokenizer)

    def train(
        self,
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
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome:
        return self._funcs["train"](
            prepared,
            cfg,
            settings,
            run_id=run_id,
            redis_hb=heartbeat,
            cancelled=cancelled,
            progress=progress,
            wandb_publisher=wandb_publisher,
        )

    def evaluate(
        self,
        *,
        run_id: str,
        cfg: ModelTrainConfig,
        settings: Settings,
    ) -> EvalOutcome:
        result = self._funcs["evaluate"](
            run_id=run_id,
            cfg=cfg,
            settings=settings,
            dataset_builder=self._ds,
        )
        return EvalOutcome(loss=result.loss, perplexity=result.perplexity)

    def score(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: ScoreConfig,
        settings: Settings,
    ) -> ScoreOutcome:
        return self._funcs["score"](prepared=prepared, cfg=cfg, settings=settings)

    def generate(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        return self._funcs["generate"](prepared=prepared, cfg=cfg, settings=settings)


def create_backend(
    funcs: BackendFuncs,
    dataset_builder: DatasetBuilder,
    caps: BackendCapabilities,
) -> ModelBackend:
    """Create a ModelBackend from function definitions.

    Args:
        funcs: TypedDict containing all backend function references.
        dataset_builder: DatasetBuilder for evaluation operations.
        caps: Capabilities declaration for this backend.

    Returns:
        A ModelBackend implementation that delegates to the provided functions.
    """
    return _FactoryBackend(funcs, dataset_builder, caps)


# Capabilities declarations for each backend
GPT2_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_evaluate": True,
    "supports_score": True,
    "supports_generate": True,
    "supports_distributed": False,
    "supported_sizes": ("tiny", "small", "medium", "large"),
}

CHAR_LSTM_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_evaluate": True,
    "supports_score": True,
    "supports_generate": True,
    "supports_distributed": False,
    "supported_sizes": ("small",),
}


# Pre-defined backend function bundles for GPT2 and CharLSTM
def gpt2_backend_funcs() -> BackendFuncs:
    """Get function references for GPT2 backend."""
    from model_trainer.core import _test_hooks

    from .backends.gpt2.evaluate import evaluate_gpt2
    from .backends.gpt2.generate import generate_gpt2
    from .backends.gpt2.io import save_prepared_gpt2
    from .backends.gpt2.prepare import prepare_gpt2_with_handle
    from .backends.gpt2.score import score_gpt2
    from .backends.gpt2.train import train_prepared_gpt2

    return BackendFuncs(
        name="gpt2",
        prepare=prepare_gpt2_with_handle,
        save=save_prepared_gpt2,
        load=_test_hooks.load_prepared_gpt2_from_handle,
        train=train_prepared_gpt2,
        evaluate=evaluate_gpt2,
        score=score_gpt2,
        generate=generate_gpt2,
    )


def char_lstm_backend_funcs() -> BackendFuncs:
    """Get function references for CharLSTM backend."""
    from .backends.char_lstm.evaluate import evaluate_char_lstm
    from .backends.char_lstm.generate import generate_char_lstm
    from .backends.char_lstm.io import (
        load_prepared_char_lstm_from_handle,
        save_prepared_char_lstm,
    )
    from .backends.char_lstm.prepare import prepare_char_lstm_with_handle
    from .backends.char_lstm.score import score_char_lstm
    from .backends.char_lstm.train import train_prepared_char_lstm

    return BackendFuncs(
        name="char_lstm",
        prepare=prepare_char_lstm_with_handle,
        save=save_prepared_char_lstm,
        load=load_prepared_char_lstm_from_handle,
        train=train_prepared_char_lstm,
        evaluate=evaluate_char_lstm,
        score=score_char_lstm,
        generate=generate_char_lstm,
    )


def create_gpt2_backend(dataset_builder: DatasetBuilder) -> ModelBackend:
    """Create a GPT2 ModelBackend."""
    return create_backend(gpt2_backend_funcs(), dataset_builder, GPT2_CAPABILITIES)


def create_char_lstm_backend(dataset_builder: DatasetBuilder) -> ModelBackend:
    """Create a CharLSTM ModelBackend."""
    return create_backend(char_lstm_backend_funcs(), dataset_builder, CHAR_LSTM_CAPABILITIES)
