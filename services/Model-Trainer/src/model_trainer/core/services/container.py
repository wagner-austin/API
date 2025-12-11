from __future__ import annotations

from platform_core.queues import TRAINER_QUEUE
from platform_workers.redis import RedisStrProto

from ...orchestrators.conversation_orchestrator import ConversationOrchestrator
from ...orchestrators.inference_orchestrator import InferenceOrchestrator
from ...orchestrators.tokenizer_orchestrator import TokenizerOrchestrator
from ...orchestrators.training_orchestrator import TrainingOrchestrator
from .. import _test_hooks
from ..config.settings import Settings
from ..contracts.dataset import DatasetBuilder
from ..contracts.tokenizer import TokenizerBackend
from ..services.dataset.local_text_builder import LocalTextDatasetBuilder
from ..services.registries import BackendRegistration, ModelRegistry, TokenizerRegistry
from .model.backend_factory import (
    CHAR_LSTM_CAPABILITIES,
    GPT2_CAPABILITIES,
    create_char_lstm_backend,
    create_gpt2_backend,
)
from .model.unavailable_backend import UNAVAILABLE_CAPABILITIES, UnavailableBackend
from .queue.rq_adapter import RQEnqueuer, RQSettings
from .tokenizer.bpe_backend import BPEBackend
from .tokenizer.char_backend import CharBackend


class ServiceContainer:
    settings: Settings
    redis: RedisStrProto
    rq_enqueuer: RQEnqueuer
    training_orchestrator: TrainingOrchestrator
    inference_orchestrator: InferenceOrchestrator
    conversation_orchestrator: ConversationOrchestrator
    tokenizer_orchestrator: TokenizerOrchestrator
    model_registry: ModelRegistry
    tokenizer_registry: TokenizerRegistry
    dataset_builder: DatasetBuilder

    def __init__(
        self: ServiceContainer,
        settings: Settings,
        redis: RedisStrProto,
        rq_enqueuer: RQEnqueuer,
        training_orchestrator: TrainingOrchestrator,
        inference_orchestrator: InferenceOrchestrator,
        conversation_orchestrator: ConversationOrchestrator,
        tokenizer_orchestrator: TokenizerOrchestrator,
        model_registry: ModelRegistry,
        tokenizer_registry: TokenizerRegistry,
        dataset_builder: DatasetBuilder,
    ) -> None:
        self.settings = settings
        self.redis = redis
        self.rq_enqueuer = rq_enqueuer
        self.training_orchestrator = training_orchestrator
        self.inference_orchestrator = inference_orchestrator
        self.conversation_orchestrator = conversation_orchestrator
        self.tokenizer_orchestrator = tokenizer_orchestrator
        self.model_registry = model_registry
        self.tokenizer_registry = tokenizer_registry
        self.dataset_builder = dataset_builder

    @classmethod
    def from_settings(cls: type[ServiceContainer], settings: Settings) -> ServiceContainer:
        redis_url = settings["redis"]["url"]
        r: RedisStrProto = _test_hooks.kv_store_factory(redis_url)
        enq = _create_enqueuer(settings)
        dataset_builder = LocalTextDatasetBuilder()

        # Registries (minimal initial backends)
        model_registry = _create_model_registry(dataset_builder)
        training = TrainingOrchestrator(
            settings=settings, redis_client=r, enqueuer=enq, model_registry=model_registry
        )
        inference = InferenceOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
        conversation = ConversationOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
        tokenizer = TokenizerOrchestrator(settings=settings, redis_client=r, enqueuer=enq)
        tokenizer_registry = _create_tokenizer_registry()
        return cls(
            settings=settings,
            redis=r,
            rq_enqueuer=enq,
            training_orchestrator=training,
            inference_orchestrator=inference,
            conversation_orchestrator=conversation,
            tokenizer_orchestrator=tokenizer,
            model_registry=model_registry,
            tokenizer_registry=tokenizer_registry,
            dataset_builder=dataset_builder,
        )


def _create_model_registry(dataset_builder: DatasetBuilder) -> ModelRegistry:
    registrations: dict[str, BackendRegistration] = {
        "gpt2": BackendRegistration(
            factory=create_gpt2_backend,
            capabilities=GPT2_CAPABILITIES,
        ),
        "char_lstm": BackendRegistration(
            factory=create_char_lstm_backend,
            capabilities=CHAR_LSTM_CAPABILITIES,
        ),
        "llama": BackendRegistration(
            factory=lambda _: UnavailableBackend("llama"),
            capabilities=UNAVAILABLE_CAPABILITIES,
        ),
        "qwen": BackendRegistration(
            factory=lambda _: UnavailableBackend("qwen"),
            capabilities=UNAVAILABLE_CAPABILITIES,
        ),
    }
    return ModelRegistry(registrations=registrations, dataset_builder=dataset_builder)


def _create_enqueuer(settings: Settings) -> RQEnqueuer:
    if settings["rq"]["queue_name"] != TRAINER_QUEUE:
        raise ValueError("RQ queue must be trainer per platform alignment")
    rq_cfg = RQSettings(
        job_timeout_sec=settings["rq"]["job_timeout_sec"],
        result_ttl_sec=settings["rq"]["result_ttl_sec"],
        failure_ttl_sec=settings["rq"]["failure_ttl_sec"],
        retry_max=settings["rq"]["retry_max"],
        retry_intervals=[int(x) for x in settings["rq"]["retry_intervals_sec"].split(",") if x],
    )
    return RQEnqueuer(redis_url=settings["redis"]["url"], settings=rq_cfg)


def _create_tokenizer_registry() -> TokenizerRegistry:
    tok_backends: dict[str, TokenizerBackend] = {"bpe": BPEBackend(), "char": CharBackend()}
    spm_cmds = ("spm_train", "spm_encode", "spm_decode")
    if all(_test_hooks.shutil_which(x) is not None for x in spm_cmds):
        from .tokenizer.spm_backend import SentencePieceBackend

        tok_backends["sentencepiece"] = SentencePieceBackend()
    return TokenizerRegistry(backends=tok_backends)
