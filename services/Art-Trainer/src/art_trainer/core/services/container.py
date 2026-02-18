"""Service container for Art-Trainer.

This module provides the ServiceContainer that manages all service dependencies.
"""

from __future__ import annotations

from platform_workers.redis import RedisStrProto

from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings
from art_trainer.orchestrators.lora_orchestrator import LoraOrchestrator

from .queue.rq_adapter import RQEnqueuer, RQSettings
from .registries import BackendFactory, BackendRegistry
from .training.backend_factory import create_kohya_backend


class ServiceContainer:
    """Container for all service dependencies.

    Manages creation and lifecycle of service components.
    """

    settings: Settings
    redis: RedisStrProto
    rq_enqueuer: RQEnqueuer
    backend_registry: BackendRegistry
    lora_orchestrator: LoraOrchestrator

    def __init__(
        self: ServiceContainer,
        settings: Settings,
        redis: RedisStrProto,
        rq_enqueuer: RQEnqueuer,
        backend_registry: BackendRegistry,
        lora_orchestrator: LoraOrchestrator,
    ) -> None:
        """Initialize service container.

        Args:
            settings: Application settings.
            redis: Redis client.
            rq_enqueuer: RQ job enqueuer.
            backend_registry: Backend registry.
            lora_orchestrator: LoRA orchestrator.
        """
        self.settings = settings
        self.redis = redis
        self.rq_enqueuer = rq_enqueuer
        self.backend_registry = backend_registry
        self.lora_orchestrator = lora_orchestrator

    @classmethod
    def from_settings(cls: type[ServiceContainer], settings: Settings) -> ServiceContainer:
        """Create ServiceContainer from settings.

        Args:
            settings: Application settings.

        Returns:
            Fully initialized ServiceContainer.
        """
        redis_url = settings["redis"]["url"]
        r: RedisStrProto = _test_hooks.kv_store_factory(redis_url)
        enq = _create_enqueuer(settings)

        # Create backend registry
        backends: dict[str, BackendFactory] = {
            "kohya": create_kohya_backend,
        }
        backend_registry = BackendRegistry(backends, settings)

        # Create orchestrator
        lora_orch = LoraOrchestrator(
            settings=settings,
            redis_client=r,
            enqueuer=enq,
            backend_registry=backend_registry,
        )

        return cls(
            settings=settings,
            redis=r,
            rq_enqueuer=enq,
            backend_registry=backend_registry,
            lora_orchestrator=lora_orch,
        )


def _create_enqueuer(settings: Settings) -> RQEnqueuer:
    """Create RQ enqueuer from settings.

    Args:
        settings: Application settings.

    Returns:
        Configured RQEnqueuer.
    """
    rq_cfg = RQSettings(
        job_timeout_sec=settings["rq"]["job_timeout_sec"],
        result_ttl_sec=settings["rq"]["result_ttl_sec"],
        failure_ttl_sec=settings["rq"]["failure_ttl_sec"],
        retry_max=settings["rq"]["retry_max"],
        retry_intervals=[int(x) for x in settings["rq"]["retry_intervals_sec"].split(",") if x],
    )
    return RQEnqueuer(redis_url=settings["redis"]["url"], settings=rq_cfg)


__all__ = [
    "ServiceContainer",
]
