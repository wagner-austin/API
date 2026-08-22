"""Shared shapes for the ML routes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypedDict

from covenant_ml.types import PredictorProtocol
from covenant_persistence import (
    CovenantResultRepository,
    DealRepository,
    MeasurementRepository,
)
from platform_core.json_utils import JSONValue
from platform_workers.rq_harness import RQClientQueue


class ModelInfo(TypedDict, total=True):
    """Information about the active ML model."""

    model_id: str
    model_path: str
    is_loaded: bool


class JobStatus(TypedDict, total=True):
    """Status of a background job."""

    job_id: str
    status: Literal["queued", "started", "finished", "failed", "not_found"]
    result: JSONValue | None


class ContainerProtocol(Protocol):
    """Protocol for service container with ML dependencies."""

    def deal_repo(self) -> DealRepository: ...

    def measurement_repo(self) -> MeasurementRepository: ...

    def covenant_result_repo(self) -> CovenantResultRepository: ...

    def rq_queue(self) -> RQClientQueue: ...

    def get_model(self) -> PredictorProtocol: ...

    def get_model_info(self) -> ModelInfo: ...

    def get_sector_encoder(self) -> dict[str, int]: ...

    def get_region_encoder(self) -> dict[str, int]: ...

    def get_job_status(self, job_id: str) -> JobStatus: ...

    def models_root(self) -> Path: ...
