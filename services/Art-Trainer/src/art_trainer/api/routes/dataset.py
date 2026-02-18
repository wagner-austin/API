"""Dataset upload routes for Art-Trainer API.

This module provides endpoints for uploading and captioning training datasets.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.params import Depends as DependsParamType
from fastapi.responses import JSONResponse
from platform_core.json_utils import JSONTypeError, load_json_str

from art_trainer.api.middleware import api_key_dependency
from art_trainer.api.schemas.dataset import (
    DatasetCaptionRequest,
    DatasetCaptionResponse,
    DatasetUploadResponse,
    encode_dataset_caption_response,
    encode_dataset_upload_response,
)
from art_trainer.api.validators.dataset import decode_dataset_caption_request
from art_trainer.core.infra.paths import dataset_dir
from art_trainer.core.services.captioning import caption_images
from art_trainer.core.services.captioning.backends import (
    CaptionConfig,
    get_caption_registry,
)
from art_trainer.core.services.captioning.blip_adapter import (
    IMAGE_EXTENSIONS,
    find_images,
)
from art_trainer.core.services.container import ServiceContainer


class _DatasetRoutes:
    """Route handlers for dataset endpoints."""

    _container: ServiceContainer

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize route handlers.

        Args:
            container: Service container.
        """
        self._container = container

    async def upload_dataset(
        self,
        files: Annotated[list[UploadFile], File(description="Image files to upload")],
        trigger_word: Annotated[str, Form(description="Trigger word for captions")],
        training_type: Annotated[
            str, Form(description="Training type: style, character, or concept")
        ],
        auto_caption: Annotated[bool, Form(description="Auto-generate captions with BLIP")],
    ) -> JSONResponse:
        """Upload images for a training dataset.

        Creates a new dataset directory with the uploaded images.
        Optionally generates captions using BLIP.

        Args:
            files: List of image files to upload.
            trigger_word: Trigger word to prepend to captions.
            training_type: Type of training (style, character, concept).
            auto_caption: Whether to auto-generate captions.

        Returns:
            JSON response with dataset info.
        """
        del training_type  # Unused for now, will be used for config selection

        # Generate dataset ID
        dataset_id = str(uuid.uuid4())

        # Create dataset directory
        settings = self._container.settings
        ds_dir = dataset_dir(settings, dataset_id)
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded images
        image_count = 0
        for upload_file in files:
            # Use empty string if filename is None (handles edge case)
            filename = upload_file.filename or ""

            # Check if it's an image file
            ext = Path(filename).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            # Save the file
            file_path = ds_dir / filename
            content = await upload_file.read()
            file_path.write_bytes(content)
            image_count += 1

        # Generate captions if requested
        caption_count = 0
        if auto_caption and image_count > 0:
            image_paths = find_images(ds_dir)
            results = caption_images(image_paths, trigger_word, ds_dir)
            caption_count = len(results)

        # Build response
        response: DatasetUploadResponse = {
            "dataset_id": dataset_id,
            "image_count": image_count,
            "caption_count": caption_count,
            "dataset_path": str(ds_dir),
        }

        return JSONResponse(
            status_code=200,
            content=encode_dataset_upload_response(response),
        )

    def get_dataset(self, dataset_id: str) -> JSONResponse:
        """Get information about a dataset.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            JSON response with dataset info.
        """
        settings = self._container.settings
        ds_dir = dataset_dir(settings, dataset_id)

        if not ds_dir.exists():
            error_detail: dict[str, str] = {"detail": f"Dataset {dataset_id} not found"}
            return JSONResponse(
                status_code=404,
                content=error_detail,
            )

        # Count images
        image_paths = find_images(ds_dir)
        image_count = len(image_paths)

        # Count captions
        caption_count = len(list(ds_dir.glob("*.txt")))

        response: DatasetUploadResponse = {
            "dataset_id": dataset_id,
            "image_count": image_count,
            "caption_count": caption_count,
            "dataset_path": str(ds_dir),
        }

        return JSONResponse(
            status_code=200,
            content=encode_dataset_upload_response(response),
        )

    async def caption_dataset(self, dataset_id: str, request: Request) -> JSONResponse:
        """Caption images in a dataset using specified backend.

        Args:
            dataset_id: Dataset identifier.
            request: FastAPI request containing caption configuration.

        Returns:
            JSON response with captioning results.

        Raises:
            JSONTypeError: If request body is invalid.
        """
        settings = self._container.settings
        ds_dir = dataset_dir(settings, dataset_id)

        if not ds_dir.exists():
            error_detail: dict[str, str] = {"detail": f"Dataset {dataset_id} not found"}
            return JSONResponse(
                status_code=404,
                content=error_detail,
            )

        # Parse and validate request body
        raw_body = await request.body()
        body = load_json_str(raw_body.decode("utf-8"))
        if not isinstance(body, dict):
            raise JSONTypeError("Request body must be a JSON object")
        req: DatasetCaptionRequest = decode_dataset_caption_request(body)

        # Get API key for the backend
        api_key = ""
        if req["backend"] == "gemini":
            api_key = settings["app"]["gemini_api_key"]
        elif req["backend"] == "openai":
            api_key = settings["app"]["openai_api_key"]

        # Build caption config
        caption_config: CaptionConfig = {
            "backend": req["backend"],
            "model_name": req["model_name"],
            "api_key": api_key,
        }

        # Get backend from registry
        registry = get_caption_registry()
        backend = registry.get_backend(caption_config)

        # Find images and caption uncaptioned ones
        image_paths = find_images(ds_dir)
        captioned_count = 0
        skipped_count = 0

        for image_path in image_paths:
            caption_path = image_path.with_suffix(".txt")

            # Skip if caption already exists
            if caption_path.exists():
                skipped_count += 1
                continue

            # Generate and save caption
            caption = backend.caption(image_path, req["trigger_word"])
            caption_path.write_text(caption, encoding="utf-8")
            captioned_count += 1

        # Build response
        response: DatasetCaptionResponse = {
            "dataset_id": dataset_id,
            "captioned_count": captioned_count,
            "skipped_count": skipped_count,
            "backend": req["backend"],
        }

        return JSONResponse(
            status_code=200,
            content=encode_dataset_caption_response(response),
        )


def build_router(container: ServiceContainer) -> APIRouter:
    """Build the dataset router.

    Args:
        container: Service container with dependencies.

    Returns:
        Configured APIRouter for dataset endpoints.
    """
    api_dep: DependsParamType = Depends(api_key_dependency(container.settings))
    router = APIRouter(dependencies=[api_dep])
    handlers = _DatasetRoutes(container)

    router.add_api_route("/upload", handlers.upload_dataset, methods=["POST"])
    router.add_api_route("/{dataset_id}", handlers.get_dataset, methods=["GET"])
    router.add_api_route("/{dataset_id}/caption", handlers.caption_dataset, methods=["POST"])

    return router


__all__ = [
    "build_router",
]
