"""Captioning services for Art-Trainer.

Provides multiple captioning backends:
- BLIP: Local model, fast and free, good for many images
- Gemini: Google's Vision API, high quality, best for small datasets
- OpenAI: GPT-4o Vision, high quality, best for small datasets

Use the registry to get the appropriate backend:

    from art_trainer.core.services.captioning import get_caption_registry, CaptionConfig

    config: CaptionConfig = {
        "backend": "blip",
        "model_name": "Salesforce/blip-image-captioning-large",
        "api_key": "",
    }
    registry = get_caption_registry()
    backend = registry.get_backend(config)
    caption = backend.caption(image_path, "sks person")
"""

from __future__ import annotations

from .backends import (
    CaptionBackend,
    CaptionBackendError,
    CaptionBackendRegistry,
    CaptionBackendType,
    CaptionConfig,
    get_caption_registry,
    reset_caption_registry,
)
from .blip_adapter import caption_image, caption_images
from .blip_model import BlipCaptioner, create_blip_caption_generator, setup_blip_caption_hook

__all__ = [
    "BlipCaptioner",
    "CaptionBackend",
    "CaptionBackendError",
    "CaptionBackendRegistry",
    "CaptionBackendType",
    "CaptionConfig",
    "caption_image",
    "caption_images",
    "create_blip_caption_generator",
    "get_caption_registry",
    "reset_caption_registry",
    "setup_blip_caption_hook",
]
