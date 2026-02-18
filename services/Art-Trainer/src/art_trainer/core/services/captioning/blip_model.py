"""BLIP model loader for production captioning.

This module provides the production BLIP model implementation for
image captioning. It uses lazy initialization to load the model
only when first used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TorchDeviceProto(Protocol):
    """Protocol for torch device type."""

    pass


class TensorProto(Protocol):
    """Protocol for tensor."""

    def __getitem__(self, idx: int) -> TensorProto:
        """Get tensor element.

        Args:
            idx: Index.

        Returns:
            Tensor element.
        """
        ...

    def to(self, device: TorchDeviceProto) -> TensorProto:
        """Move tensor to device.

        Args:
            device: Target device.

        Returns:
            Tensor on device.
        """
        ...


class ProcessorOutputProto(Protocol):
    """Protocol for BLIP processor output (BatchEncoding-like)."""

    @property
    def pixel_values(self) -> TensorProto:
        """Get pixel values tensor."""
        ...

    def to(self, device: TorchDeviceProto) -> ProcessorOutputProto:
        """Move tensors to device.

        Args:
            device: Target device.

        Returns:
            Self with tensors on device.
        """
        ...


class BlipProcessorProto(Protocol):
    """Protocol for BLIP processor."""

    def __call__(
        self,
        images: PILImageProto,
        return_tensors: str,
    ) -> ProcessorOutputProto:
        """Process image for model input.

        Args:
            images: PIL Image to process.
            return_tensors: Return tensor format ("pt" for PyTorch).

        Returns:
            Processor output tensors.
        """
        ...

    def decode(self, token_ids: TensorProto, skip_special_tokens: bool) -> str:
        """Decode token IDs to string.

        Args:
            token_ids: Token IDs tensor.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded string.
        """
        ...


class BlipModelProto(Protocol):
    """Protocol for BLIP model."""

    def to(self, device: TorchDeviceProto) -> BlipModelProto:
        """Move model to device.

        Args:
            device: Target device.

        Returns:
            Self on device.
        """
        ...

    def generate(
        self,
        pixel_values: TensorProto,
        max_new_tokens: int,
    ) -> TensorProto:
        """Generate caption tokens.

        Args:
            pixel_values: Image pixel values tensor.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Generated token tensor.
        """
        ...


class PILImageProto(Protocol):
    """Protocol for PIL Image."""

    def convert(self, mode: str) -> PILImageProto:
        """Convert image to mode.

        Args:
            mode: Target mode (e.g., "RGB").

        Returns:
            Converted image.
        """
        ...


class _TorchCudaProto(Protocol):
    """Protocol for torch.cuda module."""

    def is_available(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA is available.
        """
        ...


class _TorchModuleProto(Protocol):
    """Protocol for torch module."""

    cuda: _TorchCudaProto

    def device(self, device_name: str) -> TorchDeviceProto:
        """Create a device object.

        Args:
            device_name: Device name ("cuda" or "cpu").

        Returns:
            Device object.
        """
        ...


class _ProcessorFactoryProto(Protocol):
    """Protocol for BLIP processor factory."""

    def from_pretrained(self, model_name: str) -> BlipProcessorProto:
        """Load pretrained processor.

        Args:
            model_name: Model name.

        Returns:
            Loaded processor.
        """
        ...


class _ModelFactoryProto(Protocol):
    """Protocol for BLIP model factory."""

    def from_pretrained(self, model_name: str) -> BlipModelProto:
        """Load pretrained model.

        Args:
            model_name: Model name.

        Returns:
            Loaded model.
        """
        ...


class _TransformersModuleProto(Protocol):
    """Protocol for transformers module."""

    BlipProcessor: _ProcessorFactoryProto
    BlipForConditionalGeneration: _ModelFactoryProto


class _PILImageOpenProto(Protocol):
    """Protocol for PIL Image.open function."""

    def __call__(self, path: Path) -> PILImageProto:
        """Open an image file.

        Args:
            path: Path to image file.

        Returns:
            Opened image.
        """
        ...


class _PILImageModuleProto(Protocol):
    """Protocol for PIL.Image module."""

    open: _PILImageOpenProto


class BlipCaptioner:
    """Production BLIP captioning implementation.

    Uses lazy initialization to load the model only when first used.
    The model is cached as a singleton for efficiency.
    """

    _instance: BlipCaptioner | None = None
    _model: BlipModelProto | None
    _processor: BlipProcessorProto | None
    _device: TorchDeviceProto | None

    def __init__(self, model_name: str) -> None:
        """Initialize captioner with model name.

        Args:
            model_name: HuggingFace model name (e.g., "Salesforce/blip-image-captioning-large").
        """
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

    @classmethod
    def get_instance(cls, model_name: str) -> BlipCaptioner:
        """Get or create singleton instance.

        Args:
            model_name: HuggingFace model name.

        Returns:
            BlipCaptioner instance.
        """
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for testing."""
        cls._instance = None

    def _ensure_loaded(
        self,
    ) -> tuple[BlipProcessorProto, BlipModelProto, TorchDeviceProto]:
        """Ensure model and processor are loaded.

        Returns:
            Tuple of (processor, model, device).
        """
        # Return cached values if already loaded
        if self._processor is not None and self._model is not None and self._device is not None:
            return self._processor, self._model, self._device

        # Dynamic imports with Protocol type annotations
        torch_raw = __import__("torch")
        torch_mod: _TorchModuleProto = torch_raw
        transformers_raw = __import__("transformers")
        transformers_mod: _TransformersModuleProto = transformers_raw

        # Get device
        cuda_available: bool = torch_mod.cuda.is_available()
        device_name = "cuda" if cuda_available else "cpu"
        device: TorchDeviceProto = torch_mod.device(device_name)
        self._device = device

        # Load processor and model
        processor_cls: _ProcessorFactoryProto = transformers_mod.BlipProcessor
        model_cls: _ModelFactoryProto = transformers_mod.BlipForConditionalGeneration

        processor: BlipProcessorProto = processor_cls.from_pretrained(self._model_name)
        model: BlipModelProto = model_cls.from_pretrained(self._model_name)
        model = model.to(device)

        self._processor = processor
        self._model = model

        return processor, model, device

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image.

        Args:
            image_path: Path to the image file.
            trigger_word: Trigger word to prepend to caption.

        Returns:
            Generated caption with trigger word prefix.

        Raises:
            FileNotFoundError: If image_path does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        processor, model, device = self._ensure_loaded()

        # Load and process image with Protocol type annotation
        pil_raw = __import__("PIL.Image", fromlist=["Image"])
        pil_mod: _PILImageModuleProto = pil_raw
        image_open: _PILImageOpenProto = pil_mod.open
        image: PILImageProto = image_open(image_path).convert("RGB")

        inputs: ProcessorOutputProto = processor(image, return_tensors="pt")
        inputs = inputs.to(device)

        # Generate caption using explicit pixel_values
        pixel_values = inputs.pixel_values
        output: TensorProto = model.generate(pixel_values=pixel_values, max_new_tokens=50)
        caption_tokens = output[0]

        # Decode caption using processor
        caption: str = processor.decode(caption_tokens, skip_special_tokens=True)

        return f"{trigger_word}, {caption}"


def create_blip_caption_generator(model_name: str) -> _CaptionGeneratorFn:
    """Create a caption generator function for use as a hook.

    Args:
        model_name: HuggingFace model name.

    Returns:
        Caption generator function compatible with the captioning hook.
    """
    captioner = BlipCaptioner.get_instance(model_name)

    def caption_generator(image_path: Path, trigger_word: str) -> str:
        return captioner.caption(image_path, trigger_word)

    return caption_generator


class _CaptionGeneratorFn(Protocol):
    """Protocol for caption generator function."""

    def __call__(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image.

        Args:
            image_path: Path to image.
            trigger_word: Trigger word to prepend.

        Returns:
            Generated caption.
        """
        ...


def setup_blip_caption_hook(model_name: str) -> None:
    """Set up the BLIP caption hook for production use.

    Call this function at API/worker startup to enable BLIP captioning.

    Args:
        model_name: HuggingFace model name (e.g., "Salesforce/blip-image-captioning-large").
    """
    from art_trainer.core.services.captioning import _test_hooks

    _test_hooks.Hooks.caption_generator = create_blip_caption_generator(model_name)


__all__ = [
    "BlipCaptioner",
    "create_blip_caption_generator",
    "setup_blip_caption_hook",
]
