"""capabilities: _detect_ml_capabilities and related definitions."""

from __future__ import annotations

from platform_codebase import (
    CodebaseCapability,
    LibInfo,
    ServiceInfo,
    collect_all_dependencies,
    has_dependency,
)

# -----------------------------------------------------------------------------
# Capability Detection
# -----------------------------------------------------------------------------


def _detect_ml_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect ML-related capabilities from scanned libs/services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected ML capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # XGBoost capability
    if has_dependency(deps_tuple, "xgboost"):
        capabilities.append(
            CodebaseCapability(
                name="xgboost_tabular",
                strength="strong",
                tags=(
                    "tabular",
                    "classification",
                    "regression",
                    "xgboost",
                    "ml",
                    "ai",
                    "data-science",
                ),
                description="XGBoost gradient boosting for tabular data",
            )
        )

    # LightGBM capability
    if has_dependency(deps_tuple, "lightgbm"):
        capabilities.append(
            CodebaseCapability(
                name="lightgbm_tabular",
                strength="strong",
                tags=(
                    "tabular",
                    "classification",
                    "regression",
                    "lightgbm",
                    "ml",
                    "ai",
                    "data-science",
                ),
                description="LightGBM for large-scale tabular data",
            )
        )

    # PyTorch / Deep Learning capability
    if has_dependency(deps_tuple, "torch"):
        capabilities.append(
            CodebaseCapability(
                name="pytorch_deep_learning",
                strength="strong",
                tags=("deep-learning", "neural-network", "pytorch", "ml", "ai"),
                description="PyTorch for deep learning models",
            )
        )

    # Optuna hyperparameter optimization
    if has_dependency(deps_tuple, "optuna"):
        capabilities.append(
            CodebaseCapability(
                name="hyperparameter_optimization",
                strength="strong",
                tags=("optimization", "hyperparameter-tuning", "optuna", "ml"),
                description="Optuna for hyperparameter optimization",
            )
        )

    # scikit-learn
    if has_dependency(deps_tuple, "scikit-learn"):
        capabilities.append(
            CodebaseCapability(
                name="sklearn_ml",
                strength="moderate",
                tags=(
                    "tabular",
                    "classification",
                    "regression",
                    "sklearn",
                    "ml",
                    "ai",
                    "data-science",
                ),
                description="scikit-learn for machine learning",
            )
        )

    return tuple(capabilities)


def _detect_cv_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect computer vision capabilities from scanned libs/services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected CV capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # TorchVision - computer vision with PyTorch
    if has_dependency(deps_tuple, "torchvision"):
        capabilities.append(
            CodebaseCapability(
                name="torchvision_cv",
                strength="strong",
                tags=("computer-vision", "image", "pytorch", "image-classification"),
                description="TorchVision for computer vision tasks",
            )
        )

    # Pillow - image processing
    if has_dependency(deps_tuple, "pillow"):
        capabilities.append(
            CodebaseCapability(
                name="image_processing",
                strength="moderate",
                tags=("image", "image-processing", "pillow"),
                description="Pillow for image manipulation and processing",
            )
        )

    # OpenCV
    if has_dependency(deps_tuple, "opencv-python"):
        capabilities.append(
            CodebaseCapability(
                name="opencv_cv",
                strength="strong",
                tags=("computer-vision", "image", "opencv", "video"),
                description="OpenCV for computer vision and video processing",
            )
        )

    return tuple(capabilities)


def _detect_transformers_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect Hugging Face / transformers capabilities.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected transformers capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # Hugging Face Transformers
    if has_dependency(deps_tuple, "transformers"):
        capabilities.append(
            CodebaseCapability(
                name="huggingface_transformers",
                strength="strong",
                tags=(
                    "nlp",
                    "transformers",
                    "huggingface",
                    "text-classification",
                    "text-generation",
                    "llm",
                ),
                description="Hugging Face Transformers for NLP and LLMs",
            )
        )

    # Hugging Face Datasets
    if has_dependency(deps_tuple, "datasets"):
        capabilities.append(
            CodebaseCapability(
                name="huggingface_datasets",
                strength="moderate",
                tags=("data", "huggingface", "datasets"),
                description="Hugging Face Datasets for ML data loading",
            )
        )

    # Tokenizers
    if has_dependency(deps_tuple, "tokenizers"):
        capabilities.append(
            CodebaseCapability(
                name="tokenization",
                strength="moderate",
                tags=("nlp", "tokenization", "huggingface"),
                description="Fast tokenization for NLP models",
            )
        )

    # SentencePiece
    if has_dependency(deps_tuple, "sentencepiece"):
        capabilities.append(
            CodebaseCapability(
                name="sentencepiece_tokenization",
                strength="moderate",
                tags=("nlp", "tokenization", "sentencepiece"),
                description="SentencePiece for subword tokenization",
            )
        )

    return tuple(capabilities)
