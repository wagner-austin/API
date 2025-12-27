"""Dynamic codebase capability detection.

Scans libs/ and services/ directories to detect ML/NLP capabilities
based on installed dependencies and file patterns.
"""

from __future__ import annotations

from pathlib import Path

from platform_codebase import (
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    ServiceInfo,
    collect_all_dependencies,
    has_dependency,
    scan_libs,
    scan_services,
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


def _detect_domain_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect domain-specific capabilities from service names.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected domain capabilities.
    """
    capabilities: list[CodebaseCapability] = []

    # Check for fintech/loan-related services
    for service in services:
        name_lower = service.name.lower()
        if "covenant" in name_lower or "loan" in name_lower:
            capabilities.append(
                CodebaseCapability(
                    name="loan_covenant_monitoring",
                    strength="strong",
                    tags=(
                        "fintech",
                        "finance",
                        "banking",
                        "loans",
                        "risk",
                        "compliance",
                    ),
                    description="Loan covenant monitoring and breach prediction",
                )
            )
            break  # Only add once

    return tuple(capabilities)


def _detect_nlp_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect NLP-related capabilities from scanned libs/services.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected NLP capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # FastText (language identification) - check both package names
    has_fasttext = has_dependency(deps_tuple, "fasttext") or has_dependency(
        deps_tuple, "fasttext-wheel"
    )
    if has_fasttext:
        capabilities.append(
            CodebaseCapability(
                name="language_identification",
                strength="moderate",
                tags=("nlp", "language-detection", "multilingual"),
                description="FastText for language identification",
            )
        )

    # OpenAI (speech-to-text, translation)
    if has_dependency(deps_tuple, "openai"):
        capabilities.append(
            CodebaseCapability(
                name="speech_to_text",
                strength="moderate",
                tags=("nlp", "speech", "transcription", "whisper"),
                description="OpenAI Whisper for speech-to-text",
            )
        )

    # Check for transliteration (services with .rules files)
    for service in services:
        if service.has_rules_files:
            capabilities.append(
                CodebaseCapability(
                    name="transliteration",
                    strength="moderate",
                    tags=("nlp", "transliteration", "script-conversion"),
                    description="Rule-based transliteration between scripts",
                )
            )
            break  # Only add once

    return tuple(capabilities)


def _detect_ml_backends(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect ML backend libraries in use.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of ML backend names.
    """
    backends: set[str] = set()
    deps_tuple = collect_all_dependencies(libs, services)

    if has_dependency(deps_tuple, "xgboost"):
        backends.add("xgboost")
    if has_dependency(deps_tuple, "lightgbm"):
        backends.add("lightgbm")
    if has_dependency(deps_tuple, "torch"):
        backends.add("pytorch")
    if has_dependency(deps_tuple, "scikit-learn"):
        backends.add("sklearn")
    if has_dependency(deps_tuple, "tensorflow"):
        backends.add("tensorflow")
    if has_dependency(deps_tuple, "transformers"):
        backends.add("transformers")
    if has_dependency(deps_tuple, "torchvision"):
        backends.add("torchvision")
    if has_dependency(deps_tuple, "catboost"):
        backends.add("catboost")

    return tuple(sorted(backends))


def _detect_data_formats(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect supported data formats.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of data format names.
    """
    formats: set[str] = set()
    deps_tuple = collect_all_dependencies(libs, services)

    # pandas implies CSV, Excel support
    if has_dependency(deps_tuple, "pandas"):
        formats.add("csv")
        formats.add("excel")

    # pyarrow/parquet support
    if has_dependency(deps_tuple, "pyarrow") or has_dependency(deps_tuple, "polars"):
        formats.add("parquet")

    # polars implies CSV, parquet
    if has_dependency(deps_tuple, "polars"):
        formats.add("csv")

    return tuple(sorted(formats))


def _detect_task_types(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect supported ML task types.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of task type names.
    """
    tasks: set[str] = set()
    deps_tuple = collect_all_dependencies(libs, services)

    # Gradient boosting libs imply classification/regression
    if has_dependency(deps_tuple, "xgboost") or has_dependency(deps_tuple, "lightgbm"):
        tasks.add("binary_classification")
        tasks.add("multiclass_classification")
        tasks.add("regression")

    # PyTorch implies more advanced tasks
    if has_dependency(deps_tuple, "torch"):
        tasks.add("time_series")
        tasks.add("sequence_modeling")

    # scikit-learn
    if has_dependency(deps_tuple, "scikit-learn"):
        tasks.add("binary_classification")
        tasks.add("multiclass_classification")
        tasks.add("regression")
        tasks.add("clustering")

    # TorchVision - computer vision tasks
    if has_dependency(deps_tuple, "torchvision"):
        tasks.add("image_classification")
        tasks.add("object_detection")

    # Transformers - NLP tasks
    if has_dependency(deps_tuple, "transformers"):
        tasks.add("text_classification")
        tasks.add("text_generation")
        tasks.add("token_classification")
        tasks.add("question_answering")
        tasks.add("summarization")
        tasks.add("translation")

    # OpenAI - speech tasks
    if has_dependency(deps_tuple, "openai"):
        tasks.add("speech_recognition")
        tasks.add("translation")

    return tuple(sorted(tasks))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_profile(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> CodebaseProfile:
    """Build capability profile from pre-scanned libs and services.

    This function accepts already-scanned data, enabling use with data
    from GitHub API or other sources beyond local filesystem.

    Args:
        libs: Tuple of LibInfo from libs directory.
        services: Tuple of ServiceInfo from services directory.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    ml_caps = _detect_ml_capabilities(libs, services)
    cv_caps = _detect_cv_capabilities(libs, services)
    transformers_caps = _detect_transformers_capabilities(libs, services)
    nlp_caps = _detect_nlp_capabilities(libs, services)
    domain_caps = _detect_domain_capabilities(libs, services)

    all_caps = ml_caps + cv_caps + transformers_caps + nlp_caps + domain_caps

    return CodebaseProfile(
        capabilities=all_caps,
        ml_backends=_detect_ml_backends(libs, services),
        data_formats=_detect_data_formats(libs, services),
        task_types=_detect_task_types(libs, services),
    )


def scan_codebase(root: Path) -> CodebaseProfile:
    """Scan codebase and return capability profile.

    Scans the libs/ and services/ directories to detect installed
    dependencies and infer ML/NLP capabilities.

    Args:
        root: Path to monorepo root directory.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    libs = scan_libs(root)
    services = scan_services(root)
    return build_profile(libs, services)


__all__ = [
    "build_profile",
    "scan_codebase",
]
