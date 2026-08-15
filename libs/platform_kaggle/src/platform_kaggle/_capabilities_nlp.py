"""capabilities: _detect_nlp_capabilities and related definitions."""

from __future__ import annotations

from platform_codebase import (
    CodebaseCapability,
    LibInfo,
    ServiceInfo,
    collect_all_dependencies,
    has_dependency,
)


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
