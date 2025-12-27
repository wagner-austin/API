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


def _detect_observability_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect observability and monitoring capabilities.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected observability capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # Datadog APM tracing
    if has_dependency(deps_tuple, "ddtrace"):
        capabilities.append(
            CodebaseCapability(
                name="datadog_apm",
                strength="strong",
                tags=(
                    "observability",
                    "monitoring",
                    "apm",
                    "tracing",
                    "datadog",
                ),
                description="Datadog APM for distributed tracing and monitoring",
            )
        )

    # Prometheus metrics
    if has_dependency(deps_tuple, "prometheus-client"):
        capabilities.append(
            CodebaseCapability(
                name="prometheus_metrics",
                strength="moderate",
                tags=("observability", "monitoring", "metrics", "prometheus"),
                description="Prometheus client for metrics collection",
            )
        )

    # OpenTelemetry
    if has_dependency(deps_tuple, "opentelemetry-api"):
        capabilities.append(
            CodebaseCapability(
                name="opentelemetry",
                strength="moderate",
                tags=("observability", "tracing", "opentelemetry", "otel"),
                description="OpenTelemetry for observability instrumentation",
            )
        )

    return tuple(capabilities)


def _detect_streaming_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect streaming and message queue capabilities.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected streaming capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # Confluent Kafka
    if has_dependency(deps_tuple, "confluent-kafka"):
        capabilities.append(
            CodebaseCapability(
                name="confluent_kafka",
                strength="strong",
                tags=(
                    "streaming",
                    "kafka",
                    "confluent",
                    "real-time",
                    "event-driven",
                    "message-queue",
                ),
                description="Confluent Kafka for real-time data streaming",
            )
        )

    # kafka-python
    if has_dependency(deps_tuple, "kafka-python"):
        capabilities.append(
            CodebaseCapability(
                name="kafka_python",
                strength="moderate",
                tags=("streaming", "kafka", "real-time", "message-queue"),
                description="Kafka Python client for message streaming",
            )
        )

    # Redis (pub/sub, caching)
    if has_dependency(deps_tuple, "redis"):
        capabilities.append(
            CodebaseCapability(
                name="redis",
                strength="moderate",
                tags=("caching", "redis", "pub-sub", "message-queue"),
                description="Redis for caching and pub/sub messaging",
            )
        )

    # RabbitMQ
    if has_dependency(deps_tuple, "pika"):
        capabilities.append(
            CodebaseCapability(
                name="rabbitmq",
                strength="moderate",
                tags=("messaging", "rabbitmq", "amqp", "message-queue"),
                description="RabbitMQ for message queuing",
            )
        )

    return tuple(capabilities)


def _detect_llm_api_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect LLM API and generative AI capabilities.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected LLM API capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # Google Generative AI (Gemini)
    if has_dependency(deps_tuple, "google-generativeai"):
        capabilities.append(
            CodebaseCapability(
                name="gemini_api",
                strength="strong",
                tags=(
                    "llm",
                    "gemini",
                    "google",
                    "generative-ai",
                    "text-generation",
                    "vertex-ai",
                ),
                description="Google Gemini API for generative AI",
            )
        )

    # Google Cloud AI Platform / Vertex AI
    if has_dependency(deps_tuple, "google-cloud-aiplatform"):
        capabilities.append(
            CodebaseCapability(
                name="vertex_ai",
                strength="strong",
                tags=(
                    "llm",
                    "vertex-ai",
                    "google-cloud",
                    "ml-platform",
                    "gemini",
                ),
                description="Google Vertex AI for ML model deployment and LLMs",
            )
        )

    # Anthropic Claude
    if has_dependency(deps_tuple, "anthropic"):
        capabilities.append(
            CodebaseCapability(
                name="anthropic_claude",
                strength="strong",
                tags=("llm", "claude", "anthropic", "generative-ai", "text-generation"),
                description="Anthropic Claude API for generative AI",
            )
        )

    # OpenAI API (enhanced from existing)
    if has_dependency(deps_tuple, "openai"):
        capabilities.append(
            CodebaseCapability(
                name="openai_api",
                strength="strong",
                tags=(
                    "llm",
                    "openai",
                    "gpt",
                    "generative-ai",
                    "text-generation",
                    "whisper",
                ),
                description="OpenAI API for GPT models and Whisper",
            )
        )

    # LangChain
    if has_dependency(deps_tuple, "langchain"):
        capabilities.append(
            CodebaseCapability(
                name="langchain",
                strength="moderate",
                tags=("llm", "langchain", "orchestration", "agents", "rag"),
                description="LangChain for LLM application orchestration",
            )
        )

    return tuple(capabilities)


def _detect_cloud_capabilities(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[CodebaseCapability, ...]:
    """Detect cloud platform capabilities.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of detected cloud platform capabilities.
    """
    capabilities: list[CodebaseCapability] = []
    deps_tuple = collect_all_dependencies(libs, services)

    # Google Cloud Storage
    if has_dependency(deps_tuple, "google-cloud-storage"):
        capabilities.append(
            CodebaseCapability(
                name="google_cloud_storage",
                strength="moderate",
                tags=("cloud", "google-cloud", "storage", "gcs"),
                description="Google Cloud Storage for object storage",
            )
        )

    # Google Cloud BigQuery
    if has_dependency(deps_tuple, "google-cloud-bigquery"):
        capabilities.append(
            CodebaseCapability(
                name="bigquery",
                strength="moderate",
                tags=("cloud", "google-cloud", "bigquery", "data-warehouse", "sql"),
                description="Google BigQuery for data warehousing",
            )
        )

    # AWS SDK (boto3)
    if has_dependency(deps_tuple, "boto3"):
        capabilities.append(
            CodebaseCapability(
                name="aws_sdk",
                strength="moderate",
                tags=("cloud", "aws", "boto3", "s3"),
                description="AWS SDK for cloud services",
            )
        )

    # Azure SDK
    if has_dependency(deps_tuple, "azure-core"):
        capabilities.append(
            CodebaseCapability(
                name="azure_sdk",
                strength="moderate",
                tags=("cloud", "azure", "microsoft"),
                description="Azure SDK for Microsoft cloud services",
            )
        )

    return tuple(capabilities)


def _detect_web_frameworks(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect web frameworks in use.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of web framework names.
    """
    frameworks: set[str] = set()
    deps_tuple = collect_all_dependencies(libs, services)

    if has_dependency(deps_tuple, "fastapi"):
        frameworks.add("fastapi")
    if has_dependency(deps_tuple, "flask"):
        frameworks.add("flask")
    if has_dependency(deps_tuple, "django"):
        frameworks.add("django")
    if has_dependency(deps_tuple, "starlette"):
        frameworks.add("starlette")
    if has_dependency(deps_tuple, "aiohttp"):
        frameworks.add("aiohttp")

    return tuple(sorted(frameworks))


def _detect_technologies(
    libs: tuple[LibInfo, ...],
    services: tuple[ServiceInfo, ...],
) -> tuple[str, ...]:
    """Detect programming languages and technologies.

    Args:
        libs: Scanned library information.
        services: Scanned service information.

    Returns:
        Tuple of technology names.
    """
    technologies: set[str] = set()

    # All our libs/services are Python
    if libs or services:
        technologies.add("python")

    return tuple(sorted(technologies))


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
    observability_caps = _detect_observability_capabilities(libs, services)
    streaming_caps = _detect_streaming_capabilities(libs, services)
    llm_api_caps = _detect_llm_api_capabilities(libs, services)
    cloud_caps = _detect_cloud_capabilities(libs, services)

    all_caps = (
        ml_caps
        + cv_caps
        + transformers_caps
        + nlp_caps
        + domain_caps
        + observability_caps
        + streaming_caps
        + llm_api_caps
        + cloud_caps
    )

    return CodebaseProfile(
        capabilities=all_caps,
        technologies=_detect_technologies(libs, services),
        frameworks=_detect_web_frameworks(libs, services),
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
