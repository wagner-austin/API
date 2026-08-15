"""capabilities: _detect_llm_api_capabilities and related definitions."""

from __future__ import annotations

from platform_codebase import (
    CodebaseCapability,
    LibInfo,
    ServiceInfo,
    collect_all_dependencies,
    has_dependency,
)


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
