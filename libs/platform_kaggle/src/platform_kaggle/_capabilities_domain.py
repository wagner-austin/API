"""capabilities: _detect_domain_capabilities and related definitions."""

from __future__ import annotations

from platform_codebase import (
    CodebaseCapability,
    LibInfo,
    ServiceInfo,
    collect_all_dependencies,
    has_dependency,
)


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
