"""Kafka streaming infrastructure for covenant monitoring.

This package provides TypedDict-based Kafka integration for:
- Consuming measurement events from Confluent Cloud
- Producing prediction and alert events
- Strict schema validation with encode/decode functions
- Streaming inference worker for real-time prediction
- Generic streaming worker for multi-domain ML prediction

Modules:
- config: Configuration TypedDicts and environment parsing
- schemas: Kafka event TypedDicts with encode/decode/TypeGuards
- producer: High-level producer wrapper
- consumer: High-level consumer wrapper
- worker: Streaming inference worker (covenant-specific)
- generic_worker: Domain-agnostic streaming worker
- _test_hooks: Dependency injection for testing (private)

Strict typing: no Any, no casts, no type: ignore.
"""
