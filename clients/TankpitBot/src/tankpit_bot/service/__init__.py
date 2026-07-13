"""Long-running bot service surface.

Groups the HTTP API, threadsafe cross-thread channels, and session
lifecycle used by the phone-driven bot service. All modules follow the
project-wide invariants (strict mypy, encode/decode for every
TypedDict, Protocol-typed hooks in ``_test_hooks``, propagate — never
soften — errors).
"""
