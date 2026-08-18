"""The test suite.

A package rather than a bare directory so the shared in-repo simulator
implementations in :mod:`tests.simulators` can be imported by name. Those are
real deterministic simulators built for the suite, not mocks: a determinism
instrument tested against a mock would only prove the mock is deterministic.
"""

from __future__ import annotations
