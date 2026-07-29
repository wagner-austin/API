"""The test suite.

A package rather than a bare directory so the shared fixtures in
:mod:`tests.wire_fixtures` can be imported by name. Every test that needs a
world builds it there, which is what keeps a new wire field from being a
thirteen-file change.
"""
