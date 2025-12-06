# platform_music

Core music analytics library for Music Wrapped.

- Strict typing: TypedDict and Protocol only.
- No Any, no cast, no type: ignore, no dataclasses, no stubs.
- Parse/validate at edges with dedicated decoder functions.
- Pure, side-effect-free analytics; testable with 100% coverage.

This package provides:
- TypedDict models for plays, tracks, history, and wrapped result
- Protocol for music services (Last.fm-first), with a testing fake
- A minimal orchestrator to generate a Wrapped result from history

See tests for usage examples.
