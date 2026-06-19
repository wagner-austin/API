# Architecture

Codebase design decisions, patterns, and coding standards.

[Inheritance Chain](../pages/inheritance-chain.md) -- Bot -> DispatchMixin -> CompletionsMixin -> SessionBase, composition over inheritance
[Coding Standards](../pages/coding-standards.md) -- no Any/cast/TYPE_CHECKING, no mocks, _test_hooks DI, MonkeyPatchBanRule
