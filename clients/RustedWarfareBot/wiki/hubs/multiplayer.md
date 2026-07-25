# Multiplayer

Rusted Warfare's networked play: the lockstep model, how commands are relayed and stamped to future ticks, desync causes and detection, the default port and relay servers, third-party server implementations, and the constraints multiplayer places on bot design.

Scope: anything involving a second participant. Single-player operation lives in [Headless Harness](headless-harness.md).

Multiplayer is not currently a target — the bot plays single-player skirmishes against the built-in AI. This hub exists anyway because lockstep imposes design constraints that are cheap to honour up front and expensive to retrofit, and because "should we ever want it" was an explicit requirement.

Public relay servers are a separate question from self-hosted play: running a bot against strangers is a social and terms-of-service matter, not just a technical one, and is deliberately out of scope until asked for.

[Multiplayer Portability Invariants](../pages/multiplayer-portability-invariants.md) -- the four rules that keep single-player work from stranding us outside multiplayer

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
