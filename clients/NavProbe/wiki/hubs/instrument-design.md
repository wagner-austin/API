# Instrument Design

Why the instrument is built the way it is. Canonical byte encoding, digest folding, the record formats, the layering, and the dependency-injection boundaries.

The recurring theme is *injectivity*: a determinism instrument's core obligation is that two different rollouts never produce the same digest. Most of the design decisions here trace back to that, and at least one of them was learned the hard way after a collision shipped behind a passing test.

[Digest folding requires a length prefix per element](../pages/digest-fold-requires-length-prefix.md) -- concatenating step digests collides; the step count alone does not close the gap
[A test can encode the right intent and check the wrong case](../pages/passing-test-can-miss-its-own-premise.md) -- the collision above shipped green because the test picked the one case another prefix already covered
[A cross-process comparison is a free function because nothing is left to inject](../pages/fresh-process-comparison-has-nothing-to-inject.md) -- why the fresh-process layer takes paths rather than a service, and why persistence had no consumer until it existed
[A scene is a value, so a result can cite it instead of describing it](../pages/a-scene-is-a-value-not-a-string-literal.md) -- MJCF in a string literal made two harnesses incomparable; making the scene data is what exposed a boundary that had moved
[Bit-equality is a leading indicator, and needs a magnitude beside it](../pages/bit-equality-is-a-leading-indicator.md) -- why the package measures spread as well as agreement, and why a NaN spread reads as a pass
[The numbers here are scene-dependent; the shapes are what replicate](../pages/the-numbers-are-scene-dependent-the-shapes-replicate.md) -- two findings re-measured, both shapes held, every figure moved; how to read a number on this wiki
[Open questions, and what would answer each one](../pages/open-questions-and-what-would-answer-them.md) -- the consolidated view of what is not known, ordered by value, with the experiment that would settle each
