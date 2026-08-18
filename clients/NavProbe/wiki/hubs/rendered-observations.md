# Rendered Observations

The batch renderer: whether the pixel stream a navigation policy consumes is reproducible, and under what conditions. This is the hub the project exists for — the published determinism results for GPU-batched simulators were taken with rendering disabled, and the same papers name perception and sensor rendering as uncovered.

Rendering is a raycaster over a per-step bounding-volume hierarchy. That is a different numerical path from the contact solver those results measured, so nothing about it transfers by implication, and every page here is a measurement rather than an inference.

The short version: the raycaster is exactly reproducible on a fixed device and adds no variance of its own — it only inherits whatever the physics does. Across devices it is a different story: the **depth** channel does not survive a change of device, while colour does, because 8-bit quantisation hides the difference.

[The rendered stream reproduces exactly within a single device](../pages/warp-rendered-stream-is-reproducible-within-a-device.md) -- three rollouts, three batch widths, both devices, both channels; the control that makes the next page interpretable
[The MJX-Warp batch renderer's depth output is not portable across devices](../pages/warp-renderer-depth-is-not-device-portable.md) -- identical inputs, 34% of depth pixels differ, and RGB compares equal throughout
[The raycaster inherits non-determinism, it does not create any](../pages/the-raycaster-inherits-nondeterminism-it-does-not-create-it.md) -- twelve renders of one frozen state agree even in scenes whose physics does not, so a rendered rollout that fails to reproduce is a physics problem
