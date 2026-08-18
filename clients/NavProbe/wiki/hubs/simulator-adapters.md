# Simulator Adapters

The boundary between this package and somebody else's simulator. How a vendor API gets typed without importing untyped code, how the declared Protocols are kept honest against the installed version, and the vendor behaviours the adapter has to work with rather than around.

Everything here is version-pinned by construction: a claim about MJX with no version attached has no locator and does not belong on these pages.

[MuJoCo ships no type marker, so its surface is declared as Protocols](../pages/mujoco-untyped-requires-protocol-boundary.md) -- the `__import__`-plus-annotation pattern, and why the drift test calls rather than reads
[vmap requires every pytree leaf to carry a batch axis](../pages/vmap-requires-every-leaf-batched.md) -- replacing one field of an MJX Data does not batch it; the construction must be vectorised instead
[MuJoCo-Warp is a second vendor, not a second MJX](../pages/warp-is-a-separate-vendor-surface.md) -- in-place mutation, nworld as an allocation parameter, no pytree; why it gets its own bindings and what the two adapters do share
