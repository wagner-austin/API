from __future__ import annotations

from PIL import Image

from handwriting_ai import _test_hooks
from handwriting_ai.training.augment import maybe_add_dots


def test_maybe_add_dots_out_of_bounds_path() -> None:
    img = Image.new("L", (2, 2), 0)

    class _Rng:
        def __init__(self) -> None:
            self.n = 0

        def random(self) -> float:
            # First call decide to draw dots (p threshold), then choose val
            self.n += 1
            return 0.0  # always below any p

        def randint(self, a: int, b: int) -> int:
            # Place dot in bottom-right corner to force some out-of-bounds within the s x s loop
            _ = (a, b)
            return 1

    rng = _Rng()
    _test_hooks.random_random = rng.random
    _test_hooks.random_randint = rng.randint

    _ = maybe_add_dots(img, prob=1.0, count=1, size_px=3)
    # No assertion needed; coverage exercises the false branch of bounds check
