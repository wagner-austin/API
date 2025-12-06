from __future__ import annotations

import pytest
from PIL import Image

from handwriting_ai.training.augment import maybe_morph


def test_maybe_morph_raises_on_invalid_op() -> None:
    img = Image.new("L", (2, 2), color=0)
    with pytest.raises(ValueError):
        _ = maybe_morph(img, "invalid", 1)
