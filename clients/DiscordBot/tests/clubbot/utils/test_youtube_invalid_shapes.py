from __future__ import annotations

import logging

import pytest
from platform_core.errors import AppError

from clubbot.utils.youtube import extract_video_id


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/shorts/too_short",
        "https://www.youtube.com/live/too_short",
        "https://youtu.be/too_short",
        "https://youtu.be/",  # empty id
        "https://www.youtube.com/channel/ABcDeFgHi_J",  # unrecognized path
    ],
)
def test_invalid_ids_for_various_paths(url: str) -> None:
    with pytest.raises(AppError):
        _ = extract_video_id(url)


logger = logging.getLogger(__name__)
