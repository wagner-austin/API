from __future__ import annotations

import pytest

from procart.exceptions import ConfigError


def test_config_error_raised_with_message() -> None:
    with pytest.raises(ConfigError, match="bad config"):
        raise ConfigError("bad config")
