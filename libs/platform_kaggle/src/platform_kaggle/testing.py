"""Public test utilities for platform_kaggle consumers.

The hooks container, the fake Kaggle API implementations and the payload
factories live in the private _testing_* modules; this module is the public
surface consumers import."""

from __future__ import annotations

from platform_kaggle._testing_factories import (
    make_fake_capability,
    make_fake_competition,
    make_fake_competition_page,
    make_fake_competition_pages,
    make_fake_kaggle_competition,
    make_fake_profile,
)
from platform_kaggle._testing_fakes import (
    FakeApiTag,
    FakeCompetitionsResponse,
    FakeKaggleApi,
    FakeKaggleClient,
    FakeKaggleCompetition,
    FakeKagglePageFetcher,
)
from platform_kaggle._testing_hooks import (
    HooksContainer,
    hooks,
    reset_hooks,
)
from platform_kaggle.types import (
    KaggleApiFactoryProtocol,
    KaggleApiProtocol,
)

__all__ = [
    "FakeApiTag",
    "FakeCompetitionsResponse",
    "FakeKaggleApi",
    "FakeKaggleClient",
    "FakeKaggleCompetition",
    "FakeKagglePageFetcher",
    "HooksContainer",
    "KaggleApiFactoryProtocol",
    "KaggleApiProtocol",
    "hooks",
    "make_fake_capability",
    "make_fake_competition",
    "make_fake_competition_page",
    "make_fake_competition_pages",
    "make_fake_kaggle_competition",
    "make_fake_profile",
    "reset_hooks",
]
