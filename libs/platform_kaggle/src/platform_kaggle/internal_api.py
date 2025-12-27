"""Kaggle internal API client for fetching competition pages.

This module provides access to Kaggle's internal gRPC-web API to fetch
full competition page content (Description, Evaluation, Timeline, Rules).
"""

from __future__ import annotations

from http.cookiejar import CookieJar
from typing import Protocol
from urllib.request import HTTPCookieProcessor, Request, build_opener

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

from platform_kaggle.types import (
    CompetitionPage,
    CompetitionPages,
)

# -----------------------------------------------------------------------------
# Protocols for HTTP responses
# -----------------------------------------------------------------------------


class HTTPResponseProtocol(Protocol):
    """Protocol for HTTP response objects returned by urllib.

    This protocol defines the minimal interface needed for reading
    response data from urllib.request operations.
    """

    def read(self) -> bytes:
        """Read the response body.

        Returns:
            Response body as bytes.
        """
        ...


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_KAGGLE_BASE_URL = "https://www.kaggle.com"
_LIST_PAGES_ENDPOINT = "/api/i/competitions.PageService/ListPages"
_GET_COMPETITION_ENDPOINT = "/api/v1/competitions/get"


# -----------------------------------------------------------------------------
# Response Parsing Helpers
# -----------------------------------------------------------------------------


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict.

    Args:
        value: JSON value to check.
        context: Context for error message.

    Returns:
        The value as JSONObject.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _parse_page_from_response(data: JSONObject, index: int) -> CompetitionPage:
    """Parse a single page from API response.

    Args:
        data: Page data from API response.
        index: Index in the pages array (for error messages).

    Returns:
        Parsed CompetitionPage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return CompetitionPage(
        id=require_int(data, "id"),
        name=require_str(data, "name"),
        content=require_str(data, "content"),
    )


def _parse_pages_response(
    data: JSONObject,
    competition_id: int,
) -> CompetitionPages:
    """Parse ListPages API response into CompetitionPages.

    Args:
        data: Raw API response data.
        competition_id: The competition ID that was queried.

    Returns:
        Parsed CompetitionPages.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    pages_raw = require_list(data, "pages")
    pages: list[CompetitionPage] = []

    for i, page_data in enumerate(pages_raw):
        page_obj = _require_dict_value(page_data, f"pages[{i}]")
        pages.append(_parse_page_from_response(page_obj, i))

    # Extract common page content by name
    description = ""
    evaluation = ""
    timeline = ""
    rules = ""

    for page in pages:
        name_lower = page.name.lower()
        if name_lower == "description" or name_lower == "overview":
            description = page.content
        elif name_lower == "evaluation":
            evaluation = page.content
        elif name_lower == "timeline":
            timeline = page.content
        elif name_lower == "rules":
            rules = page.content

    return CompetitionPages(
        competition_id=competition_id,
        pages=tuple(pages),
        description=description,
        evaluation=evaluation,
        timeline=timeline,
        rules=rules,
    )


# -----------------------------------------------------------------------------
# Session Management
# -----------------------------------------------------------------------------


class KaggleSession:
    """Manages Kaggle session cookies and XSRF token.

    Creates a session by visiting Kaggle to obtain cookies, then extracts
    the XSRF token for authenticated API requests.

    Attributes:
        cookies: Cookie jar with session cookies.
        xsrf_token: XSRF token for API authentication.
    """

    __slots__ = ("_cookie_jar", "_opener", "_xsrf_token")

    def __init__(self) -> None:
        """Initialize session.

        Creates cookie jar and HTTP opener for session management.
        """
        self._cookie_jar = CookieJar()
        self._opener = build_opener(HTTPCookieProcessor(self._cookie_jar))
        self._xsrf_token: str | None = None

    def initialize(self) -> None:
        """Initialize session by visiting Kaggle homepage.

        Fetches the homepage to obtain session cookies including XSRF token.

        Raises:
            RuntimeError: If unable to obtain XSRF token.
        """
        # Visit homepage to get cookies
        request = Request(_KAGGLE_BASE_URL)
        request.add_header("User-Agent", "Mozilla/5.0 (compatible; KaggleClient/1.0)")
        self._opener.open(request)

        # Extract XSRF token from cookies
        self._xsrf_token = self._extract_xsrf_token()
        if self._xsrf_token is None:
            raise RuntimeError("Failed to obtain XSRF token from Kaggle")

    def _extract_xsrf_token(self) -> str | None:
        """Extract XSRF token from cookies.

        Returns:
            XSRF token string, or None if not found.
        """
        for cookie in self._cookie_jar:
            if cookie.name == "XSRF-TOKEN":
                return cookie.value
        return None

    @property
    def xsrf_token(self) -> str:
        """Get the XSRF token.

        Returns:
            XSRF token string.

        Raises:
            RuntimeError: If session not initialized.
        """
        if self._xsrf_token is None:
            raise RuntimeError("Session not initialized. Call initialize() first.")
        return self._xsrf_token

    def request(
        self,
        url: str,
        data: bytes | None = None,
        content_type: str = "application/x-www-form-urlencoded",
    ) -> bytes:
        """Make an authenticated request.

        Args:
            url: Full URL to request.
            data: POST data (if None, makes GET request).
            content_type: Content-Type header value.

        Returns:
            Response body as bytes.

        Raises:
            RuntimeError: If session not initialized.
            URLError: If request fails.
        """
        if self._xsrf_token is None:
            raise RuntimeError("Session not initialized. Call initialize() first.")

        request = Request(url, data=data)
        request.add_header("User-Agent", "Mozilla/5.0 (compatible; KaggleClient/1.0)")
        request.add_header("x-xsrf-token", self._xsrf_token)
        request.add_header("Content-Type", content_type)

        response: HTTPResponseProtocol = self._opener.open(request)
        return response.read()


# -----------------------------------------------------------------------------
# Page Fetcher
# -----------------------------------------------------------------------------


class KagglePageFetcher:
    """Fetches competition pages from Kaggle's internal API.

    Uses Kaggle's internal gRPC-web API to fetch full competition page content
    including Description, Evaluation, Timeline, and Rules.
    """

    __slots__ = ("_session",)

    def __init__(self, session: KaggleSession) -> None:
        """Initialize page fetcher with session.

        Args:
            session: Initialized Kaggle session.
        """
        self._session = session

    def fetch_pages(self, competition_id: int) -> CompetitionPages:
        """Fetch all pages for a competition.

        Args:
            competition_id: Numeric Kaggle competition ID.

        Returns:
            CompetitionPages containing all page content.

        Raises:
            RuntimeError: If the API request fails.
            JSONTypeError: If response parsing fails.
        """
        url = f"{_KAGGLE_BASE_URL}{_LIST_PAGES_ENDPOINT}"
        json_data = dump_json_str({"competitionId": competition_id}).encode("utf-8")

        response_bytes = self._session.request(url, json_data, content_type="application/json")
        response_text = response_bytes.decode("utf-8")

        # Parse JSON response
        response_value = load_json_str(response_text)
        response_data = narrow_json_to_dict(response_value)

        return _parse_pages_response(response_data, competition_id)

    def get_competition_id(self, slug: str) -> int:
        """Get the numeric competition ID from a slug.

        Uses Kaggle's v1 REST API to fetch competition details and extract
        the numeric ID.

        Args:
            slug: Competition slug (e.g., "titanic", "google-gemma-3n-hackathon").

        Returns:
            Numeric competition ID.

        Raises:
            URLError: If the API request fails (e.g., 404 for invalid slug).
            JSONTypeError: If response parsing fails.
        """
        url = f"{_KAGGLE_BASE_URL}{_GET_COMPETITION_ENDPOINT}/{slug}"

        response_bytes = self._session.request(url)
        response_text = response_bytes.decode("utf-8")

        response_value = load_json_str(response_text)
        response_data = narrow_json_to_dict(response_value)

        return require_int(response_data, "id")


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_page_fetcher() -> KagglePageFetcher:
    """Create an initialized page fetcher.

    Creates a new Kaggle session, initializes it, and returns
    a page fetcher ready to use.

    Returns:
        Initialized KagglePageFetcher.

    Raises:
        RuntimeError: If session initialization fails.
    """
    session = KaggleSession()
    session.initialize()
    return KagglePageFetcher(session)


__all__ = [
    "KagglePageFetcher",
    "KaggleSession",
    "create_page_fetcher",
]
