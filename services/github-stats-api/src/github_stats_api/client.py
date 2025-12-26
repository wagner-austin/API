from __future__ import annotations

from platform_core.errors import AppError, ErrorCode
from platform_core.http_client import HttpxAsyncClient
from platform_core.json_utils import JSONValue, narrow_json_to_dict

from .github_client import (
    _LANGUAGES_QUERY,
    _USER_STATS_QUERY,
    GitHubLanguageData,
    GitHubUserData,
    get_language_color,
)

_GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


def _get_dict(data: dict[str, JSONValue], key: str) -> dict[str, JSONValue]:
    """Get a nested dict from data.

    Args:
        data: Parent dict.
        key: Key to look up.

    Returns:
        Nested dict or empty dict if not found.
    """
    value = data.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _get_list(data: dict[str, JSONValue], key: str) -> list[JSONValue]:
    """Get a list from data.

    Args:
        data: Parent dict.
        key: Key to look up.

    Returns:
        List or empty list if not found.
    """
    value = data.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _get_int(data: dict[str, JSONValue], key: str, default: int = 0) -> int:
    """Get an int from data.

    Args:
        data: Parent dict.
        key: Key to look up.
        default: Default value.

    Returns:
        Int value or default.
    """
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _get_str(data: dict[str, JSONValue], key: str, default: str = "") -> str:
    """Get a string from data.

    Args:
        data: Parent dict.
        key: Key to look up.
        default: Default value.

    Returns:
        String value or default.
    """
    value = data.get(key)
    if isinstance(value, str):
        return value
    return default


def _check_graphql_errors(data: dict[str, JSONValue], username: str) -> None:
    """Check for GraphQL errors in response.

    Args:
        data: Parsed response data.
        username: Username for error messages.

    Raises:
        AppError: If errors are present in the response.
    """
    errors = data.get("errors")
    if not isinstance(errors, list) or len(errors) == 0:
        return
    first_error = errors[0]
    msg = "Unknown error"
    if isinstance(first_error, dict):
        err_msg = first_error.get("message")
        if isinstance(err_msg, str):
            msg = err_msg
    if "Could not resolve to a User" in msg:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message=f"User '{username}' not found",
            http_status=404,
        )
    raise AppError(
        code=ErrorCode.EXTERNAL_SERVICE_ERROR,
        message=f"GitHub API error: {msg}",
        http_status=502,
    )


def _get_user_from_response(data: dict[str, JSONValue], username: str) -> dict[str, JSONValue]:
    """Extract user dict from response data.

    Args:
        data: Parsed response data.
        username: Username for error messages.

    Returns:
        User data dict.

    Raises:
        AppError: If user not found.
    """
    data_field = _get_dict(data, "data")
    user = data_field.get("user")
    if user is None or not isinstance(user, dict):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message=f"User '{username}' not found",
            http_status=404,
        )
    return user


def _count_stars(repos: list[JSONValue]) -> int:
    """Count total stars from repositories.

    Args:
        repos: List of repository dicts.

    Returns:
        Total star count.
    """
    total = 0
    for repo in repos:
        if isinstance(repo, dict):
            total += _get_int(repo, "stargazerCount")
    return total


def _aggregate_languages(
    repos: list[JSONValue],
) -> tuple[dict[str, int], dict[str, str]]:
    """Aggregate language data from repositories.

    Args:
        repos: List of repository dicts.

    Returns:
        Tuple of (language sizes dict, language colors dict).
    """
    lang_sizes: dict[str, int] = {}
    lang_colors: dict[str, str] = {}

    for repo in repos:
        if not isinstance(repo, dict):
            continue
        languages_data = _get_dict(repo, "languages")
        edges = _get_list(languages_data, "edges")
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            size = _get_int(edge, "size")
            node = _get_dict(edge, "node")
            name = _get_str(node, "name", "")
            color = _get_str(node, "color", "")
            if color == "":
                color = get_language_color(name)

            if name:
                lang_sizes[name] = lang_sizes.get(name, 0) + size
                if name not in lang_colors:
                    lang_colors[name] = color

    return lang_sizes, lang_colors


class GitHubClient:
    """HTTP client for GitHub GraphQL API.

    Attributes:
        _token: GitHub personal access token.
        _client: HTTP async client.
    """

    _token: str
    _client: HttpxAsyncClient

    def __init__(self, token: str, client: HttpxAsyncClient) -> None:
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token.
            client: HTTP async client for making requests.
        """
        self._token = token
        self._client = client

    async def _post_graphql(self, query: str, username: str) -> dict[str, JSONValue]:
        """Post GraphQL query and return parsed response.

        Args:
            query: GraphQL query string.
            username: Username for variables.

        Returns:
            Parsed response data.

        Raises:
            AppError: If request fails or response is invalid.
        """
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        payload: JSONValue = {
            "query": query,
            "variables": {"login": username},
        }

        resp = await self._client.post(
            _GITHUB_GRAPHQL_URL,
            json=payload,
            headers=headers,
        )

        if resp.status_code != 200:
            raise AppError(
                code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                message=f"GitHub API returned status {resp.status_code}",
                http_status=502,
            )

        raw_data = resp.json()
        if not isinstance(raw_data, dict):
            raise AppError(
                code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                message="GitHub API returned unexpected response format",
                http_status=502,
            )
        return narrow_json_to_dict(raw_data)

    async def fetch_user_stats(self, username: str) -> GitHubUserData:
        """Fetch user statistics from GitHub GraphQL API.

        Args:
            username: GitHub username to fetch stats for.

        Returns:
            GitHubUserData with aggregated statistics.

        Raises:
            AppError: If the API request fails or user not found.
        """
        data = await self._post_graphql(_USER_STATS_QUERY, username)
        _check_graphql_errors(data, username)
        user = _get_user_from_response(data, username)

        # Extract stats
        contribs = _get_dict(user, "contributionsCollection")
        total_commits = _get_int(contribs, "totalCommitContributions") + _get_int(
            contribs, "restrictedContributionsCount"
        )
        total_prs = _get_int(_get_dict(user, "pullRequests"), "totalCount")
        open_issues = _get_int(_get_dict(user, "openIssues"), "totalCount")
        closed_issues = _get_int(_get_dict(user, "closedIssues"), "totalCount")
        total_issues = open_issues + closed_issues

        repos_data = _get_dict(user, "repositories")
        repos = _get_list(repos_data, "nodes")
        total_stars = _count_stars(repos)

        contribs_to = _get_int(_get_dict(user, "repositoriesContributedTo"), "totalCount")
        total_contributions = total_commits + total_prs + total_issues + contribs_to

        login = _get_str(user, "login", username)
        name = _get_str(user, "name", "") or login

        return {
            "login": login,
            "name": name,
            "total_commits": total_commits,
            "total_prs": total_prs,
            "total_issues": total_issues,
            "total_stars": total_stars,
            "total_contributions": total_contributions,
        }

    async def fetch_languages(self, username: str) -> list[GitHubLanguageData]:
        """Fetch language statistics from GitHub GraphQL API.

        Args:
            username: GitHub username to fetch languages for.

        Returns:
            List of GitHubLanguageData sorted by size descending.

        Raises:
            AppError: If the API request fails or user not found.
        """
        data = await self._post_graphql(_LANGUAGES_QUERY, username)
        _check_graphql_errors(data, username)
        user = _get_user_from_response(data, username)

        repos_data = _get_dict(user, "repositories")
        repos = _get_list(repos_data, "nodes")
        lang_sizes, lang_colors = _aggregate_languages(repos)

        # Sort by size descending
        def _get_size(item: tuple[str, int]) -> int:
            return item[1]

        sorted_langs = sorted(lang_sizes.items(), key=_get_size, reverse=True)

        return [
            {
                "name": name,
                "size": size,
                "color": lang_colors.get(name, "#858585"),
            }
            for name, size in sorted_langs
        ]


__all__ = ["GitHubClient"]
