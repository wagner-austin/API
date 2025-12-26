from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue
from platform_core.testing import FakeHttpxAsyncClient, FakeHttpxResponse

from github_stats_api.client import (
    GitHubClient,
    _aggregate_languages,
    _check_graphql_errors,
    _count_stars,
    _get_dict,
    _get_int,
    _get_list,
    _get_str,
    _get_user_from_response,
)


class TestGetDict:
    """Tests for _get_dict helper."""

    def test_get_dict_returns_nested_dict(self) -> None:
        """Test _get_dict returns nested dict."""
        data: dict[str, JSONValue] = {"key": {"nested": "value"}}
        result = _get_dict(data, "key")
        assert result == {"nested": "value"}

    def test_get_dict_returns_empty_for_missing_key(self) -> None:
        """Test _get_dict returns empty dict for missing key."""
        data: dict[str, JSONValue] = {"other": "value"}
        result = _get_dict(data, "key")
        assert result == {}

    def test_get_dict_returns_empty_for_none_value(self) -> None:
        """Test _get_dict returns empty dict for None value."""
        data: dict[str, JSONValue] = {"key": None}
        result = _get_dict(data, "key")
        assert result == {}

    def test_get_dict_returns_empty_for_non_dict_value(self) -> None:
        """Test _get_dict returns empty dict for non-dict value."""
        data: dict[str, JSONValue] = {"key": "string_value"}
        result = _get_dict(data, "key")
        assert result == {}


class TestGetList:
    """Tests for _get_list helper."""

    def test_get_list_returns_list(self) -> None:
        """Test _get_list returns list."""
        data: dict[str, JSONValue] = {"key": [1, 2, 3]}
        result = _get_list(data, "key")
        assert result == [1, 2, 3]

    def test_get_list_returns_empty_for_missing_key(self) -> None:
        """Test _get_list returns empty list for missing key."""
        data: dict[str, JSONValue] = {"other": "value"}
        result = _get_list(data, "key")
        assert result == []

    def test_get_list_returns_empty_for_none_value(self) -> None:
        """Test _get_list returns empty list for None value."""
        data: dict[str, JSONValue] = {"key": None}
        result = _get_list(data, "key")
        assert result == []

    def test_get_list_returns_empty_for_non_list_value(self) -> None:
        """Test _get_list returns empty list for non-list value."""
        data: dict[str, JSONValue] = {"key": "string_value"}
        result = _get_list(data, "key")
        assert result == []


class TestGetInt:
    """Tests for _get_int helper."""

    def test_get_int_returns_int(self) -> None:
        """Test _get_int returns int."""
        data: dict[str, JSONValue] = {"key": 42}
        result = _get_int(data, "key")
        assert result == 42

    def test_get_int_returns_default_for_missing_key(self) -> None:
        """Test _get_int returns default for missing key."""
        data: dict[str, JSONValue] = {"other": "value"}
        result = _get_int(data, "key", 10)
        assert result == 10

    def test_get_int_returns_default_for_bool_value(self) -> None:
        """Test _get_int returns default for bool value."""
        data: dict[str, JSONValue] = {"key": True}
        result = _get_int(data, "key", 0)
        assert result == 0

    def test_get_int_returns_default_for_string_value(self) -> None:
        """Test _get_int returns default for string value."""
        data: dict[str, JSONValue] = {"key": "not_an_int"}
        result = _get_int(data, "key", 5)
        assert result == 5


class TestGetStr:
    """Tests for _get_str helper."""

    def test_get_str_returns_string(self) -> None:
        """Test _get_str returns string."""
        data: dict[str, JSONValue] = {"key": "hello"}
        result = _get_str(data, "key")
        assert result == "hello"

    def test_get_str_returns_default_for_missing_key(self) -> None:
        """Test _get_str returns default for missing key."""
        data: dict[str, JSONValue] = {"other": "value"}
        result = _get_str(data, "key", "default")
        assert result == "default"

    def test_get_str_returns_default_for_non_string_value(self) -> None:
        """Test _get_str returns default for non-string value."""
        data: dict[str, JSONValue] = {"key": 123}
        result = _get_str(data, "key", "fallback")
        assert result == "fallback"


class TestCheckGraphqlErrors:
    """Tests for _check_graphql_errors helper."""

    def test_check_graphql_errors_no_errors(self) -> None:
        """Test _check_graphql_errors does nothing when no errors."""
        data: dict[str, JSONValue] = {"data": {"user": {}}}
        _check_graphql_errors(data, "testuser")  # Should not raise

    def test_check_graphql_errors_empty_errors_list(self) -> None:
        """Test _check_graphql_errors does nothing for empty errors list."""
        data: dict[str, JSONValue] = {"data": {}, "errors": []}
        _check_graphql_errors(data, "testuser")  # Should not raise

    def test_check_graphql_errors_user_not_found(self) -> None:
        """Test _check_graphql_errors raises for user not found."""
        data: dict[str, JSONValue] = {
            "errors": [{"message": "Could not resolve to a User with login 'unknown'"}]
        }
        with pytest.raises(AppError) as exc_info:
            _check_graphql_errors(data, "unknown")
        assert exc_info.value.http_status == 404
        assert "not found" in exc_info.value.message

    def test_check_graphql_errors_other_error(self) -> None:
        """Test _check_graphql_errors raises for other errors."""
        data: dict[str, JSONValue] = {"errors": [{"message": "Rate limit exceeded"}]}
        with pytest.raises(AppError) as exc_info:
            _check_graphql_errors(data, "testuser")
        assert exc_info.value.http_status == 502
        assert "Rate limit" in exc_info.value.message

    def test_check_graphql_errors_non_dict_error(self) -> None:
        """Test _check_graphql_errors handles non-dict error."""
        data: dict[str, JSONValue] = {"errors": ["string_error"]}
        with pytest.raises(AppError) as exc_info:
            _check_graphql_errors(data, "testuser")
        assert exc_info.value.http_status == 502
        assert "Unknown error" in exc_info.value.message

    def test_check_graphql_errors_dict_with_non_string_message(self) -> None:
        """Test _check_graphql_errors handles dict with non-string message."""
        data: dict[str, JSONValue] = {"errors": [{"message": 12345}]}
        with pytest.raises(AppError) as exc_info:
            _check_graphql_errors(data, "testuser")
        assert exc_info.value.http_status == 502
        assert "Unknown error" in exc_info.value.message


class TestGetUserFromResponse:
    """Tests for _get_user_from_response helper."""

    def test_get_user_from_response_returns_user(self) -> None:
        """Test _get_user_from_response returns user dict."""
        data: dict[str, JSONValue] = {"data": {"user": {"login": "testuser"}}}
        result = _get_user_from_response(data, "testuser")
        assert result == {"login": "testuser"}

    def test_get_user_from_response_raises_for_missing_user(self) -> None:
        """Test _get_user_from_response raises when user is missing."""
        data: dict[str, JSONValue] = {"data": {"user": None}}
        with pytest.raises(AppError) as exc_info:
            _get_user_from_response(data, "testuser")
        assert exc_info.value.http_status == 404

    def test_get_user_from_response_raises_for_non_dict_user(self) -> None:
        """Test _get_user_from_response raises when user is not dict."""
        data: dict[str, JSONValue] = {"data": {"user": "not_a_dict"}}
        with pytest.raises(AppError) as exc_info:
            _get_user_from_response(data, "testuser")
        assert exc_info.value.http_status == 404


class TestCountStars:
    """Tests for _count_stars helper."""

    def test_count_stars_sums_correctly(self) -> None:
        """Test _count_stars sums stargazer counts."""
        repos: list[JSONValue] = [
            {"stargazerCount": 10},
            {"stargazerCount": 20},
            {"stargazerCount": 30},
        ]
        result = _count_stars(repos)
        assert result == 60

    def test_count_stars_handles_missing_count(self) -> None:
        """Test _count_stars handles missing stargazerCount."""
        repos: list[JSONValue] = [
            {"stargazerCount": 10},
            {"other": "field"},
            {"stargazerCount": 30},
        ]
        result = _count_stars(repos)
        assert result == 40

    def test_count_stars_handles_non_dict_repo(self) -> None:
        """Test _count_stars handles non-dict repo entries."""
        repos: list[JSONValue] = [
            {"stargazerCount": 10},
            "not_a_dict",
            {"stargazerCount": 30},
        ]
        result = _count_stars(repos)
        assert result == 40


class TestAggregateLanguages:
    """Tests for _aggregate_languages helper."""

    def test_aggregate_languages_basic(self) -> None:
        """Test _aggregate_languages aggregates correctly."""
        repos: list[JSONValue] = [
            {
                "languages": {
                    "edges": [
                        {"size": 1000, "node": {"name": "Python", "color": "#3572A5"}},
                        {"size": 500, "node": {"name": "JavaScript", "color": "#f1e05a"}},
                    ]
                }
            },
            {
                "languages": {
                    "edges": [
                        {"size": 2000, "node": {"name": "Python", "color": "#3572A5"}},
                    ]
                }
            },
        ]
        sizes, colors = _aggregate_languages(repos)
        assert sizes["Python"] == 3000
        assert sizes["JavaScript"] == 500
        assert colors["Python"] == "#3572A5"

    def test_aggregate_languages_handles_missing_color(self) -> None:
        """Test _aggregate_languages handles missing color."""
        repos: list[JSONValue] = [
            {
                "languages": {
                    "edges": [
                        {"size": 1000, "node": {"name": "UnknownLang", "color": ""}},
                    ]
                }
            }
        ]
        sizes, _ = _aggregate_languages(repos)
        assert sizes["UnknownLang"] == 1000

    def test_aggregate_languages_handles_non_dict_edge(self) -> None:
        """Test _aggregate_languages handles non-dict edge."""
        repos: list[JSONValue] = [
            {
                "languages": {
                    "edges": [
                        "not_a_dict",
                        {"size": 1000, "node": {"name": "Python", "color": "#3572A5"}},
                    ]
                }
            }
        ]
        sizes, _ = _aggregate_languages(repos)
        assert sizes["Python"] == 1000

    def test_aggregate_languages_handles_non_dict_repo(self) -> None:
        """Test _aggregate_languages handles non-dict repo."""
        repos: list[JSONValue] = [
            "not_a_dict",
            {
                "languages": {
                    "edges": [
                        {"size": 1000, "node": {"name": "Python", "color": "#3572A5"}},
                    ]
                }
            },
        ]
        sizes, _ = _aggregate_languages(repos)
        assert sizes["Python"] == 1000

    def test_aggregate_languages_skips_empty_name(self) -> None:
        """Test _aggregate_languages skips languages with empty name."""
        repos: list[JSONValue] = [
            {
                "languages": {
                    "edges": [
                        {"size": 1000, "node": {"name": "", "color": "#3572A5"}},
                        {"size": 500, "node": {"name": "JavaScript", "color": "#f1e05a"}},
                    ]
                }
            },
        ]
        sizes, _colors = _aggregate_languages(repos)
        assert "JavaScript" in sizes
        assert sizes["JavaScript"] == 500
        assert "" not in sizes


class TestGitHubClient:
    """Tests for GitHubClient class."""

    async def test_fetch_user_stats_success(self) -> None:
        """Test fetch_user_stats returns user data."""
        response_data: dict[str, JSONValue] = {
            "data": {
                "user": {
                    "login": "testuser",
                    "name": "Test User",
                    "contributionsCollection": {
                        "totalCommitContributions": 100,
                        "restrictedContributionsCount": 10,
                    },
                    "pullRequests": {"totalCount": 20},
                    "openIssues": {"totalCount": 5},
                    "closedIssues": {"totalCount": 5},
                    "repositories": {
                        "nodes": [
                            {"stargazerCount": 50},
                        ]
                    },
                    "repositoriesContributedTo": {"totalCount": 15},
                }
            }
        }
        fake_response = FakeHttpxResponse(200, response_data)
        fake_client = FakeHttpxAsyncClient(fake_response)
        gh = GitHubClient("test-token", fake_client)

        result = await gh.fetch_user_stats("testuser")

        assert result["login"] == "testuser"
        assert result["total_commits"] == 110
        assert result["total_prs"] == 20
        assert result["total_issues"] == 10
        assert result["total_stars"] == 50

    async def test_fetch_user_stats_api_error(self) -> None:
        """Test fetch_user_stats raises on API error."""
        fake_response = FakeHttpxResponse(500, {"error": "Internal error"})
        fake_client = FakeHttpxAsyncClient(fake_response)
        gh = GitHubClient("test-token", fake_client)

        with pytest.raises(AppError) as exc_info:
            await gh.fetch_user_stats("testuser")
        assert exc_info.value.http_status == 502

    async def test_fetch_user_stats_invalid_response_format(self) -> None:
        """Test fetch_user_stats raises on invalid response format."""
        # Response that's not a dict will be rejected
        fake_response = FakeHttpxResponse(200, [1, 2, 3])
        fake_client = FakeHttpxAsyncClient(fake_response)
        gh = GitHubClient("test-token", fake_client)

        with pytest.raises(AppError) as exc_info:
            await gh.fetch_user_stats("testuser")
        assert exc_info.value.http_status == 502

    async def test_fetch_languages_success(self) -> None:
        """Test fetch_languages returns language data."""
        response_data: dict[str, JSONValue] = {
            "data": {
                "user": {
                    "repositories": {
                        "nodes": [
                            {
                                "languages": {
                                    "edges": [
                                        {
                                            "size": 5000,
                                            "node": {"name": "Python", "color": "#3572A5"},
                                        },
                                        {
                                            "size": 3000,
                                            "node": {"name": "JavaScript", "color": "#f1e05a"},
                                        },
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
        fake_response = FakeHttpxResponse(200, response_data)
        fake_client = FakeHttpxAsyncClient(fake_response)
        gh = GitHubClient("test-token", fake_client)

        result = await gh.fetch_languages("testuser")

        assert len(result) == 2
        assert result[0]["name"] == "Python"
        assert result[0]["size"] == 5000
        assert result[1]["name"] == "JavaScript"
        assert result[1]["size"] == 3000

    async def test_fetch_user_stats_uses_login_as_name_fallback(self) -> None:
        """Test fetch_user_stats uses login as name when name is empty."""
        response_data: dict[str, JSONValue] = {
            "data": {
                "user": {
                    "login": "testuser",
                    "name": "",
                    "contributionsCollection": {
                        "totalCommitContributions": 100,
                        "restrictedContributionsCount": 0,
                    },
                    "pullRequests": {"totalCount": 0},
                    "openIssues": {"totalCount": 0},
                    "closedIssues": {"totalCount": 0},
                    "repositories": {"nodes": []},
                    "repositoriesContributedTo": {"totalCount": 0},
                }
            }
        }
        fake_response = FakeHttpxResponse(200, response_data)
        fake_client = FakeHttpxAsyncClient(fake_response)
        gh = GitHubClient("test-token", fake_client)

        result = await gh.fetch_user_stats("testuser")

        assert result["name"] == "testuser"
