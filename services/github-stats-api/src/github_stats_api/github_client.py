from __future__ import annotations

from typing import Protocol

from typing_extensions import TypedDict

# Language colors mapping (subset of common languages)
_LANGUAGE_COLORS: dict[str, str] = {
    "Python": "#3572A5",
    "JavaScript": "#f1e05a",
    "TypeScript": "#3178c6",
    "Java": "#b07219",
    "C++": "#f34b7d",
    "C": "#555555",
    "C#": "#178600",
    "Go": "#00ADD8",
    "Rust": "#dea584",
    "Ruby": "#701516",
    "PHP": "#4F5D95",
    "Swift": "#F05138",
    "Kotlin": "#A97BFF",
    "Scala": "#c22d40",
    "R": "#198CE7",
    "Shell": "#89e051",
    "PowerShell": "#012456",
    "HTML": "#e34c26",
    "CSS": "#563d7c",
    "SCSS": "#c6538c",
    "Vue": "#41b883",
    "Svelte": "#ff3e00",
    "Jupyter Notebook": "#DA5B0B",
    "Dockerfile": "#384d54",
    "Makefile": "#427819",
    "Lua": "#000080",
    "Dart": "#00B4AB",
    "Elixir": "#6e4a7e",
    "Haskell": "#5e5086",
    "OCaml": "#3be133",
    "F#": "#b845fc",
    "Clojure": "#db5855",
    "Erlang": "#B83998",
    "Julia": "#a270ba",
    "MATLAB": "#e16737",
    "Perl": "#0298c3",
    "Assembly": "#6E4C13",
    "Vim Script": "#199f4b",
    "TeX": "#3D6117",
    "Objective-C": "#438eff",
    "Groovy": "#4298b8",
    "Solidity": "#AA6746",
    "GLSL": "#5686a5",
    "CUDA": "#3A4E3A",
    "HCL": "#844FBA",
    "Nix": "#7e7eff",
    "Zig": "#ec915c",
}


def get_language_color(language: str) -> str:
    """Get the color for a programming language.

    Args:
        language: Language name.

    Returns:
        Hex color code.
    """
    return _LANGUAGE_COLORS.get(language, "#858585")


class GitHubUserData(TypedDict, total=True):
    """Raw user data from GitHub API.

    Attributes:
        login: GitHub username.
        name: Display name or empty string.
        total_commits: Total commit count.
        total_prs: Total PR count.
        total_issues: Total issue count.
        total_stars: Total stars received.
        total_contributions: Total contributions.
    """

    login: str
    name: str
    total_commits: int
    total_prs: int
    total_issues: int
    total_stars: int
    total_contributions: int


class GitHubLanguageData(TypedDict, total=True):
    """Language data from GitHub API.

    Attributes:
        name: Language name.
        size: Bytes of code.
        color: Hex color.
    """

    name: str
    size: int
    color: str


class GitHubClientProto(Protocol):
    """Protocol for GitHub API client."""

    async def fetch_user_stats(self, username: str) -> GitHubUserData:
        """Fetch user statistics from GitHub API.

        Args:
            username: GitHub username.

        Returns:
            User statistics data.
        """
        ...

    async def fetch_languages(self, username: str) -> list[GitHubLanguageData]:
        """Fetch language statistics from GitHub API.

        Args:
            username: GitHub username.

        Returns:
            List of language statistics.
        """
        ...


# GraphQL query for user stats
_USER_STATS_QUERY = """
query userInfo($login: String!) {
  user(login: $login) {
    login
    name
    contributionsCollection {
      totalCommitContributions
      restrictedContributionsCount
    }
    repositoriesContributedTo(
      first: 1
      contributionTypes: [COMMIT, ISSUE, PULL_REQUEST, REPOSITORY]
    ) {
      totalCount
    }
    pullRequests(first: 1) {
      totalCount
    }
    openIssues: issues(states: OPEN) {
      totalCount
    }
    closedIssues: issues(states: CLOSED) {
      totalCount
    }
    repositories(
      first: 100
      ownerAffiliations: OWNER
      orderBy: {field: STARGAZERS, direction: DESC}
    ) {
      totalCount
      nodes {
        stargazerCount
      }
    }
  }
}
"""

# GraphQL query for languages
_LANGUAGES_QUERY = """
query userLanguages($login: String!) {
  user(login: $login) {
    repositories(
      first: 100
      ownerAffiliations: OWNER
      isFork: false
      orderBy: {field: STARGAZERS, direction: DESC}
    ) {
      nodes {
        languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
          edges {
            size
            node {
              name
              color
            }
          }
        }
      }
    }
  }
}
"""


__all__ = [
    "_LANGUAGES_QUERY",
    "_USER_STATS_QUERY",
    "GitHubClientProto",
    "GitHubLanguageData",
    "GitHubUserData",
    "get_language_color",
]
