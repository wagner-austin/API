"""Shared claim and page builders for the physics-claim guard tests."""

from __future__ import annotations

from pathlib import Path

from scripts.physics_claims import (
    CLAIM_FENCE_OPEN,
    run_physics_claim_rules,
)

FIXTURE_PACKAGE = "tests.scripts.physics_fixture"


FIXTURE_MODULE = f"{FIXTURE_PACKAGE}.facts"


#: The exactly-one-of-kinds message, kept in one place so adding a
#: claim kind updates every assertion that quotes it.
_ONE_OF = "claim needs exactly one of value/bytes/members/keys/probes/law"


GREEN_PAGE = f"""# Fixture Economy

{CLAIM_FENCE_OPEN}
{{
  "claims": [
    {{"id": "answer", "code": "{FIXTURE_MODULE}:ANSWER", "value": 42}},
    {{
      "id": "double",
      "code": "{FIXTURE_MODULE}:double",
      "formula": "2 * value",
      "probes": [
        {{"args": [2], "expect": 4}},
        {{"args": [0], "expect": 0}}
      ]
    }}
  ]
}}
```
"""


def _write_page(project_root: Path, name: str, text: str) -> None:
    """Create one wiki page inside a fake project tree.

    Args:
        project_root: Fake project root.
        name: Page filename.
        text: Page markdown text.
    """
    pages_dir = project_root / "wiki" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / name).write_text(text, encoding="utf-8")


def _claims_page(claims_json: str) -> str:
    """Wrap a claims JSON body in a minimal wiki page.

    Args:
        claims_json: The JSON text of the claim block.

    Returns:
        Markdown page text with one fenced claim block.
    """
    return f"# Page\n\n{CLAIM_FENCE_OPEN}\n{claims_json}\n```\n"


def _run(project_root: Path) -> int:
    """Run the rule against a fake tree bound to the fixture package.

    Args:
        project_root: Fake project root.

    Returns:
        Violation count.
    """
    return run_physics_claim_rules(project_root, package_name=FIXTURE_PACKAGE)
