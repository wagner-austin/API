"""The fastText language-id model catalogue.

Two packages download these files and neither one owns the answer to "which
file, from where". The tests below are mostly about the two selections staying
DISTINCT in all three of their fields -- that is what a bad edit to the
catalogue would collapse, and it would collapse silently.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.langid_models import (
    LID_176_FILENAME,
    LID_176_URL,
    LID_218E_FILENAME,
    LID_218E_URL,
    MODEL_DIRNAME,
    langid_model_file,
)


class TestTheSelection:
    def test_preferring_218e_names_the_nllb_model(self) -> None:
        wanted = langid_model_file("/data", prefer_218e=True)

        assert wanted == {
            "path": Path("/data") / MODEL_DIRNAME / LID_218E_FILENAME,
            "url": LID_218E_URL,
        }

    def test_not_preferring_218e_names_the_lid176_model(self) -> None:
        wanted = langid_model_file("/data", prefer_218e=False)

        assert wanted == {
            "path": Path("/data") / MODEL_DIRNAME / LID_176_FILENAME,
            "url": LID_176_URL,
        }

    def test_the_two_selections_agree_on_nothing(self) -> None:
        """A catalogue edit that made one selection shadow the other would send
        every caller to the same model while both still asked for their own."""
        preferred = langid_model_file("/data", prefer_218e=True)
        fallback = langid_model_file("/data", prefer_218e=False)

        assert preferred["path"] != fallback["path"]
        assert preferred["url"] != fallback["url"]

    def test_both_files_sit_under_the_same_models_subdirectory(self) -> None:
        parents = [langid_model_file("/data", prefer_218e=p)["path"].parent for p in (True, False)]

        assert parents == [Path("/data") / MODEL_DIRNAME] * 2

    def test_the_data_directory_is_taken_verbatim(self) -> None:
        wanted = langid_model_file("relative/dir", prefer_218e=True)

        assert wanted["path"] == Path("relative/dir") / MODEL_DIRNAME / LID_218E_FILENAME


class TestTheDeclaredCatalogue:
    def test_each_url_ends_with_the_file_name_it_delivers(self) -> None:
        """A caller writes the download to ``path`` and reads the bytes back
        from it, so a URL pointing at a differently-named file would leave a
        model on disk under a name that says it is the other one."""
        assert LID_218E_URL.endswith(f"/{LID_218E_FILENAME}")
        assert LID_176_URL.endswith(f"/{LID_176_FILENAME}")

    def test_both_models_are_served_over_https(self) -> None:
        assert [LID_218E_URL.startswith("https://"), LID_176_URL.startswith("https://")] == [
            True,
            True,
        ]


__all__ = ["TestTheDeclaredCatalogue", "TestTheSelection"]
