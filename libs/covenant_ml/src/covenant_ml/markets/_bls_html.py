"""Native BLS summary elements collected without interpreting unrelated tables."""

from __future__ import annotations

from html.parser import HTMLParser


class ReleaseHTML(HTMLParser):
    """Collect release text and explicitly identified Table A elements."""

    def __init__(self) -> None:
        """Initialize one fresh native document parser."""
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None], str]] = []
        self.table_count = 0
        self.in_table = False
        self.active_tag = ""
        self.active_attrs: dict[str, str | None] = {}
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect relevant elements and reject ambiguous nested structure.

        Args:
            tag: Lowercase tag supplied by HTMLParser.
            attrs: Native attribute sequence, preserving duplicates.

        Raises:
            ValueError: CPI-003 for duplicate attributes or nested target elements.
        """
        fields = dict(attrs)
        if tag == "table":
            if self.in_table:
                raise ValueError("[CPI-003] Table A must not contain a nested table")
            if fields.get("id") == "cpi_pressa":
                if len(fields) != len(attrs):
                    raise ValueError("[CPI-003] duplicated Table A attributes")
                self.table_count += 1
                self.in_table = True
        relevant = tag == "pre" or (self.in_table and tag in ("caption", "th", "td"))
        if relevant:
            if self.active_tag or len(fields) != len(attrs):
                raise ValueError("[CPI-003] nested elements or duplicated relevant attributes")
            self.active_tag, self.active_attrs, self.parts = tag, fields, []
        elif tag == "br" and self.active_tag:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        """Preserve visible text only while inside a relevant element.

        Args:
            data: Entity-decoded native text.
        """
        if self.active_tag:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Finish normalized target text and enforce table closure.

        Args:
            tag: Lowercase closing tag.

        Raises:
            ValueError: CPI-003 for a table closed inside an unfinished cell.
        """
        if tag == self.active_tag:
            self.elements.append((tag, self.active_attrs, " ".join("".join(self.parts).split())))
            self.active_tag, self.active_attrs, self.parts = "", {}, []
        if tag == "table" and self.in_table:
            if self.active_tag:
                raise ValueError("[CPI-003] Table A closed before its active element")
            self.in_table = False

    def require_complete(self) -> None:
        """Require exactly one complete Table A after native parsing.

        Raises:
            ValueError: CPI-003 for incomplete or ambiguous target structure.
        """
        if self.active_tag or self.in_table or self.table_count != 1:
            raise ValueError("[CPI-003] require one complete CPI Table A and closed release text")
