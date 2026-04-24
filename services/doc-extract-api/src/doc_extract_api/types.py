"""Strict types for the doc-extract-api service.

Defines domain types for PDF extraction jobs and results.
Every TypedDict has encode/decode functions with require_* validation.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Extraction method
# ---------------------------------------------------------------------------

ExtractionMethod = Literal["pdfplumber-text", "pdfplumber-table", "doctr-ocr"]

# ---------------------------------------------------------------------------
# Extracted page
# ---------------------------------------------------------------------------


class ExtractedPage(TypedDict):
    """A single extracted page from a PDF document."""

    page_number: int
    content: str
    method: ExtractionMethod


def decode_extracted_page(
    page_number: int,
    content: str,
    method: ExtractionMethod,
) -> ExtractedPage:
    """Create a validated ExtractedPage.

    Args:
        page_number: 1-based page number.
        content: Extracted text content.
        method: Extraction method used.

    Returns:
        A validated ExtractedPage TypedDict.
    """
    return ExtractedPage(
        page_number=page_number,
        content=content,
        method=method,
    )


def encode_extracted_page(page: ExtractedPage) -> str:
    """Format an ExtractedPage as a human-readable string.

    Args:
        page: The extracted page to format.

    Returns:
        Formatted string with page number and method.
    """
    return f"Page {page['page_number']} ({page['method']}): {len(page['content'])} chars"


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

JobStatus = Literal["queued", "processing", "completed", "failed"]


class ExtractionJob(TypedDict):
    """An extraction job tracked in Redis."""

    job_id: str
    status: JobStatus
    title: str
    source: str
    category: str
    file_path: str
    pages_total: int
    pages_done: int
    document_id: str
    error: str


def decode_extraction_job(raw: dict[str, str], job_id: str) -> ExtractionJob:
    """Decode an ExtractionJob from a Redis hash.

    Args:
        raw: Raw string dict from Redis hgetall.
        job_id: The job identifier.

    Returns:
        A validated ExtractionJob TypedDict.

    Raises:
        KeyError: If a required field is missing.
    """
    return ExtractionJob(
        job_id=job_id,
        status=_require_job_status(raw.get("status", "")),
        title=_require_str(raw, "title"),
        source=raw.get("source", ""),
        category=_require_str(raw, "category"),
        file_path=_require_str(raw, "file_path"),
        pages_total=_require_int(raw, "pages_total"),
        pages_done=_require_int(raw, "pages_done"),
        document_id=raw.get("document_id", ""),
        error=raw.get("error", ""),
    )


def encode_extraction_job(job: ExtractionJob) -> dict[str, str]:
    """Encode an ExtractionJob to a Redis hash.

    Args:
        job: The extraction job to encode.

    Returns:
        String dict suitable for Redis hset.
    """
    return {
        "status": job["status"],
        "title": job["title"],
        "source": job["source"],
        "category": job["category"],
        "file_path": job["file_path"],
        "pages_total": str(job["pages_total"]),
        "pages_done": str(job["pages_done"]),
        "document_id": job["document_id"],
        "error": job["error"],
    }


def encode_extraction_job_response(job: ExtractionJob) -> dict[str, str | int]:
    """Encode an ExtractionJob for JSON API response.

    Args:
        job: The extraction job to encode.

    Returns:
        Dict with string and int values for JSON serialization.
    """
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "title": job["title"],
        "source": job["source"],
        "category": job["category"],
        "file_path": job["file_path"],
        "pages_total": job["pages_total"],
        "pages_done": job["pages_done"],
        "document_id": job["document_id"],
        "error": job["error"],
    }


# ---------------------------------------------------------------------------
# Job creation request
# ---------------------------------------------------------------------------


class ExtractionRequest(TypedDict):
    """Request to create a new extraction job."""

    title: str
    file_path: str
    category: str
    source: str


def decode_extraction_request(
    title: str,
    file_path: str,
    category: str,
    source: str,
) -> ExtractionRequest:
    """Create a validated ExtractionRequest.

    Args:
        title: Document title.
        file_path: Absolute path to the PDF file.
        category: Document category.
        source: Source URL or reference.

    Returns:
        A validated ExtractionRequest TypedDict.

    Raises:
        ValueError: If title or file_path is empty.
    """
    if len(title.strip()) == 0:
        raise ValueError("title must not be empty")
    if len(file_path.strip()) == 0:
        raise ValueError("file_path must not be empty")
    return ExtractionRequest(
        title=title,
        file_path=file_path,
        category=category,
        source=source,
    )


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def format_table_rows(tables: list[list[list[str | None]]]) -> str:
    """Format pdfplumber tables as tab-delimited rows.

    Args:
        tables: List of tables from pdfplumber extract_tables().

    Returns:
        Tab-delimited rows joined by newlines.
    """
    lines: list[str] = []
    for table in tables:
        for row in table:
            cells = [(cell.replace("\n", " ") if cell is not None else "") for cell in row]
            lines.append("\t".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_STATUSES: frozenset[str] = frozenset({"queued", "processing", "completed", "failed"})


def _require_job_status(value: str) -> JobStatus:
    """Validate and return a JobStatus literal.

    Args:
        value: Raw status string.

    Returns:
        Validated JobStatus literal.

    Raises:
        ValueError: If value is not a valid status.
    """
    if value not in _VALID_STATUSES:
        raise ValueError(f"Invalid job status: {value!r}")
    # Exhaustive check for mypy
    if value == "queued":
        return "queued"
    if value == "processing":
        return "processing"
    if value == "completed":
        return "completed"
    return "failed"


def _require_str(raw: dict[str, str], key: str) -> str:
    """Extract a required string from a dict.

    Args:
        raw: Source dict.
        key: Key to extract.

    Returns:
        The string value.

    Raises:
        KeyError: If key is missing.
    """
    if key not in raw:
        raise KeyError(f"Missing required field: {key!r}")
    return raw[key]


def _require_int(raw: dict[str, str], key: str) -> int:
    """Extract a required int from a string dict.

    Args:
        raw: Source dict.
        key: Key to extract.

    Returns:
        The parsed integer value.

    Raises:
        KeyError: If key is missing.
        ValueError: If value is not a valid integer.
    """
    if key not in raw:
        raise KeyError(f"Missing required field: {key!r}")
    return int(raw[key])


# ---------------------------------------------------------------------------
# Category validation
# ---------------------------------------------------------------------------

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "general",
        "budget",
        "audit",
        "pension",
        "investment",
        "park",
        "energy",
        "council",
        "legislation",
        "water",
        "education",
        "waste",
        "transportation",
        "housing",
        "environmental",
    }
)


def validate_category(category: str) -> str:
    """Validate a document category.

    Args:
        category: Category string to validate.

    Returns:
        The validated category string.

    Raises:
        ValueError: If category is not in the allowed set.
    """
    if category not in VALID_CATEGORIES:
        valid = ", ".join(sorted(VALID_CATEGORIES))
        raise ValueError(f"Invalid category {category!r}. Valid categories: {valid}")
    return category


__all__ = [
    "VALID_CATEGORIES",
    "ExtractedPage",
    "ExtractionJob",
    "ExtractionMethod",
    "ExtractionRequest",
    "JobStatus",
    "_require_int",
    "_require_job_status",
    "_require_str",
    "decode_extracted_page",
    "decode_extraction_job",
    "decode_extraction_request",
    "encode_extracted_page",
    "encode_extraction_job",
    "encode_extraction_job_response",
    "format_table_rows",
    "validate_category",
]
