"""GPU-accelerated OCR extraction using docTR and pypdfium2.

Renders PDF pages to images via pypdfium2, then runs docTR
OCR on a CUDA GPU. Results are returned as per-page text
keyed by 0-based page index.

The OCR model is loaded once and cached for reuse. All
operations are serialized behind a lock because pypdfium2
is not thread-safe and the GPU model cannot process
multiple documents concurrently.
"""

from __future__ import annotations

import threading
from typing import Protocol

from platform_core.logging import get_logger

logger = get_logger(__name__)

_OCR_BATCH_SIZE: int = 50
_ocr_lock: threading.Lock = threading.Lock()
_ocr_model_cache: list[_DoctrOcrPredictor] = []


# ---------------------------------------------------------------------------
# docTR Protocols
# ---------------------------------------------------------------------------


class _DoctrWord(Protocol):
    """Protocol for a single recognized word from docTR."""

    @property
    def value(self) -> str:
        """The recognized text of this word."""
        ...


class _DoctrLine(Protocol):
    """Protocol for a line of recognized text from docTR."""

    @property
    def words(self) -> list[_DoctrWord]:
        """The words in this line."""
        ...


class _DoctrBlock(Protocol):
    """Protocol for a text block from docTR."""

    @property
    def lines(self) -> list[_DoctrLine]:
        """The lines in this block."""
        ...


class _DoctrPage(Protocol):
    """Protocol for a page result from docTR."""

    @property
    def blocks(self) -> list[_DoctrBlock]:
        """The text blocks on this page."""
        ...


class _DoctrDocument(Protocol):
    """Protocol for a full document result from docTR."""

    @property
    def pages(self) -> list[_DoctrPage]:
        """The pages in this document."""
        ...


class _DoctrOcrPredictor(Protocol):
    """Protocol for the docTR OCR predictor model."""

    def __call__(self, doc: list[_NdArrayLike]) -> _DoctrDocument:
        """Run OCR on a list of page images.

        Args:
            doc: List of numpy arrays (page images).

        Returns:
            A document result with pages, blocks, lines, and words.
        """
        ...

    def cuda(self) -> _DoctrOcrPredictor:
        """Move the model to GPU.

        Returns:
            The model on CUDA device.
        """
        ...


class _DoctrOcrPredictorFn(Protocol):
    """Protocol for doctr.models.ocr_predictor factory."""

    def __call__(
        self,
        *,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
    ) -> _DoctrOcrPredictor:
        """Create an OCR predictor model.

        Args:
            det_arch: Detection model architecture.
            reco_arch: Recognition model architecture.
            pretrained: Whether to use pretrained weights.

        Returns:
            An OCR predictor instance.
        """
        ...


# ---------------------------------------------------------------------------
# pypdfium2 Protocols
# ---------------------------------------------------------------------------


class _NdArrayLike(Protocol):
    """Protocol for a NumPy-like array with copy support."""

    def copy(self) -> _NdArrayLike:
        """Return an owned copy of the array data.

        Returns:
            A standalone copy that no longer shares memory.
        """
        ...


class _PdfBitmap(Protocol):
    """Protocol for a rendered pdfium bitmap."""

    def to_numpy(self) -> _NdArrayLike:
        """Convert the bitmap into a NumPy-like array view.

        Returns:
            An array view backed by the bitmap buffer.
        """
        ...

    def close(self) -> None:
        """Release the bitmap resources."""
        ...


class _PdfPage(Protocol):
    """Protocol for a pdfium page."""

    def render(self) -> _PdfBitmap:
        """Render the page into a bitmap.

        Returns:
            The rendered bitmap for this page.
        """
        ...

    def close(self) -> None:
        """Release the page resources."""
        ...


class _PdfDocument(Protocol):
    """Protocol for a pdfium document."""

    def __len__(self) -> int:
        """Return the number of pages in the document."""
        ...

    def __getitem__(self, index: int) -> _PdfPage:
        """Return a page for the requested index.

        Args:
            index: Zero-based page index.

        Returns:
            The corresponding page.
        """
        ...

    def close(self) -> None:
        """Release the document resources."""
        ...


class _PdfDocumentClass(Protocol):
    """Protocol for the pdfium document constructor."""

    def __call__(self, src: bytes) -> _PdfDocument:
        """Open a PDF document from raw bytes.

        Args:
            src: Raw PDF file bytes.

        Returns:
            An opened pdfium document.
        """
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_pdfium_document(pdf_bytes: bytes) -> tuple[_PdfDocument, type[Exception]]:
    """Open a pdfium document and return its concrete error type.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        Tuple of the opened pdfium document and the pdfium exception class.
    """
    _pdfium = __import__("pypdfium2")
    pdf_document: _PdfDocumentClass = _pdfium.PdfDocument
    _pdfium_misc = __import__("pypdfium2._helpers.misc", fromlist=["PdfiumError"])
    pdfium_error: type[Exception] = _pdfium_misc.PdfiumError
    return pdf_document(bytes(pdf_bytes)), pdfium_error


def _render_pdf_page_image(page: _PdfPage) -> _NdArrayLike:
    """Render one pdfium page into owned image data.

    pypdfium2 exposes NumPy arrays that share memory with the bitmap
    buffer. This function copies the array immediately so the bitmap
    can be closed before OCR batching begins.

    Args:
        page: The pdfium page to render.

    Returns:
        Standalone image data for the rendered page.
    """
    bitmap = page.render()
    copied: _NdArrayLike = bitmap.to_numpy().copy()
    bitmap.close()
    return copied


def _get_ocr_model() -> _DoctrOcrPredictor:
    """Return the shared docTR OCR model, loading it on first call.

    Uses _ocr_model_cache to store the singleton. The model is
    loaded once and reused for all subsequent calls. Must be called
    while holding _ocr_lock.

    Returns:
        The shared OCR predictor instance.
    """
    if len(_ocr_model_cache) == 0:
        _doctr_models = __import__("doctr.models", fromlist=["ocr_predictor"])
        ocr_predictor: _DoctrOcrPredictorFn = _doctr_models.ocr_predictor
        model: _DoctrOcrPredictor = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        ).cuda()
        _ocr_model_cache.append(model)
        logger.info("docTR OCR model loaded (CUDA)")
    return _ocr_model_cache[0]


# ---------------------------------------------------------------------------
# Public OCR function
# ---------------------------------------------------------------------------


def ocr_pdf(pdf_bytes: bytes, pages: list[int]) -> dict[int, str]:
    """Extract text from specific PDF pages using docTR GPU-accelerated OCR.

    Loads requested pages through pdfium, copies each rendered page
    into owned memory, closes pdfium resources immediately, then
    runs OCR in batches and returns per-page text keyed by 0-based
    page index.

    The entire operation is serialized behind _ocr_lock because
    pypdfium2 is not thread-safe and the GPU model cannot process
    multiple documents concurrently.

    Args:
        pdf_bytes: Raw PDF file bytes.
        pages: 0-based page indices to OCR.

    Returns:
        Dict mapping 0-based page index to OCR text for that page.
    """
    with _ocr_lock:
        pdf, pdfium_error = _get_pdfium_document(pdf_bytes)
        selected_pairs: list[tuple[int, _NdArrayLike]] = []
        failed_pages: list[int] = []
        total_pages: int = len(pdf)

        for page_index in pages:
            page: _PdfPage | None = None
            try:
                page = pdf[page_index]
            except pdfium_error:
                logger.warning("Failed to load PDF page %d", page_index)
                failed_pages.append(page_index)
                continue

            try:
                rendered = _render_pdf_page_image(page)
                selected_pairs.append((page_index, rendered))
            except pdfium_error:
                logger.warning("Failed to render PDF page %d", page_index)
                failed_pages.append(page_index)
            finally:
                page.close()

        pdf.close()

        selected: list[_NdArrayLike] = [rendered for _, rendered in selected_pairs]
        logger.info(
            "Running docTR OCR on %d/%d pages (%d bytes)",
            len(selected),
            total_pages,
            len(pdf_bytes),
        )
        if len(failed_pages) > 0:
            logger.warning(
                "Skipping %d unreadable PDF pages: %s",
                len(failed_pages),
                failed_pages,
            )
        if len(selected) == 0:
            logger.info("docTR OCR complete: 0 chars from %d pages", len(pages))
            return {}

        model: _DoctrOcrPredictor = _get_ocr_model()

        result_pages: list[_DoctrPage] = []
        for batch_start in range(0, len(selected), _OCR_BATCH_SIZE):
            batch: list[_NdArrayLike] = selected[batch_start : batch_start + _OCR_BATCH_SIZE]
            batch_result: _DoctrDocument = model(batch)
            result_pages.extend(batch_result.pages)

    per_page: dict[int, str] = {}
    total_chars: int = 0
    selected_indices: list[int] = [idx for idx, _ in selected_pairs]
    for idx, result_page in zip(selected_indices, result_pages, strict=True):
        lines: list[str] = []
        for block in result_page.blocks:
            for line in block.lines:
                line_text: str = " ".join(w.value for w in line.words)
                lines.append(line_text)
        text: str = "\n".join(lines)
        per_page[idx] = text
        total_chars += len(text)

    logger.info("docTR OCR complete: %d chars from %d pages", total_chars, len(pages))
    return per_page


def configure_ocr_hook() -> None:
    """Set the OCR hook to the real GPU implementation."""
    from . import _test_hooks

    _test_hooks.ocr_pdf = ocr_pdf


__all__ = [
    "_OCR_BATCH_SIZE",
    "_get_ocr_model",
    "_get_pdfium_document",
    "_ocr_lock",
    "_ocr_model_cache",
    "_render_pdf_page_image",
    "configure_ocr_hook",
    "ocr_pdf",
]
