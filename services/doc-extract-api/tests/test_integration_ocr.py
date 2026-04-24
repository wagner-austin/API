"""Integration tests for docTR GPU OCR pipeline.

These tests run real docTR OCR on the RTX 3090 Ti GPU.
They verify that the full extraction pipeline (pdfplumber + docTR)
works end-to-end on real PDFs.
"""

from __future__ import annotations

from typing import Protocol

from doc_extract_api import _test_hooks
from doc_extract_api import ocr as ocr_mod
from doc_extract_api.extract import extract_pdf_pages
from doc_extract_api.ocr import (
    _DoctrBlock,
    _DoctrDocument,
    _DoctrLine,
    _DoctrOcrPredictor,
    _DoctrPage,
    _DoctrWord,
    _get_ocr_model,
    _get_pdfium_document,
    _NdArrayLike,
    _ocr_lock,
    _render_pdf_page_image,
    configure_ocr_hook,
    ocr_pdf,
)


class _FpdfProto(Protocol):
    """Protocol for fpdf2 FPDF class."""

    def add_page(self) -> None: ...
    def set_font(self, family: str, size: int) -> None: ...
    def cell(self, text: str) -> None: ...
    def output(self) -> bytes: ...


def _create_test_pdf(text: str = "Hello World OCR Test") -> bytes:
    """Create a PDF with known text for OCR testing.

    Args:
        text: Text to render in the PDF.

    Returns:
        Raw PDF bytes.
    """
    _fpdf2 = __import__("fpdf", fromlist=["FPDF"])
    fpdf: _FpdfProto = _fpdf2.FPDF()
    fpdf.add_page()
    fpdf.set_font("Helvetica", size=24)
    fpdf.cell(text=text)
    return fpdf.output()


class TestPdfiumRendering:
    """Tests for pypdfium2 page rendering."""

    def test_render_pdf_page(self) -> None:
        """Render a PDF page to an image array via pypdfium2."""
        pdf_bytes = _create_test_pdf()
        pdf, _pdfium_error = _get_pdfium_document(pdf_bytes)
        assert len(pdf) == 1

        page = pdf[0]
        rendered = _render_pdf_page_image(page)
        copy = rendered.copy()
        assert copy is not None and copy is not rendered
        page.close()
        pdf.close()


class TestDoctrOcr:
    """Tests for docTR GPU OCR on real PDFs."""

    def test_ocr_model_loads_on_gpu(self) -> None:
        """The docTR OCR model loads successfully on CUDA GPU."""
        with _ocr_lock:
            model = _get_ocr_model()
            assert callable(model)

    def test_ocr_single_page(self) -> None:
        """OCR a single page PDF and verify text extraction."""
        pdf_bytes = _create_test_pdf("DocTR GPU Integration Test")
        result = ocr_pdf(pdf_bytes, [0])
        assert 0 in result
        extracted = result[0]
        assert len(extracted) >= 5  # at least some recognized text

    def test_ocr_empty_pages_list(self) -> None:
        """OCR with empty pages list returns empty dict."""
        pdf_bytes = _create_test_pdf()
        result = ocr_pdf(pdf_bytes, [])
        assert result == {}

    def test_ocr_multi_page(self) -> None:
        """OCR multiple pages from a multi-page PDF."""
        _fpdf2 = __import__("fpdf", fromlist=["FPDF"])
        fpdf: _FpdfProto = _fpdf2.FPDF()
        for i in range(3):
            fpdf.add_page()
            fpdf.set_font("Helvetica", size=20)
            fpdf.cell(text=f"Page {i + 1} content for OCR")
        pdf_bytes: bytes = fpdf.output()

        result = ocr_pdf(pdf_bytes, [0, 1, 2])
        assert len(result) == 3
        for i in range(3):
            assert i in result
            assert len(result[i]) >= 3


class TestDualExtraction:
    """Tests for the full pdfplumber + docTR dual extraction pipeline."""

    def test_extract_pdf_pages_with_ocr(self) -> None:
        """Full extraction with both pdfplumber and docTR on a real PDF."""
        configure_ocr_hook()
        assert _test_hooks.ocr_pdf is ocr_pdf

        pdf_bytes = _create_test_pdf("Dual Extraction Pipeline Test")
        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 1
        assert len(pages[0]["content"]) >= 5
        assert pages[0]["method"] in ("pdfplumber-text", "pdfplumber-table", "doctr-ocr")

    def test_extract_multi_page_with_ocr(self) -> None:
        """Full dual extraction on a multi-page PDF."""
        configure_ocr_hook()

        _fpdf2 = __import__("fpdf", fromlist=["FPDF"])
        fpdf: _FpdfProto = _fpdf2.FPDF()
        for i in range(3):
            fpdf.add_page()
            fpdf.set_font("Helvetica", size=16)
            fpdf.cell(text=f"Multi page dual extraction test page {i + 1}")
        pdf_bytes: bytes = fpdf.output()

        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 3
        for page in pages:
            assert len(page["content"]) >= 5


# ---------------------------------------------------------------------------
# Fake pdfium for error-path tests (same pattern as irvine-scraper)
# ---------------------------------------------------------------------------

_FAKE_CLOSE_LOG: list[str] = []
_FAKE_GET_FAILURES: set[int] = {1}
_FAKE_RENDER_FAILURES: set[int] = set()


class _FakePdfiumError(Exception):
    """Concrete fake pdfium exception."""


class _FakeArray:
    """Concrete array-like with copy."""

    def __init__(self, value: int) -> None:
        self._value = value

    def copy(self) -> _FakeArray:
        """Return a detached copy."""
        return _FakeArray(self._value)


class _FakeBitmap:
    """Concrete fake bitmap that records closure."""

    def __init__(self, value: int) -> None:
        self._value = value

    def to_numpy(self) -> _FakeArray:
        """Return a NumPy-like array wrapper."""
        return _FakeArray(self._value)

    def close(self) -> None:
        """Record bitmap closure."""
        _FAKE_CLOSE_LOG.append(f"bitmap:{self._value}")


class _FakeRenderedPage:
    """Concrete fake page that records closure and rendering."""

    def __init__(self, value: int) -> None:
        self._value = value

    def render(self) -> _FakeBitmap:
        """Return a fake bitmap or raise for render failures."""
        if self._value in _FAKE_RENDER_FAILURES:
            raise _FakePdfiumError(f"failed to render page {self._value}")
        return _FakeBitmap(self._value)

    def close(self) -> None:
        """Record page closure."""
        _FAKE_CLOSE_LOG.append(f"page:{self._value}")


class _FakePdfDocument:
    """Concrete fake pdfium document."""

    def __init__(self, _pdf_bytes: bytes) -> None:
        pass

    def __len__(self) -> int:
        """Return fake page count."""
        return 3

    def __getitem__(self, index: int) -> _FakeRenderedPage:
        """Return a page or raise for unreadable index."""
        if index in _FAKE_GET_FAILURES:
            raise _FakePdfiumError("failed to load page")
        return _FakeRenderedPage(index)

    def close(self) -> None:
        """Record document closure."""
        _FAKE_CLOSE_LOG.append("document")


class _FakeOcrWord:
    """Concrete fake OCR word."""

    def __init__(self, value: str) -> None:
        self.value = value


class _FakeOcrLine:
    """Concrete fake OCR line."""

    def __init__(self, text: str) -> None:
        self.words: list[_DoctrWord] = [_FakeOcrWord(text)]


class _FakeOcrBlock:
    """Concrete fake OCR block."""

    def __init__(self, text: str) -> None:
        self.lines: list[_DoctrLine] = [_FakeOcrLine(text)]


class _FakeOcrPage:
    """Concrete fake OCR page."""

    def __init__(self, text: str) -> None:
        self.blocks: list[_DoctrBlock] = [_FakeOcrBlock(text)]


class _FakeOcrDocument:
    """Concrete fake OCR document."""

    def __init__(self, texts: list[str]) -> None:
        self.pages: list[_DoctrPage] = [_FakeOcrPage(t) for t in texts]


class _FakeModel:
    """Concrete fake OCR model."""

    def __call__(self, doc: list[_NdArrayLike]) -> _DoctrDocument:
        """Convert rendered page payloads into OCR text pages."""
        texts = [f"PAGE-{i}" for i in range(len(doc))]
        return _FakeOcrDocument(texts)

    def cuda(self) -> _DoctrOcrPredictor:
        """Return self to satisfy the production OCR protocol."""
        return self


def _fake_get_ocr_model() -> _DoctrOcrPredictor:
    """Return the fake OCR model."""
    return _FakeModel()


def _fake_get_pdfium_document(
    pdf_bytes: bytes,
) -> tuple[_FakePdfDocument, type[Exception]]:
    """Return a fake pdfium document and error type."""
    return _FakePdfDocument(pdf_bytes), _FakePdfiumError


class TestOcrWithFakePdfium:
    """Tests for OCR error paths using swapped pdfium/model functions."""

    def _run_with_fakes(self, pages: list[int]) -> dict[int, str]:
        """Run ocr_pdf with fake pdfium and fake model.

        Args:
            pages: Page indices to OCR.

        Returns:
            OCR result dict.
        """
        original_get_pdfium = ocr_mod._get_pdfium_document
        original_get_ocr = ocr_mod._get_ocr_model
        ocr_mod._get_pdfium_document = _fake_get_pdfium_document
        ocr_mod._get_ocr_model = _fake_get_ocr_model
        result = ocr_pdf(b"%PDF-fake", pages)
        ocr_mod._get_pdfium_document = original_get_pdfium
        ocr_mod._get_ocr_model = original_get_ocr
        return result

    def test_skips_unreadable_pages(self) -> None:
        """Unreadable pages are skipped, readable pages are OCR'd."""
        _FAKE_CLOSE_LOG.clear()
        _FAKE_GET_FAILURES.clear()
        _FAKE_GET_FAILURES.add(1)
        _FAKE_RENDER_FAILURES.clear()

        result = self._run_with_fakes([0, 1, 2])
        assert result == {0: "PAGE-0", 2: "PAGE-1"}
        assert "document" in _FAKE_CLOSE_LOG

    def test_render_failure_skips_page(self) -> None:
        """Render failures skip the page without aborting."""
        _FAKE_CLOSE_LOG.clear()
        _FAKE_GET_FAILURES.clear()
        _FAKE_GET_FAILURES.add(1)
        _FAKE_RENDER_FAILURES.clear()
        _FAKE_RENDER_FAILURES.add(0)

        result = self._run_with_fakes([0, 2])
        assert 0 not in result
        assert 2 in result

    def test_all_pages_unreadable_returns_empty(self) -> None:
        """When all requested pages fail, returns empty dict."""
        _FAKE_CLOSE_LOG.clear()
        _FAKE_GET_FAILURES.clear()
        _FAKE_GET_FAILURES.add(1)
        _FAKE_GET_FAILURES.add(2)
        _FAKE_RENDER_FAILURES.clear()
        _FAKE_RENDER_FAILURES.add(0)

        result = self._run_with_fakes([0, 1, 2])
        assert result == {}


class TestOcrErrorHandling:
    """Tests for OCR error handling with invalid pages."""

    def test_ocr_out_of_range_page(self) -> None:
        """OCR with out-of-range page index logs warning and skips."""
        pdf_bytes = _create_test_pdf("Single page PDF")
        # Request page index 99 which doesn't exist in a 1-page PDF
        result = ocr_pdf(pdf_bytes, [99])
        assert result == {}

    def test_ocr_mixed_valid_and_invalid_pages(self) -> None:
        """OCR with mix of valid and invalid page indices."""
        pdf_bytes = _create_test_pdf("Mixed pages test")
        # Page 0 exists, page 50 doesn't
        result = ocr_pdf(pdf_bytes, [0, 50])
        assert 0 in result
        assert 50 not in result
        assert len(result[0]) >= 3


class TestConfigureOcrHook:
    """Tests for OCR hook configuration."""

    def test_configure_sets_hook(self) -> None:
        """configure_ocr_hook sets the OCR hook to the real implementation."""
        _test_hooks.ocr_pdf = None
        configure_ocr_hook()
        assert _test_hooks.ocr_pdf is ocr_pdf

    def test_configured_hook_is_callable(self) -> None:
        """The configured OCR hook is callable."""
        configure_ocr_hook()
        assert callable(_test_hooks.ocr_pdf)
