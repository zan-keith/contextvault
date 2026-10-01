from __future__ import annotations

import shutil
from io import BytesIO


class OcrUnavailable(RuntimeError):
    """OCR cannot run because a required local dependency is unavailable."""


def extract_pdf_ocr(content: bytes, *, dpi_scale: float = 2.0) -> str:
    """Render a scanned PDF locally and extract text through Tesseract.

    Tesseract is deliberately a host prerequisite instead of a hidden cloud API:
    technical documents stay local and contributors can reproduce the workflow.
    """

    if shutil.which("tesseract") is None:
        raise OcrUnavailable(
            "PDF has no extractable text and OCR requires Tesseract. "
            "Install tesseract-ocr, then retry the upload."
        )

    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as exc:
        raise OcrUnavailable("OCR Python dependencies are unavailable; reinstall ContextVault.") from exc

    try:
        document = pdfium.PdfDocument(BytesIO(content))
        pages = []
        for index in range(len(document)):
            image = document[index].render(scale=dpi_scale).to_pil()
            text = pytesseract.image_to_string(image).strip()
            if text:
                pages.append(text)
    except Exception as exc:
        raise OcrUnavailable("OCR could not process the PDF.") from exc

    if not pages:
        raise OcrUnavailable("OCR completed but no readable text was found in the PDF.")
    return "\n\n".join(pages)
