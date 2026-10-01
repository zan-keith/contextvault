from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.ocr import OcrUnavailable, extract_pdf_ocr


class UnsupportedDocument(ValueError):
    """Raised when a document has no usable text representation."""


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise UnsupportedDocument("PDF could not be parsed") from exc
        if not text.strip():
            try:
                return extract_pdf_ocr(content)
            except OcrUnavailable as exc:
                raise UnsupportedDocument(str(exc)) from exc
        return text.strip()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UnsupportedDocument("Uploaded text must be UTF-8 encoded") from exc
    if not text.strip():
        raise UnsupportedDocument("File is empty")
    return text
