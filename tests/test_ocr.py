import pytest

from app.ocr import OcrUnavailable, extract_pdf_ocr


def test_ocr_reports_when_tesseract_is_not_installed(monkeypatch):
    monkeypatch.setattr("app.ocr.shutil.which", lambda command: None)

    with pytest.raises(OcrUnavailable, match="Tesseract"):
        extract_pdf_ocr(b"not-read-until-tesseract-is-available")
