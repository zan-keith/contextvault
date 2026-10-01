# OCR support

ContextVault first extracts a PDF text layer with `pypdf`. If a PDF has no text layer, it falls back to local OCR through Tesseract.

## Install the host prerequisite

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

macOS with Homebrew:

```bash
brew install tesseract
```

Then restart ContextVault and upload the scanned PDF again.

## Behaviour

- Text-layer PDF: extracted immediately without OCR.
- Scanned PDF with Tesseract available: pages render locally through `pypdfium2`; `pytesseract` extracts text locally.
- Scanned PDF without Tesseract: the API returns HTTP 422 and tells the caller to install `tesseract-ocr`.
- OCR returns no readable text: the API returns HTTP 422 rather than indexing an empty document.

The application does not send scanned documents to an external OCR service.

## Limitations

The synchronous OCR path is appropriate for a local MVP. Large PDFs should move to a background worker before a shared deployment, with page limits, resource controls, and an explicit ingestion status.
