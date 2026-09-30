from __future__ import annotations

import hashlib
from pathlib import Path


class FileStore:
    """Local content-addressed storage for the prototype.

    A production deployment can replace this class with an S3-compatible adapter
    without changing the database or API layer.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> dict[str, str | int]:
        digest = hashlib.sha256(content).hexdigest()
        extension = Path(filename).suffix.lower()
        destination = self.root / f"{digest}{extension}"
        destination.write_bytes(content)
        return {
            "source_uri": str(destination),
            "sha256": digest,
            "size_bytes": len(content),
        }
