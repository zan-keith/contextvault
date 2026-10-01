from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("CONTEXTVAULT_DB", "/tmp/contextvault-openapi-global.db")


ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "openapi.json"


def render_schema() -> str:
    from app.main import create_app

    with tempfile.TemporaryDirectory(prefix="contextvault-openapi-") as directory:
        app = create_app(Path(directory) / "openapi.db")
        return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write ContextVault's OpenAPI schema")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render_schema()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("OpenAPI schema is stale; run `make openapi`.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
