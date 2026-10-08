FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    CONTEXTVAULT_DB=/data/contextvault.db \
    CONTEXTVAULT_STORAGE_DIR=/data/files

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY app ./app
COPY benchmarks ./benchmarks

RUN addgroup --system contextvault \
    && adduser --system --ingroup contextvault --no-create-home contextvault \
    && mkdir -p /data/files \
    && chown -R contextvault:contextvault /app /data

USER contextvault

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
