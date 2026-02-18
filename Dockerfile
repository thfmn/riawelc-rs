# Multi-stage Dockerfile for RIAWELC API
# Build: docker build -t riawelc-api .
# Run:   docker run -p 8000:8000 -v ./outputs:/app/outputs riawelc-api

# --- Builder stage ---
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
RUN uv pip install --system ".[api]"

# --- Runtime stage ---
FROM python:3.11-slim AS runtime

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY configs/ configs/

RUN mkdir -p outputs && chown -R appuser:appuser /app

USER appuser

ENV RIAWELC_HOST=0.0.0.0
ENV RIAWELC_PORT=8000
ENV RIAWELC_LOG_LEVEL=info

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "riawelc.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
