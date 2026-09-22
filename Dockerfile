# fly-trader: the trading engine and its console, on the CPU (a Linux container has no Metal GPU; the fly infers fine on CPU).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . .
# CPU-only torch first (the default Linux wheel pulls CUDA, ~3 GB); the project's own pin is then already satisfied.
RUN uv venv /app/.venv \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.12" \
    && uv pip install . \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/app/docker/entrypoint.sh"]
