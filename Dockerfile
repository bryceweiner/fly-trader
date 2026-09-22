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
# Which torch: TORCH_INDEX=cpu (default: any machine, ~200 MB) or cu126 / cu130 for an NVIDIA GPU (the CUDA runtime is
# inside the wheel; the host needs only the driver and the NVIDIA Container Toolkit -- see docker-compose.cuda.yml).
# Installed first so the project's own pin is already satisfied and PyPI's 3 GB CUDA default is never pulled by accident.
# Editable install: the package IS /app/fly_trader, so config.REPO_ROOT is /app -- the mounted .env the wallet writes
# to, the console's static directory, config/. A plain install ran a copy under site-packages and looked there instead.
ARG TORCH_INDEX=cpu
RUN uv venv /app/.venv \
    && uv pip install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" "torch>=2.12" \
    && uv pip install -e . \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/app/docker/entrypoint.sh"]
