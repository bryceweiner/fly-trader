# fly-trader: the trading engine and its console, on the CPU (a Linux container has no Metal GPU; the fly infers fine on CPU).
# Two images from one file:
#   default (last stage, "brain"): the engine, console and vault worker. It holds NO wallet key.
#   --target signer: the only holder of the hot keys (fly_trader/signer): standard library + solders + httpx, nothing
#   else of fly_trader, so none of the brain's dependencies ever run next to a key.
# Base images are pinned by digest and every Python dependency by hash (uv.lock, docker/signer-requirements.txt): a
# release builds exactly what was reviewed, whatever the registries serve later.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b

# ---------------------------------------------------------------- the signer
FROM ${PYTHON_IMAGE} AS signer
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends age && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10000 fly && useradd -u 10002 -g 10000 -M -s /usr/sbin/nologin signer
COPY docker/signer-requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/req.txt && rm /tmp/req.txt
WORKDIR /app
COPY fly_trader/__init__.py fly_trader/__init__.py
COPY fly_trader/signer fly_trader/signer
USER 10002:10000
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD python -m fly_trader.signer ping || exit 1
CMD ["python", "-m", "fly_trader.signer", "serve"]

# ---------------------------------------------------------------- the brain (default)
FROM ${PYTHON_IMAGE} AS brain
ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    HOME=/tmp \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client ca-certificates curl age \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10000 fly && useradd -u 10001 -g 10000 -M -s /usr/sbin/nologin fly
COPY --from=ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv

WORKDIR /app
COPY . .
# Which torch: TORCH_INDEX=cpu (default: any machine, ~200 MB) or cu126 / cu130 for an NVIDIA GPU (the CUDA runtime is
# inside the wheel; the host needs only the driver and the NVIDIA Container Toolkit -- see docker-compose.cuda.yml).
# It names the project extra that pins torch to that index. --frozen: exactly uv.lock with its hashes, never a fresh
# resolution. The project installs editable: the package IS /app/fly_trader, so config.REPO_ROOT is /app.
ARG TORCH_INDEX=cpu
RUN uv sync --frozen --no-dev --extra "${TORCH_INDEX}" \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/fly_trader/ui/static/brain && chown -R 10001:10000 /app/fly_trader/ui/static/brain
USER 10001:10000

EXPOSE 8501
# healthy = the database answers, the console's worker threads run and (on the vault server) the signer answers
HEALTHCHECK --interval=30s --timeout=20s --start-period=300s --retries=5 CMD fly-trader health || exit 1
ENTRYPOINT ["/app/docker/entrypoint.sh"]
