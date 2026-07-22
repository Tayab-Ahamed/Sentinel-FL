# SENTINEL-FL — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder):  install Python deps into /install
#   Stage 2 (runtime):  copy /install + project code, run the API
#
# Usage:
#   docker build -t sentinel-fl .
#   docker run -p 8000:8000 sentinel-fl
#
# For local development, prefer docker-compose (docker-compose.yml).

# ---------------------------------------------------------------------------
# Stage 1: Builder — install deps into a layer we can COPY in
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (needed for some numpy/scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest only (cached unless pyproject.toml changes)
COPY pyproject.toml ./

# Install all runtime deps into /install
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefix=/install ".[dev]"

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only what's needed to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="SENTINEL-FL"
LABEL org.opencontainers.image.description="Federated Backdoor Immune System — IEEE GSC26 Challenge 1"
LABEL org.opencontainers.image.version="0.1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTINEL_LOG_LEVEL=INFO \
    SENTINEL_API_HOST=0.0.0.0 \
    SENTINEL_API_PORT=8000

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY ai/ ./ai/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/

# Create directories that are written to at runtime
RUN mkdir -p experiments/checkpoints

# Expose the API port
EXPOSE 8000

# Default command: start the FastAPI backend
# Override entrypoint for other uses (e.g. python scripts/run_demo.py)
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
