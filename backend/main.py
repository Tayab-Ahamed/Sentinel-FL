"""
backend/main.py — SENTINEL-FL FastAPI application entry point.

Serves pre-computed experiment artefacts (JSON) to the React dashboard.
No live training is triggered during a judged demo — all endpoints are
read-only except POST /experiments/run (local development only).

See API.md for the full endpoint specification.
Run: uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import experiments, remediation, visualizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("SENTINEL_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SENTINEL-FL API",
    description=(
        "Backend API for the SENTINEL-FL federated backdoor immune system. "
        "Serves pre-computed experiment artefacts to the React dashboard. "
        "See API.md for the full endpoint specification."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server (Vite default: 5173)
# ---------------------------------------------------------------------------
_cors_origins_raw = os.environ.get(
    "SENTINEL_API_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
)
_cors_origins = [origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(experiments.router, prefix="/api/v1")
app.include_router(visualizer.router, prefix="/api/v1")
app.include_router(remediation.router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Health check endpoint — returns 200 if the API is running."""
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:
    logger.info("SENTINEL-FL API starting up.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    logger.info("SENTINEL-FL API shutting down.")
