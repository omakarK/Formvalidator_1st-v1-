"""
FastAPI Application — Main Entry Point.

Production-grade form validation API with:
  - Async endpoints backed by ThreadPoolExecutor
  - Template preloading at startup
  - Graceful shutdown
  - Health checks
  - CORS support
  - Request size limiting
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.models.schemas import HealthResponse, ValidationResponse
from app.utils.template_cache import preload_templates
from app.workers.executor import (
    get_executor,
    shutdown_executor,
    validate_form_async,
)

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload resources, then clean up on exit."""
    setup_logging()
    logger = get_logger("app.main")

    settings = get_settings()
    logger.info(
        "starting_form_validator",
        max_workers=settings.max_workers,
        templates_dir=str(settings.templates_dir),
    )

    # Preload templates into cache (avoids cold-start latency)
    try:
        preload_templates()
    except Exception as exc:
        logger.error("template_preload_failed", error=str(exc))

    # Eagerly create the thread pool
    get_executor()

    yield

    # Shutdown
    shutdown_executor()
    logger.info("form_validator_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Form Validator",
    description=(
        "Production-grade AI-powered form validation system. "
        "Validates uploaded application forms against predefined templates."
    ),
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger("app.main")


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Processing-Time-Ms"] = str(elapsed)
    return response


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        templates_loaded=True,
        worker_pool_size=settings.max_workers,
    )


@app.post(
    "/validate-form",
    response_model=ValidationResponse,
    tags=["Validation"],
    summary="Validate an uploaded application form",
    description=(
        "Upload a PDF or image file. The system validates it against "
        "predefined templates, checks for passport photo, applicant "
        "signature, page completeness, and structural accuracy."
    ),
)
async def validate_form(file: UploadFile = File(...)):
    """
    Validate an uploaded form document.

    Accepts PDF or image (JPEG, PNG) uploads.
    Returns comprehensive validation results with confidence scores.
    """
    settings = get_settings()

    # --- Input validation ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {"pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {allowed_extensions}",
        )

    # --- Read file bytes (async I/O on event loop — fast) ---
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.max_upload_size_mb}MB",
        )

    logger.info(
        "form_upload_received",
        filename=file.filename,
        size_kb=round(len(file_bytes) / 1024, 1),
    )

    # --- Dispatch to thread pool (event loop stays free) ---
    result = await validate_form_async(file_bytes, file.filename)
    return result


@app.post(
    "/validate-form/debug",
    response_model=ValidationResponse,
    tags=["Validation"],
    summary="Validate with debug visualization data",
)
async def validate_form_debug(file: UploadFile = File(...)):
    """
    Same as /validate-form but includes debug information
    such as region coordinates, intermediate scores, and
    processing breakdown.
    """
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = {"pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '.{ext}'")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    result = await validate_form_async(file_bytes, file.filename)

    # Inject debug metadata
    result.debug = {
        "photo_region_page1": settings.photo_region_page1,
        "signature_region_page1": settings.signature_region_page1,
        "signature_region_page2": settings.signature_region_page2,
        "template_match_threshold": settings.template_match_threshold,
        "max_workers": settings.max_workers,
    }
    return result
