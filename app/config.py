"""
Centralized configuration for the AI Form Validator.

All thresholds, paths, and tuning knobs live here so they can be
overridden via environment variables or a .env file without touching code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


# ---------------------------------------------------------------------------
# Resolve project root (parent of /app)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings, populated from env vars / .env."""

    # -- Paths ---------------------------------------------------------------
    project_root: Path = _PROJECT_ROOT
    templates_dir: Path = _PROJECT_ROOT / "templates"

    # -- Template files ------------------------------------------------------
    page1_template: str = "page1-base.jpg"
    page2_template: str = "page2-base.jpg"
    mkcl_logo: str = "mkcl-logo.png"

    # -- Concurrency ---------------------------------------------------------
    max_workers: int = Field(
        default=min(os.cpu_count() or 4, 8),
        description="Thread-pool size for CPU-bound validation work",
    )
    uvicorn_workers: int = Field(default=4, description="Uvicorn process workers")

    # -- Server --------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 9000
    debug: bool = False

    # -- Template Matching ---------------------------------------------------
    template_match_threshold: float = 0.20
    header_match_threshold: float = 0.30
    logo_match_threshold: float = 0.35
    structural_similarity_threshold: float = 0.20

    # -- OCR Anchors ---------------------------------------------------------
    page1_ocr_anchors: list[str] = [
        "Application Form",
        "To be filled in by the Applicant only",
        "Page 1 of 2",
    ]
    page2_ocr_anchors: list[str] = [
        "Application Form",
        "Page 2 of 2",
    ]

    # -- Passport Photo Detection --------------------------------------------
    photo_region_page1: dict = {
        "x_ratio": 0.78,
        "y_ratio": 0.12,
        "w_ratio": 0.16,
        "h_ratio": 0.12,
    }
    photo_min_face_confidence: float = 0.50
    photo_min_content_ratio: float = 0.08

    # -- Applicant Signature Detection (Page 1) ------------------------------
    # Relative coordinates on page 1 for "Signature of Applicant" box
    signature_region_page1: dict = {
        "x_ratio": 0.55,
        "y_ratio": 0.885,
        "w_ratio": 0.18,
        "h_ratio": 0.035,
    }
    # -- Applicant Signature Detection (Page 2) ------------------------------
    signature_region_page2: dict = {
        "x_ratio": 0.45,
        "y_ratio": 0.91,
        "w_ratio": 0.20,
        "h_ratio": 0.04,
    }
    signature_min_stroke_density: float = 0.005
    signature_min_contour_count: int = 3
    signature_max_contour_count: int = 500

    # -- Blank Page Detection ------------------------------------------------
    blank_page_content_threshold: float = 0.02

    # -- Image Preprocessing -------------------------------------------------
    target_dpi: int = 300
    max_image_dimension: int = 3000

    # -- Redis (optional caching) --------------------------------------------
    redis_url: Optional[str] = None
    redis_ttl_seconds: int = 300

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    # -- Upload limits -------------------------------------------------------
    max_upload_size_mb: int = 20

    model_config = {
        "env_prefix": "FV_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
