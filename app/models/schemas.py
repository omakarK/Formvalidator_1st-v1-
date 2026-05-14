"""
Pydantic response/request schemas for the validation API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-results
# ---------------------------------------------------------------------------

class TemplateMatchResult(BaseModel):
    """Result of template-matching against known form types."""
    matched: bool = False
    template_name: Optional[str] = None
    confidence: float = 0.0
    method: Optional[str] = None  # e.g. "feature_match", "ssim", "ocr_anchor"


class PageValidationResult(BaseModel):
    """Per-page validation result."""
    present: bool = False
    page_number: int = 0
    confidence: float = 0.0
    is_blank: bool = False
    is_correct_form: bool = False
    logo_detected: bool = False
    header_match_score: float = 0.0


class PhotoValidationResult(BaseModel):
    """Passport photo presence validation."""
    photo_present: bool = False
    face_detected: bool = False
    confidence: float = 0.0
    region_has_content: bool = False


class SignatureValidationResult(BaseModel):
    """Applicant signature validation (only the applicant's, not parent/guardian)."""
    applicant_signature_present: bool = False
    confidence: float = 0.0
    stroke_density: float = 0.0
    contour_count: int = 0
    has_handwriting: bool = False


class PagePresenceResult(BaseModel):
    """Multi-page presence summary."""
    page1: bool = False
    page2: bool = False
    total_pages: int = 0
    missing_pages: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class ValidationResponse(BaseModel):
    """Complete validation result returned by POST /validate-form."""
    success: bool = True
    template_matched: Optional[str] = None
    template_match: TemplateMatchResult = Field(default_factory=TemplateMatchResult)
    pages: PagePresenceResult = Field(default_factory=PagePresenceResult)
    page_details: List[PageValidationResult] = Field(default_factory=list)
    photo_present: bool = False
    photo_validation: PhotoValidationResult = Field(default_factory=PhotoValidationResult)
    applicant_signature_present: bool = False
    signature_validation: SignatureValidationResult = Field(
        default_factory=SignatureValidationResult
    )
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    debug: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    version: str = "1.0.0"
    templates_loaded: bool = False
    worker_pool_size: int = 0
