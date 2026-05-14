"""
Page Validation Service.

Validates individual pages of the uploaded form:
  - Determines page identity (Page 1 / Page 2 / unknown)
  - Checks for blank pages
  - Verifies correct form structure
  - Detects MKCL logo presence
  - Computes header match score
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import PagePresenceResult, PageValidationResult
from app.services.template_matching import TemplateMatchingService
from app.utils.image_processing import (
    compute_content_ratio,
    normalize_dimensions,
    preprocess_for_matching,
    to_grayscale,
)

logger = get_logger(__name__)


class PageValidationService:
    """
    Stateless page validator.

    Delegates template matching to ``TemplateMatchingService`` and adds
    blank-page detection, page-sequence checks, and structural validation.
    """

    def __init__(self) -> None:
        self._template_matcher = TemplateMatchingService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_pages(
        self,
        page_images: List[np.ndarray],
        *,
        ocr_texts_per_page: Optional[List[List[str]]] = None,
    ) -> Tuple[PagePresenceResult, List[PageValidationResult]]:
        """
        Validate all pages of an uploaded document.

        Parameters
        ----------
        page_images : list[np.ndarray]
            BGR images, one per page.
        ocr_texts_per_page : list[list[str]], optional
            OCR-extracted text per page (for anchor verification).

        Returns
        -------
        tuple[PagePresenceResult, list[PageValidationResult]]
        """
        settings = get_settings()
        page_details: List[PageValidationResult] = []
        found_pages: set[int] = set()

        for idx, img in enumerate(page_images):
            ocr_texts = (
                ocr_texts_per_page[idx]
                if ocr_texts_per_page and idx < len(ocr_texts_per_page)
                else None
            )
            result = self._validate_single_page(img, idx + 1, ocr_texts=ocr_texts)
            page_details.append(result)

            if result.present and result.is_correct_form:
                found_pages.add(result.page_number)

        # Determine missing mandatory pages
        mandatory = {1, 2}
        missing = sorted(mandatory - found_pages)

        presence = PagePresenceResult(
            page1=1 in found_pages,
            page2=2 in found_pages,
            total_pages=len(page_images),
            missing_pages=missing,
        )

        logger.info(
            "page_validation_complete",
            total=len(page_images),
            page1=presence.page1,
            page2=presence.page2,
            missing=missing,
        )
        return presence, page_details

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_single_page(
        self,
        img: np.ndarray,
        ordinal: int,
        *,
        ocr_texts: Optional[List[str]] = None,
    ) -> PageValidationResult:
        """Validate a single page image."""
        settings = get_settings()

        # 1) Blank-page check
        gray = to_grayscale(normalize_dimensions(img))
        content_ratio = compute_content_ratio(gray)
        is_blank = content_ratio < settings.blank_page_content_threshold

        if is_blank:
            logger.warning("blank_page_detected", ordinal=ordinal, content_ratio=content_ratio)
            return PageValidationResult(
                present=False,
                page_number=ordinal,
                confidence=1.0 - content_ratio,
                is_blank=True,
                is_correct_form=False,
            )

        # 2) Template matching → identify which page
        page_num, confidence = self._template_matcher.identify_page(
            img, ocr_texts=ocr_texts
        )

        # 3) Logo detection score
        logo_detected = False
        try:
            from app.utils.template_cache import cached_imread_gray

            logo_gray = cached_imread_gray(settings.mkcl_logo)
            logo_score = self._detect_logo(gray, logo_gray)
            logo_detected = logo_score >= settings.logo_match_threshold
        except Exception:
            logo_score = 0.0

        # 4) Header match (basic structural check)
        is_correct = page_num is not None and confidence >= settings.template_match_threshold

        return PageValidationResult(
            present=True,
            page_number=page_num if page_num else ordinal,
            confidence=round(confidence, 4),
            is_blank=False,
            is_correct_form=is_correct,
            logo_detected=logo_detected,
            header_match_score=round(confidence, 4),
        )

    @staticmethod
    def _detect_logo(
        page_gray: np.ndarray, logo_gray: np.ndarray
    ) -> float:
        """Multi-scale logo detection returning max correlation value."""
        best = 0.0
        for scale in (0.3, 0.5, 0.7):
            lh, lw = logo_gray.shape[:2]
            nw, nh = int(lw * scale), int(lh * scale)
            if nw < 10 or nh < 10:
                continue
            logo_s = cv2.resize(logo_gray, (nw, nh), interpolation=cv2.INTER_AREA)
            if logo_s.shape[0] > page_gray.shape[0] or logo_s.shape[1] > page_gray.shape[1]:
                continue
            res = cv2.matchTemplate(page_gray, logo_s, cv2.TM_CCOEFF_NORMED)
            _, mx, _, _ = cv2.minMaxLoc(res)
            best = max(best, float(mx))
        return best
