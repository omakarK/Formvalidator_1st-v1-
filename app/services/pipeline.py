"""
Validation Pipeline — the orchestrator.

Coordinates all validation services for a single uploaded document:
  1. Detect file type (PDF vs image)
  2. Convert PDF to page images (if needed)
  3. Run template matching on each page
  4. Validate page presence and sequence
  5. Validate passport photo (Page 1)
  6. Validate applicant signature (Page 1 primary, Page 2 secondary)
  7. Run OCR for anchor verification
  8. Aggregate results into a single ValidationResponse

This module is called from the worker layer (thread pool) and is
entirely synchronous / CPU-bound — no async here.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import (
    PagePresenceResult,
    SignatureValidationResult,
    ValidationResponse,
)
from app.services.ocr_service import OCRService
from app.services.page_validation import PageValidationService
from app.services.photo_detection import PhotoDetectionService
from app.services.signature_detection import SignatureDetectionService
from app.services.template_matching import TemplateMatchingService
from app.utils.image_processing import decode_image, normalize_dimensions

logger = get_logger(__name__)


class ValidationPipeline:
    """
    Full document validation pipeline.

    Stateless: each call processes a fresh document independently.
    Safe to call from multiple threads simultaneously.
    """

    def __init__(self) -> None:
        self._template_svc = TemplateMatchingService()
        self._page_svc = PageValidationService()
        self._photo_svc = PhotoDetectionService()
        self._signature_svc = SignatureDetectionService()
        self._ocr_svc = OCRService()

    def run(self, file_bytes: bytes, filename: str) -> ValidationResponse:
        """
        Execute the full validation pipeline.

        Parameters
        ----------
        file_bytes : bytes
            Raw uploaded file content.
        filename : str
            Original filename (used to detect PDF vs image).

        Returns
        -------
        ValidationResponse
        """
        start = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []

        try:
            # -------------------------------------------------------
            # Step 1: Convert to page images
            # -------------------------------------------------------
            page_images = self._extract_pages(file_bytes, filename)
            if not page_images:
                return ValidationResponse(
                    success=False,
                    errors=["No valid pages extracted from document"],
                    processing_time_ms=self._elapsed_ms(start),
                )

            logger.info("pages_extracted", count=len(page_images))

            # -------------------------------------------------------
            # Step 2: OCR extraction per page (for anchor matching)
            # -------------------------------------------------------
            ocr_per_page: List[List[str]] = []
            for idx, img in enumerate(page_images):
                try:
                    texts = self._ocr_svc.extract_text(img)
                    ocr_per_page.append(texts)
                except Exception:
                    ocr_per_page.append([])
                    warnings.append(f"OCR failed on page {idx + 1}")

            # -------------------------------------------------------
            # Step 3: Page validation + template matching
            # -------------------------------------------------------
            pages_result, page_details = self._page_svc.validate_pages(
                page_images, ocr_texts_per_page=ocr_per_page
            )

            # -------------------------------------------------------
            # Step 4: Template match (use best page for overall match)
            # -------------------------------------------------------
            template_result = self._template_svc.match(
                page_images[0],
                ocr_texts=ocr_per_page[0] if ocr_per_page else None,
            )

            # -------------------------------------------------------
            # Step 5: Photo validation (Page 1 only)
            # -------------------------------------------------------
            page1_img = self._find_page_image(page_images, page_details, target_page=1)
            photo_result = self._photo_svc.validate(
                page1_img, page_number=1
            ) if page1_img is not None else None

            # -------------------------------------------------------
            # Step 6: Signature validation (applicant ONLY)
            # -------------------------------------------------------
            sig_result = self._validate_applicant_signature(
                page_images, page_details
            )

            # -------------------------------------------------------
            # Step 7: Aggregate
            # -------------------------------------------------------
            overall_confidence = self._compute_overall_confidence(
                template_result, pages_result, photo_result, sig_result
            )

            response = ValidationResponse(
                success=True,
                template_matched=template_result.template_name,
                template_match=template_result,
                pages=pages_result,
                page_details=page_details,
                photo_present=photo_result.photo_present if photo_result else False,
                photo_validation=photo_result if photo_result else None,
                applicant_signature_present=sig_result.applicant_signature_present,
                signature_validation=sig_result,
                confidence=round(overall_confidence, 4),
                processing_time_ms=self._elapsed_ms(start),
                errors=errors,
                warnings=warnings,
            )

            logger.info(
                "validation_complete",
                template=response.template_matched,
                photo=response.photo_present,
                signature=response.applicant_signature_present,
                confidence=response.confidence,
                time_ms=response.processing_time_ms,
            )
            return response

        except Exception as exc:
            logger.error("pipeline_error", error=str(exc), exc_info=True)
            return ValidationResponse(
                success=False,
                errors=[f"Validation failed: {str(exc)}"],
                processing_time_ms=self._elapsed_ms(start),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_pages(
        self, file_bytes: bytes, filename: str
    ) -> List[np.ndarray]:
        """Convert uploaded file to list of page images."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "pdf":
            from app.utils.pdf_converter import pdf_to_images
            return pdf_to_images(file_bytes)

        # Single image
        img = decode_image(file_bytes)
        return [img]

    @staticmethod
    def _find_page_image(
        page_images: List[np.ndarray],
        page_details: list,
        target_page: int,
    ) -> Optional[np.ndarray]:
        """Find the image corresponding to a specific page number."""
        for i, detail in enumerate(page_details):
            if detail.page_number == target_page and detail.is_correct_form:
                return page_images[i]
        # Fallback: if only one page, assume it's the target
        if len(page_images) == 1:
            return page_images[0]
        # Fallback: return by index
        if target_page <= len(page_images):
            return page_images[target_page - 1]
        return None

    def _validate_applicant_signature(
        self,
        page_images: List[np.ndarray],
        page_details: list,
    ) -> SignatureValidationResult:
        """
        Validate applicant signature — try Page 1 first, then Page 2.
        NEVER validates parent/guardian signature regions.
        """
        # Try page 1 first
        p1 = self._find_page_image(page_images, page_details, target_page=1)
        if p1 is not None:
            result = self._signature_svc.validate(p1, page_number=1)
            if result.applicant_signature_present:
                return result

        # Try page 2 as secondary
        p2 = self._find_page_image(page_images, page_details, target_page=2)
        if p2 is not None:
            result = self._signature_svc.validate(p2, page_number=2)
            if result.applicant_signature_present:
                return result

        # Return best attempt (page 1 if available)
        if p1 is not None:
            return self._signature_svc.validate(p1, page_number=1)
        return SignatureValidationResult()

    @staticmethod
    def _compute_overall_confidence(
        template_result, pages_result, photo_result, sig_result
    ) -> float:
        """Weighted overall confidence score."""
        scores = []
        weights = []

        scores.append(template_result.confidence)
        weights.append(0.30)

        page_score = 0.0
        if pages_result.page1:
            page_score += 0.5
        if pages_result.page2:
            page_score += 0.5
        scores.append(page_score)
        weights.append(0.25)

        if photo_result:
            scores.append(photo_result.confidence)
        else:
            scores.append(0.0)
        weights.append(0.20)

        scores.append(sig_result.confidence)
        weights.append(0.25)

        total_w = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_w

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
