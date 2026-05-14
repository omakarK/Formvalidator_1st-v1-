"""
Applicant Signature Detection Service.

IMPORTANT: Validates ONLY the applicant's signature region.
Does NOT validate parent/guardian, office, or verifier signatures.

Detection pipeline:
  1. Crop the applicant signature region using template coordinates
  2. Binarize and threshold to isolate ink strokes
  3. Contour analysis for handwriting characteristics
  4. Connected component analysis for stroke density
  5. Reject blank boxes, printed text, and template lines
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import SignatureValidationResult
from app.utils.image_processing import (
    compute_content_ratio,
    crop_region,
    normalize_dimensions,
    to_grayscale,
)

logger = get_logger(__name__)


class SignatureDetectionService:
    """
    Stateless applicant-signature detector.

    Uses template-defined coordinate regions to crop ONLY the
    applicant signature box — never the parent/guardian area.
    """

    def validate(
        self,
        page_img: np.ndarray,
        *,
        page_number: int = 1,
    ) -> SignatureValidationResult:
        """
        Validate the applicant signature on the given page.

        Parameters
        ----------
        page_img : np.ndarray
            Full-page BGR image.
        page_number : int
            Page number (signature expected on pages 1 and 2).
        """
        settings = get_settings()

        # Select the correct region for this page
        if page_number == 1:
            region_cfg = settings.signature_region_page1
        elif page_number == 2:
            region_cfg = settings.signature_region_page2
        else:
            return SignatureValidationResult()

        try:
            img = normalize_dimensions(page_img)
            sig_crop = crop_region(
                img,
                region_cfg["x_ratio"],
                region_cfg["y_ratio"],
                region_cfg["w_ratio"],
                region_cfg["h_ratio"],
            )

            gray = to_grayscale(sig_crop)

            # Step 1: Check for content presence
            content_ratio = compute_content_ratio(gray, threshold=180)
            if content_ratio < settings.signature_min_stroke_density:
                logger.info(
                    "signature_region_empty",
                    page=page_number,
                    content_ratio=round(content_ratio, 6),
                )
                return SignatureValidationResult(
                    confidence=round(1.0 - content_ratio, 4),
                )

            # Step 2: Binary thresholding to isolate ink
            _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

            # Step 3: Remove template lines (horizontal/vertical)
            cleaned = self._remove_template_lines(binary)

            # Step 4: Contour analysis
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_count = len(contours)

            # Step 5: Stroke density on cleaned image
            stroke_density = float(np.count_nonzero(cleaned)) / max(cleaned.size, 1)

            # Step 6: Check for handwriting characteristics
            has_handwriting = self._has_handwriting_characteristics(
                contours, cleaned.shape
            )

            # Step 7: Reject printed text patterns
            is_printed = self._is_printed_text(contours, cleaned.shape)

            # Decision logic
            min_strokes = settings.signature_min_stroke_density
            min_contours = settings.signature_min_contour_count
            max_contours = settings.signature_max_contour_count

            sig_present = (
                stroke_density >= min_strokes
                and min_contours <= contour_count <= max_contours
                and has_handwriting
                and not is_printed
            )

            # Confidence scoring
            if sig_present:
                density_score = min(stroke_density / 0.05, 1.0)
                contour_score = min(contour_count / 20.0, 1.0)
                confidence = 0.5 * density_score + 0.3 * contour_score + 0.2
            else:
                confidence = max(stroke_density * 2, 0.05)

            result = SignatureValidationResult(
                applicant_signature_present=sig_present,
                confidence=round(min(confidence, 1.0), 4),
                stroke_density=round(stroke_density, 6),
                contour_count=contour_count,
                has_handwriting=has_handwriting,
            )

            logger.info(
                "signature_validation_done",
                page=page_number,
                present=sig_present,
                confidence=result.confidence,
                stroke_density=result.stroke_density,
                contours=contour_count,
            )
            return result

        except Exception as exc:
            logger.error("signature_validation_error", error=str(exc))
            return SignatureValidationResult()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_template_lines(binary: np.ndarray) -> np.ndarray:
        """
        Remove horizontal and vertical template lines from the
        binary image so they don't get counted as signature strokes.
        """
        h, w = binary.shape[:2]
        cleaned = binary.copy()

        # Remove horizontal lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 20), 1))
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
        cleaned = cv2.subtract(cleaned, horiz_lines)

        # Remove vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 4, 20)))
        vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
        cleaned = cv2.subtract(cleaned, vert_lines)

        return cleaned

    @staticmethod
    def _has_handwriting_characteristics(
        contours: list, shape: tuple
    ) -> bool:
        """
        Heuristic: handwriting has varied contour sizes, non-uniform
        spacing, and a mix of small and medium contours.
        """
        if len(contours) < 2:
            return False

        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return False

        total_area = shape[0] * shape[1]
        # Filter out noise (very tiny contours)
        significant = [a for a in areas if a > total_area * 0.0001]
        if len(significant) < 2:
            return False

        # Handwriting: coefficient of variation of contour areas > 0.3
        mean_area = float(np.mean(significant))
        if mean_area == 0:
            return False
        std_area = float(np.std(significant))
        cv = std_area / mean_area

        return cv > 0.3

    @staticmethod
    def _is_printed_text(contours: list, shape: tuple) -> bool:
        """
        Heuristic: printed text has very uniform contour sizes and
        regular horizontal spacing — unlike handwritten signatures.
        """
        if len(contours) < 5:
            return False

        # Get bounding boxes
        bboxes = [cv2.boundingRect(c) for c in contours]
        heights = [h for (_, _, _, h) in bboxes]
        if not heights:
            return False

        mean_h = float(np.mean(heights))
        if mean_h == 0:
            return False
        std_h = float(np.std(heights))

        # Printed text: very uniform character heights (CV < 0.15)
        cv_height = std_h / mean_h

        # Also check Y-coordinate alignment
        y_coords = [y for (_, y, _, _) in bboxes]
        y_std = float(np.std(y_coords))
        img_h = shape[0]

        # Printed text: characters aligned on same baseline
        is_aligned = y_std < img_h * 0.1
        is_uniform = cv_height < 0.15

        return is_aligned and is_uniform and len(contours) > 10
