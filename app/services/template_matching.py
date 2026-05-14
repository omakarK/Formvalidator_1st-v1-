"""
Template Matching Engine — optimized for FILLED scanned forms.

Matches an uploaded form page against known templates using methods
that are robust to handwriting, photos, and scan quality variations:

  1. Header ORB matching  (top 20% — captures title + course-selection row)
  2. Structural grid matching (morphological line extraction — grid is constant)
  3. Edge-map spatial correlation (Canny edge template matching)
  4. Footer ORB matching  (bottom 8% — captures "Page X of 2")
  5. Logo detection        (multi-scale MKCL logo matching)
  6. OCR anchor text verification

Key insight: filled forms differ dramatically from blank templates in
pixel content, but the HEADER region, STRUCTURAL GRID, FOOTER region,
and LOGO remain nearly identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import TemplateMatchResult
from app.utils.image_processing import (
    normalize_dimensions,
    preprocess_for_matching,
    to_grayscale,
)
from app.utils.template_cache import cached_imread, cached_imread_gray

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Region percentages — tuned for MKCL application forms
# -----------------------------------------------------------------------
# Header: MKCL logo + "Application Form" + course-selection row
_HEADER_PCT = 0.20
# Footer: "Page X of 2" line + bottom margin
_FOOTER_PCT = 0.08
# Standard working width for feature extraction (keeps aspect ratio)
_WORKING_WIDTH = 800


@dataclass
class TemplateDescriptor:
    """Pre-computed features for a known template."""
    name: str
    page_number: int
    gray: np.ndarray
    # Header region (top 20%)
    header_gray: Optional[np.ndarray] = None
    header_kp: Optional[list] = None
    header_des: Optional[np.ndarray] = None
    # Footer region (bottom 8%)
    footer_gray: Optional[np.ndarray] = None
    footer_kp: Optional[list] = None
    footer_des: Optional[np.ndarray] = None
    # Full-page ORB (supplementary)
    full_kp: Optional[list] = None
    full_des: Optional[np.ndarray] = None
    # Structural grid fingerprint (binary image of extracted lines)
    grid_lines: Optional[np.ndarray] = None
    # Edge map for spatial correlation
    edge_map: Optional[np.ndarray] = None
    # OCR anchors
    ocr_anchors: Optional[List[str]] = None


class TemplateMatchingService:
    """
    Template matching engine optimized for filled scanned forms.

    Uses a multi-signal ensemble:
      - Header ORB    (PRIMARY — header is identical blank vs filled)
      - Structural grid (grid lines / borders are permanent)
      - Edge-map spatial correlation (preserves spatial layout info)
      - Footer ORB    (page-number region discriminates page 1 vs 2)
      - Logo detection (confirms MKCL form type)
      - OCR anchors   (text verification when available)
    """

    def __init__(self) -> None:
        self._descriptors: Dict[str, TemplateDescriptor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        page_img: np.ndarray,
        *,
        ocr_texts: Optional[List[str]] = None,
    ) -> TemplateMatchResult:
        """
        Match a single page image against all known templates.

        Returns the best-matching template with a fused confidence score.
        """
        settings = get_settings()
        self._ensure_descriptors_loaded()

        best_name: Optional[str] = None
        best_confidence: float = 0.0
        best_method: str = "none"

        # Preprocess the uploaded page
        page_gray = preprocess_for_matching(page_img)
        h, w = page_gray.shape[:2]
        page_header = page_gray[0 : int(h * _HEADER_PCT), :]
        page_footer = page_gray[int(h * (1.0 - _FOOTER_PCT)) :, :]

        for tpl_name, tpl_desc in self._descriptors.items():
            scores: Dict[str, float] = {}

            # 1) Header ORB — PRIMARY discriminator (top 20%)
            scores["header_orb"] = self._header_orb_score(page_header, tpl_desc)

            # 2) Structural grid matching (morphological lines)
            scores["grid"] = self._structural_grid_score(page_gray, tpl_desc)

            # 3) Edge-map spatial correlation
            scores["edge_spatial"] = self._edge_spatial_score(page_gray, tpl_desc)

            # 4) Footer ORB — page-number region
            scores["footer_orb"] = self._footer_orb_score(page_footer, tpl_desc)

            # 5) Logo detection
            scores["logo"] = self._logo_score(page_img)

            # 6) OCR anchor verification
            if ocr_texts and tpl_desc.ocr_anchors:
                scores["ocr"] = self._ocr_anchor_score(ocr_texts, tpl_desc.ocr_anchors)
            else:
                scores["ocr"] = 0.0

            # Fuse scores
            fused = self._fuse_scores(scores)

            logger.info(
                "template_match_scores",
                template=tpl_name,
                header_orb=round(scores["header_orb"], 4),
                grid=round(scores["grid"], 4),
                edge_spatial=round(scores["edge_spatial"], 4),
                footer_orb=round(scores["footer_orb"], 4),
                logo=round(scores["logo"], 4),
                ocr=round(scores["ocr"], 4),
                fused=round(fused, 4),
            )

            if fused > best_confidence:
                best_confidence = fused
                best_name = tpl_name
                best_method = max(scores, key=scores.get)

        matched = best_confidence >= settings.template_match_threshold
        return TemplateMatchResult(
            matched=matched,
            template_name=best_name if matched else None,
            confidence=round(best_confidence, 4),
            method=best_method if matched else None,
        )

    def identify_page(
        self,
        page_img: np.ndarray,
        *,
        ocr_texts: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Identify which page number this image corresponds to.

        Returns (page_number, confidence) or (None, 0.0).
        """
        result = self.match(page_img, ocr_texts=ocr_texts)
        if result.matched and result.template_name:
            desc = self._descriptors.get(result.template_name)
            if desc:
                return desc.page_number, result.confidence
        return None, 0.0

    # ------------------------------------------------------------------
    # Descriptor construction
    # ------------------------------------------------------------------

    def _ensure_descriptors_loaded(self) -> None:
        """Lazily build template descriptors from the cached images."""
        if self._descriptors:
            return

        settings = get_settings()
        orb = cv2.ORB_create(nfeatures=1000)

        for tpl_name, filename, page_num, anchors in [
            ("mkcl_page1", settings.page1_template, 1, settings.page1_ocr_anchors),
            ("mkcl_page2", settings.page2_template, 2, settings.page2_ocr_anchors),
        ]:
            tpl_img = cached_imread(filename)
            gray = preprocess_for_matching(tpl_img)
            h, w = gray.shape[:2]

            # -- Header (top 20%) --
            header = gray[0 : int(h * _HEADER_PCT), :]
            hdr_resized = self._resize_to_working(header)
            h_kp, h_des = orb.detectAndCompute(hdr_resized, None)

            # -- Footer (bottom 8%) --
            footer = gray[int(h * (1.0 - _FOOTER_PCT)) :, :]
            ftr_resized = self._resize_to_working(footer)
            f_kp, f_des = orb.detectAndCompute(ftr_resized, None)

            # -- Full-page ORB (supplementary) --
            full_resized = self._resize_to_working(gray)
            full_kp, full_des = orb.detectAndCompute(full_resized, None)

            # -- Structural grid (morphological line extraction) --
            grid_lines = self._extract_grid_lines(gray)

            # -- Edge map --
            edge_map = cv2.Canny(
                self._resize_to_working(gray), 50, 150
            )

            self._descriptors[tpl_name] = TemplateDescriptor(
                name=tpl_name,
                page_number=page_num,
                gray=gray,
                header_gray=hdr_resized,
                header_kp=h_kp,
                header_des=h_des,
                footer_gray=ftr_resized,
                footer_kp=f_kp,
                footer_des=f_des,
                full_kp=full_kp,
                full_des=full_des,
                grid_lines=grid_lines,
                edge_map=edge_map,
                ocr_anchors=anchors,
            )

        logger.info(
            "template_descriptors_built",
            templates=list(self._descriptors.keys()),
        )

    # ------------------------------------------------------------------
    # Matching methods
    # ------------------------------------------------------------------

    def _header_orb_score(
        self, page_header: np.ndarray, tpl: TemplateDescriptor
    ) -> float:
        """
        ORB matching on header region ONLY (top 20%).

        This is the PRIMARY discriminator because the header
        (title, logo, course selection) is IDENTICAL between
        blank templates and filled forms.

        Uses Lowe's ratio test with knnMatch for robust matching,
        then computes score from the ratio of good matches.
        """
        try:
            if tpl.header_des is None:
                return 0.0

            orb = cv2.ORB_create(nfeatures=1000)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            # Resize page header to match template header dimensions
            page_hdr = cv2.resize(
                page_header,
                (tpl.header_gray.shape[1], tpl.header_gray.shape[0]),
            )
            kp, des = orb.detectAndCompute(page_hdr, None)
            if des is None or len(des) < 2:
                return 0.0

            # Use knnMatch + ratio test (more robust than crossCheck)
            matches = bf.knnMatch(des, tpl.header_des, k=2)
            good = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            # Score: ratio of good matches to total keypoints
            max_possible = min(len(kp), len(tpl.header_kp)) if tpl.header_kp else 1
            score = len(good) / max(max_possible, 1)
            # Clamp and scale to [0, 1]
            return min(score * 2.5, 1.0)

        except Exception as exc:
            logger.warning("header_orb_error", error=str(exc))
            return 0.0

    def _footer_orb_score(
        self, page_footer: np.ndarray, tpl: TemplateDescriptor
    ) -> float:
        """
        ORB matching on footer region (bottom 8%).

        The footer contains "Page 1 of 2" or "Page 2 of 2" — a
        critical discriminator between page 1 and page 2.
        Also includes signature labels that differ between pages.
        """
        try:
            if tpl.footer_des is None:
                return 0.0

            orb = cv2.ORB_create(nfeatures=500)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            page_ftr = cv2.resize(
                page_footer,
                (tpl.footer_gray.shape[1], tpl.footer_gray.shape[0]),
            )
            kp, des = orb.detectAndCompute(page_ftr, None)
            if des is None or len(des) < 2:
                return 0.0

            matches = bf.knnMatch(des, tpl.footer_des, k=2)
            good = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            max_possible = min(len(kp), len(tpl.footer_kp)) if tpl.footer_kp else 1
            score = len(good) / max(max_possible, 1)
            return min(score * 2.5, 1.0)

        except Exception as exc:
            logger.warning("footer_orb_error", error=str(exc))
            return 0.0

    def _structural_grid_score(
        self, page_gray: np.ndarray, tpl: TemplateDescriptor
    ) -> float:
        """
        Compare structural grid lines between page and template.

        Uses morphological operations to extract horizontal and vertical
        lines (table borders, section dividers) which are PERMANENT
        structural features — they exist identically in blank and filled
        forms, regardless of handwriting or photos.

        The extracted line masks are compared using normalized correlation.
        """
        try:
            if tpl.grid_lines is None:
                return 0.0

            # Extract grid lines from the uploaded page
            page_grid = self._extract_grid_lines(page_gray)

            # Resize both to same dimensions for comparison
            target_h, target_w = tpl.grid_lines.shape[:2]
            page_grid_resized = cv2.resize(
                page_grid, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

            # Normalized cross-correlation on the grid-line masks
            result = cv2.matchTemplate(
                page_grid_resized, tpl.grid_lines, cv2.TM_CCOEFF_NORMED
            )
            # matchTemplate returns a 1×1 result when images are same size
            score = float(result[0, 0]) if result.size > 0 else 0.0

            # Clamp to [0, 1]
            return max(score, 0.0)

        except Exception as exc:
            logger.warning("structural_grid_error", error=str(exc))
            return 0.0

    def _edge_spatial_score(
        self, page_gray: np.ndarray, tpl: TemplateDescriptor
    ) -> float:
        """
        Edge-map spatial correlation — compares WHERE edges are, not just
        how many there are.

        Unlike the old edge_density (single scalar), this preserves
        spatial structure: grid borders at specific positions will
        correlate strongly even when handwriting adds extra edges.
        """
        try:
            if tpl.edge_map is None:
                return 0.0

            # Compute edge map for the page
            page_resized = self._resize_to_working(page_gray)
            page_edges = cv2.Canny(page_resized, 50, 150)

            # Ensure same dimensions
            if page_edges.shape != tpl.edge_map.shape:
                page_edges = cv2.resize(
                    page_edges,
                    (tpl.edge_map.shape[1], tpl.edge_map.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

            # Normalized cross-correlation
            result = cv2.matchTemplate(
                page_edges, tpl.edge_map, cv2.TM_CCOEFF_NORMED
            )
            score = float(result[0, 0]) if result.size > 0 else 0.0
            return max(score, 0.0)

        except Exception:
            return 0.0

    def _logo_score(self, page_img: np.ndarray) -> float:
        """
        Detect MKCL logo using multi-scale template matching.

        Uses smaller scales (0.05-0.30) tuned for scanned forms
        where the logo appears smaller than in the original template.
        """
        try:
            settings = get_settings()
            logo_gray = cached_imread_gray(settings.mkcl_logo)
            page_gray = to_grayscale(normalize_dimensions(page_img))

            best_val = 0.0
            for scale in (0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3):
                lh, lw = logo_gray.shape[:2]
                new_w, new_h = int(lw * scale), int(lh * scale)
                if new_w < 10 or new_h < 10:
                    continue
                logo_scaled = cv2.resize(
                    logo_gray, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                if (
                    logo_scaled.shape[0] > page_gray.shape[0]
                    or logo_scaled.shape[1] > page_gray.shape[1]
                ):
                    continue
                result = cv2.matchTemplate(
                    page_gray, logo_scaled, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)
                best_val = max(best_val, float(max_val))

            return best_val
        except Exception:
            return 0.0

    @staticmethod
    def _ocr_anchor_score(
        ocr_texts: List[str], anchors: List[str]
    ) -> float:
        """Check how many OCR anchor strings are found in the text."""
        if not anchors:
            return 0.0
        full_text = " ".join(ocr_texts).lower()
        found = sum(1 for a in anchors if a.lower() in full_text)
        return found / len(anchors)

    # ------------------------------------------------------------------
    # Structural line extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_grid_lines(gray: np.ndarray) -> np.ndarray:
        """
        Extract horizontal and vertical structural lines using
        morphological operations.

        This isolates the TABLE BORDERS, SECTION DIVIDERS, and BOX
        OUTLINES that are permanent in the form template. These lines
        are identical whether the form is blank or filled with
        handwriting / photos.

        Returns a binary image containing only the structural lines.
        """
        # Work at a consistent scale
        scale_w = 600
        aspect = gray.shape[0] / max(gray.shape[1], 1)
        scale_h = int(scale_w * aspect)
        resized = cv2.resize(gray, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

        # Binarize (Otsu threshold)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # -- Extract horizontal lines --
        h_kernel_len = scale_w // 15  # ~40px at 600w
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # -- Extract vertical lines --
        v_kernel_len = scale_h // 20  # ~proportional
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # Combine horizontal + vertical
        grid = cv2.add(h_lines, v_lines)

        # Light dilation to thicken lines for better correlation
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid = cv2.dilate(grid, dilate_kernel, iterations=1)

        return grid

    # ------------------------------------------------------------------
    # Score fusion — tuned for structural stability in filled forms
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_scores(scores: Dict[str, float]) -> float:
        """
        Weighted average tuned for scanned filled forms.

        Header ORB and structural grid get the highest weights because
        they are the most stable features across blank and filled forms.
        """
        weights = {
            "header_orb":    0.30,   # PRIMARY — header is identical in filled forms
            "grid":          0.25,   # Structural grid lines are permanent
            "edge_spatial":  0.15,   # Edge layout correlation
            "footer_orb":    0.10,   # Footer "Page X of 2" discrimination
            "logo":          0.10,   # Logo is always present
            "ocr":           0.10,   # OCR anchors (when available)
        }
        total_weight = sum(weights.get(k, 0.0) for k in scores)
        if total_weight == 0:
            return 0.0
        fused = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in scores)
        return fused / total_weight

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_to_working(img: np.ndarray) -> np.ndarray:
        """Resize image to standard working width, preserving aspect ratio."""
        h, w = img.shape[:2]
        if w == 0:
            return img
        new_h = int(_WORKING_WIDTH * h / w)
        return cv2.resize(img, (_WORKING_WIDTH, new_h), interpolation=cv2.INTER_AREA)
