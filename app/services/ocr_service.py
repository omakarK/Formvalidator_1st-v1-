"""
OCR Service.

Provides text extraction for OCR anchor verification during
template matching and field validation.

Uses PaddleOCR as primary engine with Tesseract fallback.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import numpy as np

from app.logging_config import get_logger
from app.utils.image_processing import normalize_dimensions, preprocess_for_ocr

logger = get_logger(__name__)

_paddle_ocr = None
_paddle_lock = threading.Lock()


def _get_paddle_ocr():
    """Lazy-load PaddleOCR (heavy import, ~2s first call)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        with _paddle_lock:
            if _paddle_ocr is None:
                try:
                    from paddleocr import PaddleOCR
                    _paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                        show_log=False,
                        use_gpu=False,
                    )
                    logger.info("paddleocr_initialized")
                except ImportError:
                    _paddle_ocr = "unavailable"
                    logger.warning("paddleocr_not_available")
    return _paddle_ocr


class OCRService:
    """Stateless OCR text extractor."""

    def extract_text(self, img: np.ndarray) -> List[str]:
        """
        Extract text lines from an image.

        Returns list of detected text strings.
        """
        img = normalize_dimensions(img)
        # Try PaddleOCR first
        result = self._extract_paddle(img)
        if result:
            return result
        # Fallback to Tesseract
        return self._extract_tesseract(img)

    def extract_text_from_region(
        self,
        img: np.ndarray,
        x_ratio: float,
        y_ratio: float,
        w_ratio: float,
        h_ratio: float,
    ) -> List[str]:
        """Extract text from a specific region of the image."""
        from app.utils.image_processing import crop_region
        crop = crop_region(img, x_ratio, y_ratio, w_ratio, h_ratio)
        return self.extract_text(crop)

    @staticmethod
    def _extract_paddle(img: np.ndarray) -> List[str]:
        """Extract text using PaddleOCR."""
        ocr = _get_paddle_ocr()
        if ocr == "unavailable" or ocr is None:
            return []
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = ocr.ocr(rgb, cls=True)
            texts = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        texts.append(text)
            return texts
        except Exception as exc:
            logger.error("paddle_ocr_error", error=str(exc))
            return []

    @staticmethod
    def _extract_tesseract(img: np.ndarray) -> List[str]:
        """Extract text using Tesseract (fallback)."""
        try:
            import pytesseract
            gray = preprocess_for_ocr(img)
            text = pytesseract.image_to_string(gray, lang="eng")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            return lines
        except ImportError:
            logger.warning("tesseract_not_available")
            return []
        except Exception as exc:
            logger.error("tesseract_error", error=str(exc))
            return []
