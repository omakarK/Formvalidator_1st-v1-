"""
Passport Photo Detection Service.

Detects passport-size photo in the designated region of Page 1.
Uses MediaPipe + Haar cascade ensemble for face detection.
"""

from __future__ import annotations
from typing import Optional
import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import PhotoValidationResult
from app.utils.image_processing import (
    compute_content_ratio, crop_region, normalize_dimensions, to_grayscale,
)

logger = get_logger(__name__)

_haar_cascade = None
_mediapipe_detector = None


def _get_haar_cascade():
    global _haar_cascade
    if _haar_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(path)
    return _haar_cascade


def _get_mediapipe_detector():
    global _mediapipe_detector
    if _mediapipe_detector is None:
        try:
            import mediapipe as mp
            _mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.4,
            )
        except ImportError:
            _mediapipe_detector = "unavailable"
    return _mediapipe_detector


class PhotoDetectionService:
    """Stateless passport photo detector."""

    def validate(self, page_img: np.ndarray, *, page_number: int = 1) -> PhotoValidationResult:
        settings = get_settings()
        if page_number != 1:
            return PhotoValidationResult()

        try:
            img = normalize_dimensions(page_img)
            r = settings.photo_region_page1
            photo_crop = crop_region(img, r["x_ratio"], r["y_ratio"], r["w_ratio"], r["h_ratio"])
            gray_crop = to_grayscale(photo_crop)
            content_ratio = compute_content_ratio(gray_crop, threshold=220)
            has_content = content_ratio >= settings.photo_min_content_ratio

            if not has_content:
                return PhotoValidationResult(confidence=round(1.0 - content_ratio, 4))

            face_det, face_conf = self._detect_face(photo_crop)
            is_logo = self._is_logo_or_stamp(gray_crop)
            if is_logo and not face_det:
                return PhotoValidationResult(confidence=0.2, region_has_content=True)

            present = face_det and face_conf >= settings.photo_min_face_confidence
            conf = face_conf if present else max(0.1, content_ratio * 0.5)
            return PhotoValidationResult(
                photo_present=present, face_detected=face_det,
                confidence=round(conf, 4), region_has_content=has_content,
            )
        except Exception as exc:
            logger.error("photo_validation_error", error=str(exc))
            return PhotoValidationResult()

    def _detect_face(self, crop_bgr: np.ndarray) -> tuple[bool, float]:
        mp_r = self._detect_face_mediapipe(crop_bgr)
        if mp_r[0]:
            return mp_r
        return self._detect_face_haar(crop_bgr)

    @staticmethod
    def _detect_face_mediapipe(crop_bgr: np.ndarray) -> tuple[bool, float]:
        det = _get_mediapipe_detector()
        if det == "unavailable" or det is None:
            return False, 0.0
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            res = det.process(rgb)
            if res.detections:
                return True, float(max(d.score[0] for d in res.detections))
            return False, 0.0
        except Exception:
            return False, 0.0

    @staticmethod
    def _detect_face_haar(crop_bgr: np.ndarray) -> tuple[bool, float]:
        try:
            cascade = _get_haar_cascade()
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
            if len(faces) > 0:
                h, w = gray.shape[:2]
                best_area = max(fw * fh for (_, _, fw, fh) in faces)
                conf = min(0.5 + (best_area / (h * w)) * 2.0, 0.95)
                return True, conf
            return False, 0.0
        except Exception:
            return False, 0.0

    @staticmethod
    def _is_logo_or_stamp(gray_crop: np.ndarray) -> bool:
        if gray_crop.size == 0:
            return False
        hist = cv2.calcHist([gray_crop], [0], None, [256], [0, 256]).flatten() / gray_crop.size
        extreme = float(hist[:30].sum() + hist[225:].sum())
        mid = float(hist[60:200].sum())
        return extreme > 0.7 and mid < 0.2
