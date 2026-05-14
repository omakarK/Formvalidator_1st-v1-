"""
Image preprocessing utilities.

Shared helpers for skew correction, denoising, contrast enhancement,
and dimension normalization — used by every validation service.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw file bytes into a BGR numpy array.

    Parameters
    ----------
    raw_bytes : bytes
        Raw image bytes (JPEG, PNG, etc.).

    Returns
    -------
    np.ndarray
        Decoded BGR image.

    Raises
    ------
    ValueError
        If decoding fails.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from bytes")
    return img


def normalize_dimensions(img: np.ndarray) -> np.ndarray:
    """Resize image so the largest dimension is at most ``max_image_dimension``."""
    settings = get_settings()
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > settings.max_image_dimension:
        scale = settings.max_image_dimension / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light Gaussian denoising."""
    return cv2.GaussianBlur(gray, (3, 3), 0)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Correct minor skew using Hough line detection.

    Only corrects angles within ±5 degrees to avoid over-rotation.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 5.0:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.1:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_for_matching(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: normalize → gray → enhance → denoise."""
    img = normalize_dimensions(img)
    gray = to_grayscale(img)
    gray = enhance_contrast(gray)
    gray = denoise(gray)
    return gray


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Preprocessing optimized for OCR: normalize → gray → deskew → enhance."""
    img = normalize_dimensions(img)
    gray = to_grayscale(img)
    gray = deskew(gray)
    gray = enhance_contrast(gray)
    return gray


def crop_region(
    img: np.ndarray,
    x_ratio: float,
    y_ratio: float,
    w_ratio: float,
    h_ratio: float,
) -> np.ndarray:
    """
    Crop a region from an image using relative ratios.

    Parameters
    ----------
    img : np.ndarray
        Source image.
    x_ratio, y_ratio : float
        Top-left corner as ratios of image dimensions.
    w_ratio, h_ratio : float
        Width and height as ratios of image dimensions.

    Returns
    -------
    np.ndarray
        Cropped region.
    """
    h, w = img.shape[:2]
    x1 = int(w * x_ratio)
    y1 = int(h * y_ratio)
    x2 = int(w * (x_ratio + w_ratio))
    y2 = int(h * (y_ratio + h_ratio))

    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return img[y1:y2, x1:x2]


def compute_content_ratio(gray: np.ndarray, threshold: int = 200) -> float:
    """
    Compute the ratio of non-white pixels in a grayscale image.

    Useful for blank-page detection and content-presence checks.
    """
    if gray.size == 0:
        return 0.0
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return float(np.count_nonzero(binary)) / gray.size


def resize_to_match(
    img: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Resize ``img`` to match the dimensions of ``target``."""
    h, w = target.shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
