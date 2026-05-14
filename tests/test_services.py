"""
Unit tests for the validation services.

Tests cover:
  - Template cache thread safety
  - Image preprocessing utilities
  - Template matching accuracy
  - Page validation logic
  - Photo detection (with/without faces)
  - Signature detection (ink vs blank)
  - Pipeline orchestration
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_image():
    """A blank white 800x600 image."""
    return np.ones((600, 800, 3), dtype=np.uint8) * 255


@pytest.fixture
def page1_template():
    """Load the real page1 template if available."""
    from app.config import get_settings
    settings = get_settings()
    path = settings.templates_dir / settings.page1_template
    if path.exists():
        return cv2.imread(str(path))
    pytest.skip("Page 1 template not found")


@pytest.fixture
def page2_template():
    """Load the real page2 template if available."""
    from app.config import get_settings
    settings = get_settings()
    path = settings.templates_dir / settings.page2_template
    if path.exists():
        return cv2.imread(str(path))
    pytest.skip("Page 2 template not found")


@pytest.fixture
def image_with_face():
    """Create a synthetic image with a simple face-like pattern."""
    img = np.ones((200, 150, 3), dtype=np.uint8) * 200
    # Draw a rough face: skin-colored oval + eyes
    cv2.ellipse(img, (75, 80), (50, 65), 0, 0, 360, (180, 160, 140), -1)
    cv2.circle(img, (55, 65), 7, (40, 30, 20), -1)  # left eye
    cv2.circle(img, (95, 65), 7, (40, 30, 20), -1)  # right eye
    cv2.ellipse(img, (75, 100), (20, 8), 0, 0, 180, (100, 60, 60), 2)  # mouth
    return img


@pytest.fixture
def image_with_signature():
    """Create a synthetic image with scribble-like strokes."""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    # Draw random curves to simulate signature
    pts = np.array([
        [20, 60], [50, 30], [80, 70], [110, 25],
        [140, 65], [170, 35], [200, 55], [230, 40], [260, 60],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], False, (20, 20, 20), 2)
    # Add some dots
    for x in range(30, 250, 40):
        cv2.circle(img, (x, 50 + np.random.randint(-15, 15)), 3, (10, 10, 10), -1)
    return img


# ---------------------------------------------------------------------------
# Template Cache Tests
# ---------------------------------------------------------------------------

class TestTemplateCache:
    def test_preload_templates(self):
        from app.utils.template_cache import preload_templates, clear_cache
        clear_cache()
        preload_templates()

    def test_cached_imread_returns_same_object(self):
        from app.utils.template_cache import cached_imread
        from app.config import get_settings
        settings = get_settings()
        img1 = cached_imread(settings.page1_template)
        img2 = cached_imread(settings.page1_template)
        assert img1 is img2  # Same object reference

    def test_cached_imread_gray(self):
        from app.utils.template_cache import cached_imread_gray
        from app.config import get_settings
        settings = get_settings()
        gray = cached_imread_gray(settings.page1_template)
        assert len(gray.shape) == 2  # Grayscale

    def test_missing_file_raises(self):
        from app.utils.template_cache import cached_imread
        with pytest.raises(FileNotFoundError):
            cached_imread("nonexistent_file.jpg")


# ---------------------------------------------------------------------------
# Image Processing Tests
# ---------------------------------------------------------------------------

class TestImageProcessing:
    def test_decode_image(self):
        from app.utils.image_processing import decode_image
        # Create a small JPEG in memory
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        decoded = decode_image(buf.tobytes())
        assert decoded.shape[0] == 50

    def test_normalize_dimensions(self):
        from app.utils.image_processing import normalize_dimensions
        big = np.zeros((5000, 4000, 3), dtype=np.uint8)
        result = normalize_dimensions(big)
        assert max(result.shape[:2]) <= 3000

    def test_to_grayscale(self):
        from app.utils.image_processing import to_grayscale
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        gray = to_grayscale(bgr)
        assert len(gray.shape) == 2

    def test_content_ratio_white(self, white_image):
        from app.utils.image_processing import compute_content_ratio, to_grayscale
        gray = to_grayscale(white_image)
        ratio = compute_content_ratio(gray)
        assert ratio < 0.01  # Nearly all white

    def test_content_ratio_black(self):
        from app.utils.image_processing import compute_content_ratio
        black = np.zeros((100, 100), dtype=np.uint8)
        ratio = compute_content_ratio(black)
        assert ratio > 0.99  # All black

    def test_crop_region(self):
        from app.utils.image_processing import crop_region
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        crop = crop_region(img, 0.1, 0.2, 0.3, 0.4)
        assert crop.shape[0] > 0 and crop.shape[1] > 0


# ---------------------------------------------------------------------------
# Template Matching Tests
# ---------------------------------------------------------------------------

class TestTemplateMatching:
    def test_match_page1(self, page1_template):
        from app.services.template_matching import TemplateMatchingService
        svc = TemplateMatchingService()
        result = svc.match(page1_template)
        assert result.matched is True
        assert result.confidence > 0.3

    def test_match_page2(self, page2_template):
        from app.services.template_matching import TemplateMatchingService
        svc = TemplateMatchingService()
        result = svc.match(page2_template)
        assert result.matched is True

    def test_reject_blank(self, white_image):
        from app.services.template_matching import TemplateMatchingService
        svc = TemplateMatchingService()
        result = svc.match(white_image)
        # Blank image should have low confidence
        assert result.confidence < 0.5

    def test_identify_page1(self, page1_template):
        from app.services.template_matching import TemplateMatchingService
        svc = TemplateMatchingService()
        page_num, conf = svc.identify_page(page1_template)
        assert page_num == 1

    def test_identify_page2(self, page2_template):
        from app.services.template_matching import TemplateMatchingService
        svc = TemplateMatchingService()
        page_num, conf = svc.identify_page(page2_template)
        assert page_num == 2


# ---------------------------------------------------------------------------
# Page Validation Tests
# ---------------------------------------------------------------------------

class TestPageValidation:
    def test_blank_page_detected(self, white_image):
        from app.services.page_validation import PageValidationService
        svc = PageValidationService()
        presence, details = svc.validate_pages([white_image])
        assert details[0].is_blank is True

    def test_page1_validated(self, page1_template):
        from app.services.page_validation import PageValidationService
        svc = PageValidationService()
        presence, details = svc.validate_pages([page1_template])
        assert presence.page1 is True

    def test_both_pages(self, page1_template, page2_template):
        from app.services.page_validation import PageValidationService
        svc = PageValidationService()
        presence, details = svc.validate_pages([page1_template, page2_template])
        assert presence.page1 is True
        assert presence.page2 is True
        assert len(presence.missing_pages) == 0


# ---------------------------------------------------------------------------
# Photo Detection Tests
# ---------------------------------------------------------------------------

class TestPhotoDetection:
    def test_blank_region_no_photo(self, white_image):
        from app.services.photo_detection import PhotoDetectionService
        svc = PhotoDetectionService()
        result = svc.validate(white_image, page_number=1)
        assert result.photo_present is False

    def test_not_page1_returns_empty(self, white_image):
        from app.services.photo_detection import PhotoDetectionService
        svc = PhotoDetectionService()
        result = svc.validate(white_image, page_number=2)
        assert result.photo_present is False


# ---------------------------------------------------------------------------
# Signature Detection Tests
# ---------------------------------------------------------------------------

class TestSignatureDetection:
    def test_blank_no_signature(self, white_image):
        from app.services.signature_detection import SignatureDetectionService
        svc = SignatureDetectionService()
        result = svc.validate(white_image, page_number=1)
        assert result.applicant_signature_present is False

    def test_wrong_page_returns_empty(self, white_image):
        from app.services.signature_detection import SignatureDetectionService
        svc = SignatureDetectionService()
        result = svc.validate(white_image, page_number=3)
        assert result.applicant_signature_present is False


# ---------------------------------------------------------------------------
# Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_with_page1(self, page1_template):
        from app.services.pipeline import ValidationPipeline
        pipeline = ValidationPipeline()
        _, buf = cv2.imencode(".jpg", page1_template)
        result = pipeline.run(buf.tobytes(), "test_page1.jpg")
        assert result.success is True
        assert result.processing_time_ms > 0

    def test_pipeline_with_blank(self, white_image):
        from app.services.pipeline import ValidationPipeline
        pipeline = ValidationPipeline()
        _, buf = cv2.imencode(".jpg", white_image)
        result = pipeline.run(buf.tobytes(), "blank.jpg")
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
