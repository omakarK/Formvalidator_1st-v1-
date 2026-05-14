"""
API-level tests using FastAPI TestClient.

Tests the HTTP endpoints for correct status codes, response shapes,
and error handling.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"

    def test_health_has_worker_info(self):
        resp = client.get("/health")
        data = resp.json()
        assert "worker_pool_size" in data
        assert data["worker_pool_size"] > 0


class TestValidateFormEndpoint:
    def test_no_file_returns_422(self):
        resp = client.post("/validate-form")
        assert resp.status_code == 422

    def test_empty_file_returns_400(self):
        resp = client.post(
            "/validate-form",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_unsupported_extension_returns_400(self):
        resp = client.post(
            "/validate-form",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_valid_image_returns_200(self):
        # Create a simple test image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 200
        _, buf = cv2.imencode(".jpg", img)
        file_bytes = buf.tobytes()

        resp = client.post(
            "/validate-form",
            files={"file": ("test_form.jpg", io.BytesIO(file_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0

    def test_valid_image_response_shape(self):
        img = np.ones((600, 800, 3), dtype=np.uint8) * 128
        _, buf = cv2.imencode(".jpg", img)

        resp = client.post(
            "/validate-form",
            files={"file": ("form.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        )
        data = resp.json()
        assert "template_match" in data
        assert "pages" in data
        assert "photo_present" in data
        assert "applicant_signature_present" in data
        assert "confidence" in data

    def test_real_page1_template(self):
        """Test with actual page1 template if available."""
        from app.config import get_settings
        settings = get_settings()
        path = settings.templates_dir / settings.page1_template
        if not path.exists():
            pytest.skip("Page 1 template not found")

        with open(path, "rb") as f:
            resp = client.post(
                "/validate-form",
                files={"file": ("page1.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_timing_header_present(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        _, buf = cv2.imencode(".jpg", img)

        resp = client.post(
            "/validate-form",
            files={"file": ("t.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert "x-processing-time-ms" in resp.headers


class TestDebugEndpoint:
    def test_debug_returns_debug_info(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        _, buf = cv2.imencode(".jpg", img)

        resp = client.post(
            "/validate-form/debug",
            files={"file": ("t.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("debug") is not None
        assert "photo_region_page1" in data["debug"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
