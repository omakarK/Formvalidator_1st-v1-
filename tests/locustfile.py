"""
Locust load testing configuration.

Run with:
    locust -f tests/locustfile.py --host http://localhost:9000
    
Then open http://localhost:8089 to configure and start the test.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from locust import HttpUser, task, between


def _make_test_image() -> bytes:
    """Generate a synthetic form image for load testing."""
    img = np.ones((1200, 900, 3), dtype=np.uint8) * 235
    cv2.rectangle(img, (50, 20), (850, 80), (200, 200, 200), -1)
    cv2.putText(img, "Application Form", (200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    for y in range(120, 1100, 80):
        cv2.rectangle(img, (50, y), (850, y + 30), (180, 180, 180), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# Pre-generate to avoid per-request overhead
_TEST_IMAGE = _make_test_image()


class FormValidatorUser(HttpUser):
    """Simulated user uploading forms for validation."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def validate_form(self):
        """Upload a form for validation."""
        self.client.post(
            "/validate-form",
            files={"file": ("test.jpg", io.BytesIO(_TEST_IMAGE), "image/jpeg")},
        )

    @task(1)
    def health_check(self):
        """Hit the health endpoint."""
        self.client.get("/health")
