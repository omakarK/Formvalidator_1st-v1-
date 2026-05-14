"""
PDF-to-images converter using PyMuPDF.

Converts each page of an uploaded PDF into a high-resolution BGR numpy array.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def pdf_to_images(pdf_bytes: bytes) -> List[np.ndarray]:
    """
    Convert a PDF to a list of BGR numpy arrays (one per page).

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.

    Returns
    -------
    list[np.ndarray]
        List of BGR images, one per page.

    Raises
    ------
    ValueError
        If the PDF has no pages or cannot be opened.
    """
    import fitz  # PyMuPDF

    settings = get_settings()
    dpi = settings.target_dpi

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    if len(doc) == 0:
        raise ValueError("PDF contains no pages")

    images: List[np.ndarray] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        # PyMuPDF outputs RGB; convert to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

        logger.debug(
            "pdf_page_rendered",
            page=page_num + 1,
            total=len(doc),
            size=f"{pix.width}x{pix.height}",
        )

    doc.close()
    logger.info("pdf_converted", total_pages=len(images))
    return images
