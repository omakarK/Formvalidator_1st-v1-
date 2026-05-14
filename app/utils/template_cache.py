"""
Thread-safe template cache.

Templates (base page images, logos) are loaded from disk ONCE and cached
in memory.  Double-checked locking ensures thread safety during the
initial load while all subsequent reads are lock-free.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache + lock
# ---------------------------------------------------------------------------
_cache: Dict[str, np.ndarray] = {}
_lock = threading.Lock()


def cached_imread(
    filename: str,
    *,
    flags: int = cv2.IMREAD_COLOR,
    directory: Optional[Path] = None,
) -> np.ndarray:
    """
    Read an image from the templates directory, caching the result.

    Parameters
    ----------
    filename : str
        File name inside the templates directory (e.g. ``page1-base.jpg``).
    flags : int
        OpenCV imread flags (default ``IMREAD_COLOR``).
    directory : Path, optional
        Override for the templates directory.

    Returns
    -------
    np.ndarray
        The loaded image (BGR by default).

    Raises
    ------
    FileNotFoundError
        If the template file does not exist on disk.
    ValueError
        If OpenCV cannot decode the file.
    """
    cache_key = f"{filename}:{flags}"

    # Fast path — lock-free read
    if cache_key in _cache:
        return _cache[cache_key]

    # Slow path — first access
    with _lock:
        # Double-check after acquiring lock
        if cache_key in _cache:
            return _cache[cache_key]

        settings = get_settings()
        img_dir = directory or settings.templates_dir
        path = img_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")

        img = cv2.imread(str(path), flags)
        if img is None:
            raise ValueError(f"OpenCV failed to decode: {path}")

        _cache[cache_key] = img
        logger.info("template_cached", file=filename, shape=img.shape)

    return _cache[cache_key]


def cached_imread_gray(filename: str, *, directory: Optional[Path] = None) -> np.ndarray:
    """Convenience wrapper — load template as grayscale."""
    return cached_imread(filename, flags=cv2.IMREAD_GRAYSCALE, directory=directory)


def preload_templates() -> None:
    """Eagerly load all templates at startup to avoid cold-start latency."""
    settings = get_settings()
    try:
        cached_imread(settings.page1_template)
        cached_imread(settings.page2_template)
        cached_imread(settings.mkcl_logo)
        cached_imread_gray(settings.page1_template)
        cached_imread_gray(settings.page2_template)
        cached_imread_gray(settings.mkcl_logo)
        logger.info("all_templates_preloaded")
    except Exception as exc:
        logger.error("template_preload_failed", error=str(exc))
        raise


def clear_cache() -> None:
    """Clear the template cache (useful in tests)."""
    with _lock:
        _cache.clear()
