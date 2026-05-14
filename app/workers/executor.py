"""
Concurrency layer — ThreadPoolExecutor + asyncio bridge.

This module is the SOLE bridge between FastAPI's async event loop and
the CPU-bound validation pipeline.  It:

  1. Maintains a bounded ThreadPoolExecutor
  2. Provides an async wrapper that dispatches work via run_in_executor()
  3. Keeps the event loop 100% free for I/O (accepting new requests)

Architecture (from how_concurrency_can_be_achieved.txt):

  Event Loop Thread:
    Request → read bytes → dispatch to thread pool → await → respond
    (FREE within milliseconds)

  Thread Pool:
    Worker picks up task → decode image → run pipeline → return result
    (Multiple workers run in TRUE parallel for OpenCV C-extension code)

Why threads work for OpenCV:
  OpenCV operations (cvtColor, GaussianBlur, matchTemplate, findContours,
  etc.) are implemented in C/C++ and RELEASE Python's GIL. Multiple
  threads achieve REAL parallelism for the heavy computation.

Why NOT multiprocessing:
  - Each validation is ~1-2s, not minutes
  - Process startup overhead would exceed the benefit
  - IPC serialization of large numpy arrays is expensive
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import ValidationResponse
from app.services.pipeline import ValidationPipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level executor — shared across all requests within a process
# ---------------------------------------------------------------------------
_executor: Optional[ThreadPoolExecutor] = None
_pipeline: Optional[ValidationPipeline] = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create the bounded thread pool executor."""
    global _executor
    if _executor is None:
        settings = get_settings()
        _executor = ThreadPoolExecutor(
            max_workers=settings.max_workers,
            thread_name_prefix="fv-worker",
        )
        logger.info(
            "thread_pool_created",
            max_workers=settings.max_workers,
        )
    return _executor


def get_pipeline() -> ValidationPipeline:
    """Get or create the singleton validation pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ValidationPipeline()
        logger.info("validation_pipeline_created")
    return _pipeline


def shutdown_executor() -> None:
    """Gracefully shut down the thread pool."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        logger.info("thread_pool_shutdown")
        _executor = None


# ---------------------------------------------------------------------------
# Synchronous worker function (runs IN the thread pool)
# ---------------------------------------------------------------------------
def _validate_worker(file_bytes: bytes, filename: str) -> ValidationResponse:
    """
    Worker function executed in a thread-pool thread.

    This is entirely synchronous — no async, no event-loop references.
    Each invocation is independent: its own numpy arrays, its own result.

    Thread safety:
      - file_bytes: immutable bytes, no shared state
      - Pipeline services are stateless (no mutable attrs)
      - Template images are read-only cached data
      - Result dict is constructed fresh per call
    """
    pipeline = get_pipeline()
    return pipeline.run(file_bytes, filename)


# ---------------------------------------------------------------------------
# Async wrapper (called from FastAPI endpoint)
# ---------------------------------------------------------------------------
async def validate_form_async(
    file_bytes: bytes,
    filename: str,
) -> ValidationResponse:
    """
    Async entry point for form validation.

    Dispatches the CPU-bound validation to the thread pool via
    ``asyncio.run_in_executor()``, keeping the event loop free.

    Parameters
    ----------
    file_bytes : bytes
        Raw uploaded file content (already read from UploadFile).
    filename : str
        Original filename for type detection.

    Returns
    -------
    ValidationResponse
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()

    logger.debug("dispatching_to_thread_pool", filename=filename)

    result = await loop.run_in_executor(
        executor,
        _validate_worker,
        file_bytes,
        filename,
    )

    return result
