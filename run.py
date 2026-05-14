"""
Production server launcher.

Usage:
    python run.py                        # Development (1 worker)
    python run.py --workers 4            # Production (4 processes)
    python run.py --workers 4 --port 8000
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="AI Form Validator Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9000, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers (use 4 for production)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    import uvicorn

    print(f"Starting AI Form Validator on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Thread pool per worker: {min(os.cpu_count() or 4, 8)}")
    print(f"Max parallel validations: {args.workers * min(os.cpu_count() or 4, 8)}")
    print()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
