"""
Concurrency & Load Testing Script.

Simulates concurrent users uploading forms simultaneously.
Tests thread-pool behavior under 100, 500, and 1000 concurrent loads.

Usage:
    python tests/test_concurrency.py --users 100 --url http://localhost:9000
    python tests/test_concurrency.py --users 500
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_test_image() -> bytes:
    """Create a realistic-ish test image (grayscale form with content)."""
    img = np.ones((1200, 900, 3), dtype=np.uint8) * 240
    # Add some text-like rectangles
    for y in range(100, 1100, 80):
        cv2.rectangle(img, (50, y), (850, y + 30), (180, 180, 180), 1)
        cv2.rectangle(img, (60, y + 5), (300, y + 25), (100, 100, 100), -1)
    # Add a header area
    cv2.rectangle(img, (50, 20), (850, 80), (200, 200, 200), -1)
    cv2.putText(img, "Application Form", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def create_real_test_image() -> bytes:
    """Load actual page1 template if available, else use synthetic."""
    template_path = Path(__file__).resolve().parent.parent / "templates" / "page1-base.jpg"
    if template_path.exists():
        with open(template_path, "rb") as f:
            return f.read()
    return create_test_image()


def send_request(url: str, file_bytes: bytes, request_id: int) -> dict:
    """Send a single validation request and return timing info."""
    import requests

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/validate-form",
            files={"file": (f"test_{request_id}.jpg", io.BytesIO(file_bytes), "image/jpeg")},
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "request_id": request_id,
            "status_code": resp.status_code,
            "success": resp.json().get("success", False) if resp.status_code == 200 else False,
            "elapsed_ms": round(elapsed_ms, 2),
            "server_time_ms": resp.json().get("processing_time_ms", 0) if resp.status_code == 200 else 0,
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "request_id": request_id,
            "status_code": 0,
            "success": False,
            "elapsed_ms": round(elapsed_ms, 2),
            "server_time_ms": 0,
            "error": str(exc),
        }


def run_load_test(
    url: str,
    num_users: int,
    use_real_image: bool = False,
) -> None:
    """Run a concurrent load test."""
    print(f"\n{'=' * 70}")
    print(f"  LOAD TEST: {num_users} concurrent users → {url}")
    print(f"{'=' * 70}")

    # Prepare test data
    print("Preparing test image...")
    if use_real_image:
        file_bytes = create_real_test_image()
    else:
        file_bytes = create_test_image()
    print(f"Test image size: {len(file_bytes) / 1024:.1f} KB")

    # Execute concurrent requests
    print(f"\nLaunching {num_users} concurrent requests...")
    start_total = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=min(num_users, 200)) as executor:
        futures = {
            executor.submit(send_request, url, file_bytes, i): i
            for i in range(num_users)
        }

        for future in as_completed(futures):
            results.append(future.result())
            done = len(results)
            if done % 50 == 0 or done == num_users:
                print(f"  Completed: {done}/{num_users}")

    total_time = time.perf_counter() - start_total

    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    response_times = [r["elapsed_ms"] for r in results]
    server_times = [r["server_time_ms"] for r in successful if r["server_time_ms"] > 0]

    print(f"\n{'─' * 70}")
    print(f"  RESULTS")
    print(f"{'─' * 70}")
    print(f"  Total requests:     {num_users}")
    print(f"  Successful:         {len(successful)} ({len(successful)/num_users*100:.1f}%)")
    print(f"  Failed:             {len(failed)}")
    print(f"  Total wall time:    {total_time:.2f}s")
    print(f"  Throughput:         {num_users / total_time:.1f} req/s")
    print()

    if response_times:
        print(f"  Response times (client-side):")
        print(f"    Min:    {min(response_times):.0f} ms")
        print(f"    Max:    {max(response_times):.0f} ms")
        print(f"    Mean:   {statistics.mean(response_times):.0f} ms")
        print(f"    Median: {statistics.median(response_times):.0f} ms")
        print(f"    P95:    {sorted(response_times)[int(len(response_times)*0.95)]:.0f} ms")
        print(f"    P99:    {sorted(response_times)[int(len(response_times)*0.99)]:.0f} ms")

    if server_times:
        print(f"\n  Server processing times:")
        print(f"    Min:    {min(server_times):.0f} ms")
        print(f"    Max:    {max(server_times):.0f} ms")
        print(f"    Mean:   {statistics.mean(server_times):.0f} ms")
        print(f"    Median: {statistics.median(server_times):.0f} ms")

    if failed:
        print(f"\n  Errors:")
        error_types = {}
        for r in failed:
            err = r.get("error", "Unknown")
            error_types[err] = error_types.get(err, 0) + 1
        for err, count in error_types.items():
            print(f"    {err}: {count}x")

    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Form Validator Load Test")
    parser.add_argument("--url", default="http://localhost:9000", help="Server URL")
    parser.add_argument("--users", type=int, default=100, help="Number of concurrent users")
    parser.add_argument("--real", action="store_true", help="Use real template image")
    parser.add_argument("--tiers", action="store_true", help="Run 100, 500, 1000 tier test")
    args = parser.parse_args()

    if args.tiers:
        for n in [100, 500, 1000]:
            run_load_test(args.url, n, use_real_image=args.real)
    else:
        run_load_test(args.url, args.users, use_real_image=args.real)


if __name__ == "__main__":
    main()
