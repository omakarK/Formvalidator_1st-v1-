"""
Concurrent Load Test — 50 simultaneous users uploading forms.

Simulates 50 concurrent users each uploading a random form from
C:\\Users\\Omkark\\Downloads\\forms to the /validate-form endpoint.

Measures: accuracy, response times, throughput, error rates.

Usage:
    1. Start the server:   python run.py
    2. Run this script:    python test_concurrent_load.py
    
    Options:
        --users 50          Number of concurrent users (default: 50)
        --url http://...    Server URL (default: http://localhost:9000)
        --rounds 3          Number of rounds to run (default: 3)
"""

import argparse
import io
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


FORMS_DIR = r"C:\Users\Omkark\Downloads\forms"


@dataclass
class RequestResult:
    """Result of a single validation request."""
    file: str = ""
    status_code: int = 0
    matched: bool = False
    template_name: str = ""
    confidence: float = 0.0
    response_time_ms: float = 0.0
    error: str = ""
    photo_present: bool = False
    signature_present: bool = False


def load_form_files() -> List[str]:
    """Load all form file paths."""
    files = [
        os.path.join(FORMS_DIR, f)
        for f in os.listdir(FORMS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not files:
        print(f"ERROR: No image files found in {FORMS_DIR}")
        sys.exit(1)
    return files


def send_validation_request(file_path: str, base_url: str, timeout: float = 60.0) -> RequestResult:
    """Send a single form validation request."""
    fname = os.path.basename(file_path)
    result = RequestResult(file=fname)

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        start = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base_url}/validate-form",
                files={"file": (fname, file_bytes, "image/jpeg")},
            )
        result.response_time_ms = (time.perf_counter() - start) * 1000
        result.status_code = resp.status_code

        if resp.status_code == 200:
            data = resp.json()
            result.matched = data.get("template_match", {}).get("matched", False)
            result.template_name = data.get("template_matched", "") or ""
            result.confidence = data.get("template_match", {}).get("confidence", 0.0)
            result.photo_present = data.get("photo_present", False)
            result.signature_present = data.get("applicant_signature_present", False)
        else:
            result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"

    except httpx.ConnectError:
        result.status_code = 0
        result.error = "Connection refused - is the server running?"
    except Exception as exc:
        result.status_code = 0
        result.error = str(exc)[:200]

    return result


def run_concurrent_round(
    form_files: List[str],
    num_users: int,
    base_url: str,
    round_num: int,
) -> List[RequestResult]:
    """Run one round of concurrent requests."""
    # Pick random forms for each user
    selected_files = [random.choice(form_files) for _ in range(num_users)]

    print(f"\n  Round {round_num}: Launching {num_users} concurrent requests...")
    results: List[RequestResult] = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = {
            executor.submit(send_validation_request, fpath, base_url): i
            for i, fpath in enumerate(selected_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append(RequestResult(
                    file=os.path.basename(selected_files[idx]),
                    status_code=0,
                    error=str(exc)[:200],
                ))

    total_ms = (time.perf_counter() - start) * 1000
    print(f"  Round {round_num}: Completed in {total_ms:.0f}ms")
    return results


def print_round_results(results: List[RequestResult], round_num: int):
    """Print detailed results for a round."""
    total = len(results)
    successful = [r for r in results if r.status_code == 200]
    matched = [r for r in successful if r.matched]
    failed_http = [r for r in results if r.status_code != 200]
    errors = [r for r in results if r.error]

    times = [r.response_time_ms for r in successful]
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    p50 = sorted(times)[len(times) // 2] if times else 0
    p95 = sorted(times)[int(len(times) * 0.95)] if times else 0

    confidences = [r.confidence for r in matched]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print(f"\n  {'='*70}")
    print(f"  ROUND {round_num} RESULTS")
    print(f"  {'='*70}")
    print(f"  Total requests:        {total}")
    print(f"  HTTP 200 OK:           {len(successful)}/{total}")
    print(f"  Template matched:      {len(matched)}/{len(successful)}  "
          f"({len(matched)/max(len(successful),1)*100:.0f}% accuracy)")
    print(f"  Avg confidence:        {avg_conf:.4f}")
    print(f"  HTTP errors:           {len(failed_http)}")
    print(f"  {'─'*70}")
    print(f"  Response Times:")
    print(f"    Avg:     {avg_time:>8.0f}ms")
    print(f"    Min:     {min_time:>8.0f}ms")
    print(f"    Max:     {max_time:>8.0f}ms")
    print(f"    P50:     {p50:>8.0f}ms")
    print(f"    P95:     {p95:>8.0f}ms")
    print(f"  Throughput:            {len(successful)/max(max_time/1000, 0.001):.1f} req/sec (wall)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors[:5]:
            print(f"    - {r.file}: {r.error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")

    # Show per-template breakdown
    tpl_counts = {}
    for r in matched:
        tpl_counts[r.template_name] = tpl_counts.get(r.template_name, 0) + 1
    if tpl_counts:
        print(f"\n  Template distribution:")
        for tpl, count in sorted(tpl_counts.items()):
            print(f"    {tpl}: {count} forms")

    print(f"  {'='*70}")
    return {
        "total": total,
        "successful": len(successful),
        "matched": len(matched),
        "accuracy": len(matched) / max(len(successful), 1) * 100,
        "avg_time": avg_time,
        "p95_time": p95,
        "avg_conf": avg_conf,
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrent Form Validation Load Test")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users (default: 50)")
    parser.add_argument("--url", default="http://localhost:9000", help="Server URL")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    args = parser.parse_args()

    form_files = load_form_files()

    print(f"\n{'#'*70}")
    print(f"  CONCURRENT LOAD TEST")
    print(f"{'#'*70}")
    print(f"  Server:          {args.url}")
    print(f"  Concurrent users: {args.users}")
    print(f"  Rounds:          {args.rounds}")
    print(f"  Forms available: {len(form_files)}")
    print(f"  Forms directory: {FORMS_DIR}")
    print(f"{'#'*70}")

    # Quick health check
    try:
        with httpx.Client(timeout=5) as client:
            health = client.get(f"{args.url}/health")
            if health.status_code == 200:
                hdata = health.json()
                print(f"\n  Server healthy: status={hdata['status']}, "
                      f"workers={hdata.get('worker_pool_size', '?')}, "
                      f"templates={hdata.get('templates_loaded', '?')}")
            else:
                print(f"\n  WARNING: Health check returned {health.status_code}")
    except httpx.ConnectError:
        print(f"\n  ERROR: Cannot connect to {args.url}")
        print(f"  Start the server first: python run.py")
        sys.exit(1)

    # Run rounds
    all_stats = []
    for round_num in range(1, args.rounds + 1):
        results = run_concurrent_round(form_files, args.users, args.url, round_num)
        stats = print_round_results(results, round_num)
        all_stats.append(stats)

    # Final summary across all rounds
    print(f"\n{'#'*70}")
    print(f"  FINAL SUMMARY (across {args.rounds} rounds)")
    print(f"{'#'*70}")
    total_requests = sum(s["total"] for s in all_stats)
    total_matched = sum(s["matched"] for s in all_stats)
    total_successful = sum(s["successful"] for s in all_stats)
    avg_accuracy = sum(s["accuracy"] for s in all_stats) / len(all_stats)
    avg_time = sum(s["avg_time"] for s in all_stats) / len(all_stats)
    avg_p95 = sum(s["p95_time"] for s in all_stats) / len(all_stats)
    avg_conf = sum(s["avg_conf"] for s in all_stats) / len(all_stats)

    print(f"  Total requests sent:   {total_requests}")
    print(f"  Total successful:      {total_successful}")
    print(f"  Total matched:         {total_matched}")
    print(f"  Overall accuracy:      {avg_accuracy:.1f}%")
    print(f"  Avg confidence:        {avg_conf:.4f}")
    print(f"  Avg response time:     {avg_time:.0f}ms")
    print(f"  Avg P95 response time: {avg_p95:.0f}ms")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
