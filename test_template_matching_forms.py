"""
Test script — validate reworked template matching against real filled forms.

Reads all .jpg files from C:\\Users\\Omkark\\Downloads\\forms and runs
the template matching engine on each one, printing match results.
"""

import io
import os
import sys
import time

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Set environment to disable JSON logging for readable output
os.environ["FV_LOG_JSON"] = "false"
os.environ["FV_LOG_LEVEL"] = "WARNING"  # Suppress noisy logs during test

import cv2
import numpy as np

from app.logging_config import setup_logging
setup_logging()

from app.services.template_matching import TemplateMatchingService


def main():
    forms_dir = r"C:\Users\Omkark\Downloads\forms"

    if not os.path.isdir(forms_dir):
        print(f"ERROR: Forms directory not found: {forms_dir}")
        return

    form_files = sorted([
        f for f in os.listdir(forms_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not form_files:
        print(f"ERROR: No image files found in {forms_dir}")
        return

    print(f"\n{'='*80}")
    print(f"  TEMPLATE MATCHING TEST — {len(form_files)} forms from {forms_dir}")
    print(f"{'='*80}\n")

    matcher = TemplateMatchingService()

    results = []
    total_start = time.perf_counter()

    for i, fname in enumerate(form_files, 1):
        fpath = os.path.join(forms_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [{i:02d}] SKIP — cannot read: {fname}")
            continue

        start = time.perf_counter()
        result = matcher.match(img)
        elapsed_ms = (time.perf_counter() - start) * 1000

        status = "MATCHED" if result.matched else "FAILED"
        icon = "[OK]" if result.matched else "[XX]"

        results.append({
            "file": fname,
            "matched": result.matched,
            "template": result.template_name,
            "confidence": result.confidence,
            "method": result.method,
            "time_ms": elapsed_ms,
        })

        print(
            f"  [{i:02d}] {icon} {status:8s} | "
            f"conf={result.confidence:.4f} | "
            f"tpl={result.template_name or 'None':15s} | "
            f"method={result.method or 'none':15s} | "
            f"time={elapsed_ms:.0f}ms | "
            f"{fname}"
        )

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # Summary
    matched_count = sum(1 for r in results if r["matched"])
    failed_count = sum(1 for r in results if not r["matched"])
    avg_confidence = (
        np.mean([r["confidence"] for r in results if r["matched"]])
        if matched_count else 0.0
    )
    avg_time = np.mean([r["time_ms"] for r in results]) if results else 0.0

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total forms:     {len(results)}")
    print(f"  Matched:         {matched_count}  ({matched_count/max(len(results),1)*100:.0f}%)")
    print(f"  Failed:          {failed_count}  ({failed_count/max(len(results),1)*100:.0f}%)")
    print(f"  Avg confidence:  {avg_confidence:.4f}")
    print(f"  Avg time/form:   {avg_time:.0f}ms")
    print(f"  Total time:      {total_elapsed:.0f}ms")
    print(f"{'='*80}\n")

    if failed_count > 0:
        print("  FAILED forms:")
        for r in results:
            if not r["matched"]:
                print(f"    [XX] {r['file']} -- conf={r['confidence']:.4f}")
        print()


if __name__ == "__main__":
    main()
