"""
Detailed sub-score diagnostic for template matching.
Shows individual signal scores per form to verify each method contributes.
"""

import io
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
os.environ["FV_LOG_JSON"] = "false"
os.environ["FV_LOG_LEVEL"] = "WARNING"

import cv2
import numpy as np

from app.logging_config import setup_logging
setup_logging()

from app.services.template_matching import TemplateMatchingService
from app.utils.image_processing import preprocess_for_matching

_HEADER_PCT = 0.20
_FOOTER_PCT = 0.08


def main():
    forms_dir = r"C:\Users\Omkark\Downloads\forms"
    form_files = sorted([
        f for f in os.listdir(forms_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    matcher = TemplateMatchingService()
    matcher._ensure_descriptors_loaded()

    print(f"\n{'='*110}")
    print(f"  DETAILED SUB-SCORES — {len(form_files)} forms")
    print(f"{'='*110}")
    print(f"  {'File':<45s} | {'Match':<12s} | {'Header':>7s} | {'Grid':>7s} | {'Edge':>7s} | {'Footer':>7s} | {'Logo':>7s} | {'Fused':>7s}")
    print(f"  {'-'*45}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for fname in form_files:
        fpath = os.path.join(forms_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        page_gray = preprocess_for_matching(img)
        h, w = page_gray.shape[:2]
        page_header = page_gray[0:int(h * _HEADER_PCT), :]
        page_footer = page_gray[int(h * (1.0 - _FOOTER_PCT)):, :]

        best_name = None
        best_fused = 0.0
        best_scores = {}

        for tpl_name, tpl_desc in matcher._descriptors.items():
            scores = {}
            scores["header_orb"] = matcher._header_orb_score(page_header, tpl_desc)
            scores["grid"] = matcher._structural_grid_score(page_gray, tpl_desc)
            scores["edge_spatial"] = matcher._edge_spatial_score(page_gray, tpl_desc)
            scores["footer_orb"] = matcher._footer_orb_score(page_footer, tpl_desc)
            scores["logo"] = matcher._logo_score(img)
            scores["ocr"] = 0.0

            fused = matcher._fuse_scores(scores)
            if fused > best_fused:
                best_fused = fused
                best_name = tpl_name
                best_scores = scores.copy()

        short_name = fname[:42] + "..." if len(fname) > 45 else fname
        print(
            f"  {short_name:<45s} | {best_name or 'None':<12s} | "
            f"{best_scores.get('header_orb', 0):.4f}  | "
            f"{best_scores.get('grid', 0):.4f}  | "
            f"{best_scores.get('edge_spatial', 0):.4f}  | "
            f"{best_scores.get('footer_orb', 0):.4f}  | "
            f"{best_scores.get('logo', 0):.4f}  | "
            f"{best_fused:.4f}"
        )

    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
