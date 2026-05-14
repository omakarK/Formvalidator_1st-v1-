"""
Template metadata configuration.

Defines the complete structure for each known form template including:
  - Page regions (photo, signature, header)
  - OCR anchor texts
  - Expected structural elements

This file is the single source of truth for template coordinates.
All services read from here via the Settings object.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# MKCL Application Form — Template Metadata
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, dict] = {
    "mkcl_page1": {
        "display_name": "MKCL Application Form — Page 1",
        "filename": "page1-base.jpg",
        "page_number": 1,

        # Header region (top portion with "Application Form" title)
        "header_region": {
            "x_ratio": 0.0,
            "y_ratio": 0.0,
            "w_ratio": 1.0,
            "h_ratio": 0.08,
        },

        # Logo region (MKCL logo, top-left)
        "logo_region": {
            "x_ratio": 0.02,
            "y_ratio": 0.01,
            "w_ratio": 0.12,
            "h_ratio": 0.06,
        },

        # Passport photo region (top-right area)
        "photo_region": {
            "x_ratio": 0.78,
            "y_ratio": 0.12,
            "w_ratio": 0.16,
            "h_ratio": 0.12,
        },

        # Applicant signature ONLY (in the declaration section)
        "applicant_signature_region": {
            "x_ratio": 0.55,
            "y_ratio": 0.885,
            "w_ratio": 0.18,
            "h_ratio": 0.035,
        },

        # Parent/Guardian signature (DO NOT validate — for reference only)
        "parent_signature_region": {
            "x_ratio": 0.75,
            "y_ratio": 0.885,
            "w_ratio": 0.18,
            "h_ratio": 0.035,
        },

        # Form number region
        "form_number_region": {
            "x_ratio": 0.72,
            "y_ratio": 0.02,
            "w_ratio": 0.22,
            "h_ratio": 0.03,
        },

        # OCR anchors for template verification
        "ocr_anchors": [
            "Application Form",
            "To be filled in by the Applicant only",
            "Signature of Applicant",
            "Page 1 of 2",
        ],

        # Structural sections (for completeness validation)
        "sections": [
            "personal_info",
            "address",
            "educational_qualification",
            "current_profile",
            "declaration",
            "alc_section",
        ],
    },

    "mkcl_page2": {
        "display_name": "MKCL Application Form — Page 2",
        "filename": "page2-base.jpg",
        "page_number": 2,

        # Header region
        "header_region": {
            "x_ratio": 0.0,
            "y_ratio": 0.0,
            "w_ratio": 1.0,
            "h_ratio": 0.06,
        },

        # Logo region (MKCL logo, top-left)
        "logo_region": {
            "x_ratio": 0.02,
            "y_ratio": 0.01,
            "w_ratio": 0.10,
            "h_ratio": 0.05,
        },

        # No passport photo on page 2
        "photo_region": None,

        # Applicant signature ONLY (bottom of page 2)
        "applicant_signature_region": {
            "x_ratio": 0.45,
            "y_ratio": 0.91,
            "w_ratio": 0.20,
            "h_ratio": 0.04,
        },

        # Parent/Guardian signature (DO NOT validate)
        "parent_signature_region": {
            "x_ratio": 0.68,
            "y_ratio": 0.91,
            "w_ratio": 0.20,
            "h_ratio": 0.04,
        },

        # OCR anchors
        "ocr_anchors": [
            "Application Form",
            "Signature of Applicant",
            "Page 2 of 2",
        ],

        # Structural sections
        "sections": [
            "full_name",
            "school_board",
            "medium_of_school",
            "stream",
            "school_college_name",
            "travel_details",
            "career_choice",
            "social_media",
            "source_of_admission",
            "declaration",
        ],
    },
}


def get_template_metadata(template_name: str) -> dict:
    """Get metadata for a specific template."""
    return TEMPLATES.get(template_name, {})


def get_all_template_names() -> List[str]:
    """Get all registered template names."""
    return list(TEMPLATES.keys())
