from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent


def _bootstrap_import_paths() -> None:
    env_pythonpath = os.getenv("PYTHONPATH", "")
    if env_pythonpath:
        for item in env_pythonpath.split(os.pathsep):
            candidate = item.strip()
            if candidate and candidate not in sys.path:
                sys.path.insert(0, candidate)

    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_bootstrap_import_paths()

try:
    from llm_validation.advisor import generate_advice
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "llm_validation":
        raise
    from AgriVision.backend.llm_validation.advisor import generate_advice


def _default_recommendation() -> Dict[str, str]:
    return {
        "immediate_action": "Isolate affected plants and remove heavily infected leaves today.",
        "local_treatment": "Use locally available neem-based spray and consult nearest agri-input center for crop-specific fungicide.",
        "weather_warning": "Avoid spraying during rain or strong winds; prefer early morning applications.",
    }


def _normalize_disease_for_kb(crop: str, disease: str) -> str:
    """Map backend disease names to llm_validation knowledge-base labels."""
    c = (crop or "").strip().lower()
    d = (disease or "").strip().lower()

    aliases = {
        ("tomato", "tomato yellow leaf curl virus"): "Tomato Yellow Leaf Curl Virus",
        ("tomato", "early blight"): "Early Blight (Tomato)",
        ("tomato", "late blight"): "Late Blight (Tomato)",
        ("apple", "apple scab"): "Apple Scab",
        ("apple", "powdery mildew"): "Powdery Mildew (Apple)",
        ("grape", "powdery mildew"): "Powdery Mildew (Grape)",
        ("grape", "downy mildew"): "Downy Mildew (Grape)",
        ("grape", "black rot"): "Black Rot (Grape)",
    }

    if (c, d) in aliases:
        return aliases[(c, d)]

    # Keep existing disease if no alias was found.
    return disease


def _severity_from_score(severity_score: int) -> str:
    if severity_score >= 70:
        return "Severe"
    if severity_score >= 40:
        return "Moderate"
    return "Mild"


def _to_legacy_recommendation(advice: Dict[str, Any]) -> Dict[str, str]:
    """Convert structured advisor output to existing frontend contract."""
    organic = advice.get("organic_treatment", []) or []
    chemical = advice.get("chemical_treatment", []) or []
    preventive = advice.get("preventive_measures", []) or []
    warnings = advice.get("warnings", []) or []
    notes = advice.get("notes", []) or []

    immediate_action = organic[0] if organic else (chemical[0] if chemical else "Monitor crop and consult local agri expert.")

    local_treatment_parts = []
    if organic:
        local_treatment_parts.append("Organic: " + "; ".join(organic[:2]))
    if chemical:
        local_treatment_parts.append("Chemical: " + "; ".join(chemical[:2]))
    if preventive:
        local_treatment_parts.append("Prevention: " + "; ".join(preventive[:2]))

    local_treatment = " | ".join(local_treatment_parts) if local_treatment_parts else advice.get("summary", "")
    weather_warning = warnings[0] if warnings else (notes[0] if notes else "Avoid spraying during rain or strong winds.")

    return {
        "immediate_action": str(immediate_action).strip(),
        "local_treatment": str(local_treatment).strip(),
        "weather_warning": str(weather_warning).strip(),
    }


def get_recommendation(
    crop: str,
    disease: str,
    confidence: float,
    severity_score: int,
    location: str,
    month: str,
) -> Dict[str, str]:
    # Existing backend sends confidence as percentage; normalize to 0..1 for llm_validation.
    normalized_confidence = float(confidence) / 100.0 if confidence > 1 else float(confidence)
    mapped_disease = _normalize_disease_for_kb(crop, disease)

    if "healthy" in (mapped_disease or "").lower():
        return {
            "immediate_action": "No disease detected. Continue regular crop monitoring.",
            "local_treatment": "Maintain balanced irrigation, nutrition, and field hygiene to keep plants healthy.",
            "weather_warning": "Watch humidity and rainfall changes; they can increase disease risk.",
        }

    prediction_context = {
        "crop": crop,
        "disease": mapped_disease,
        "confidence": normalized_confidence,
        "severity": _severity_from_score(int(severity_score)),
        "location": location,
        "time_context": month,
    }

    try:
        advice = generate_advice(prediction_context, use_llm=True)
        if not isinstance(advice, dict):
            raise ValueError("llm_validation returned non-dict advice payload")

        required_keys = {"summary", "organic_treatment", "chemical_treatment", "preventive_measures"}
        if not required_keys.issubset(set(advice.keys())):
            missing_keys = sorted(required_keys.difference(set(advice.keys())))
            raise ValueError(f"llm_validation advice missing keys: {missing_keys}")

        return _to_legacy_recommendation(advice)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[AgriVision] llm_validation adapter failed: {exc}")
        return _default_recommendation()
