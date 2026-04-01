from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_SYSTEM_PROMPT = (
    "You are AgriVision AgroAdvisor for Indian crop-disease management. "
    "Respond with valid JSON only. Output keys must be exactly: "
    "immediate_action, local_treatment, weather_warning, "
    "organic_treatment, chemical_treatment, recovery_time, preventive_measures. "
    "Guidance must be practical, safe, and location-aware. "
    "If model confidence is below 60 percent, emphasize verification before spray decisions."
)


def _kb_fallback_recommendation(crop: str, disease: str) -> Dict[str, Any]:
    """Generate disease-specific fallback advice from local knowledge base."""
    try:
        from ..llm_validation import knowledge_base as kb
        from ..llm_validation import utils as llm_utils
    except ImportError:
        from llm_validation import knowledge_base as kb
        from llm_validation import utils as llm_utils

    logger = llm_utils.setup_logger(__name__)
    kb_entry = kb.get_disease_context(crop, disease)

    if kb_entry is None:
        logger.warning("Disease not in KB: %s - %s; using generic fallback", crop, disease)
        return {
            "immediate_action": "Isolate affected plants and remove heavily infected leaves today.",
            "local_treatment": "Consult your nearest agricultural extension office for crop-specific treatment advice.",
            "weather_warning": "Avoid spraying during rain or strong winds; prefer early morning applications.",
            "organic_treatment": ["Isolate infected leaves and improve airflow around plants."],
            "chemical_treatment": ["Consult local extension office before selecting fungicide."],
            "recovery_time": "Unknown",
            "preventive_measures": [
                "Inspect nearby leaves daily for spread.",
                "Avoid overhead irrigation late in the day.",
            ],
        }

    organic = kb_entry.get("organic_treatments", [])
    chemical = kb_entry.get("chemical_treatments", [])
    recovery = kb_entry.get("recovery_time_days", 21)

    immediate_action = organic[0] if organic else "Isolate affected plants and remove infected leaves."
    local_treatment = chemical[0] if chemical else "Consult your nearest agricultural extension office."

    return {
        "immediate_action": immediate_action,
        "local_treatment": local_treatment,
        "weather_warning": (
            f"Recovery timeline: ~{recovery} days with proper treatment. "
            "Avoid spraying during rain and high wind."
        ),
        "organic_treatment": organic[:3],
        "chemical_treatment": chemical[:3],
        "recovery_time": f"~{recovery} days",
        "preventive_measures": kb_entry.get("preventive_measures", [])[:4],
    }


def _build_user_prompt(
    crop: str,
    disease: str,
    confidence: float,
    severity_score: int,
    location: str,
    month: str,
) -> str:
    confidence_gate = "LOW" if confidence < 60.0 else "PASS"
    return (
        "CASE INPUT:\n"
        f"- crop: {crop}\n"
        f"- disease: {disease}\n"
        f"- model_confidence_percent: {confidence:.2f}\n"
        f"- severity_score_0_to_100: {severity_score}\n"
        f"- location: {location}\n"
        f"- month: {month}\n"
        f"- confidence_gate_status: {confidence_gate} (threshold 60)\n\n"
        "TASK:\n"
        "1) immediate_action: first thing farmer should do in next 24 hours.\n"
        "2) local_treatment: practical, locally available treatment steps.\n"
        "3) weather_warning: weather-aware caution for spray/treatment timing.\n\n"
        "4) organic_treatment: list 2-4 organic options.\n"
        "5) chemical_treatment: list 1-3 chemical options.\n"
        "6) recovery_time: expected timeline.\n"
        "7) preventive_measures: list 3-5 preventive steps.\n\n"
        "RULES:\n"
        "- Use concise farmer-friendly language.\n"
        "- Do not mention that you are an AI model.\n"
        "- If confidence gate is LOW, ask for clearer image/re-validation before chemical spray.\n"
        "- Return JSON only with required keys."
    )


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _call_groq(client: Groq, user_prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    return {
        "immediate_action": str(payload.get("immediate_action", "")).strip(),
        "local_treatment": str(payload.get("local_treatment", "")).strip(),
        "weather_warning": str(payload.get("weather_warning", "")).strip(),
        "organic_treatment": _as_list(payload.get("organic_treatment")),
        "chemical_treatment": _as_list(payload.get("chemical_treatment")),
        "recovery_time": str(payload.get("recovery_time", "")).strip(),
        "preventive_measures": _as_list(payload.get("preventive_measures")),
    }


def get_recommendation(
    crop: str,
    disease: str,
    confidence: float,
    severity_score: int,
    location: str,
    month: str,
) -> Dict[str, Any]:
    """Get LLM recommendation with disease-specific knowledge-base fallback."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[AgriVision] GROQ_API_KEY not found; using knowledge base fallback.")
        return _kb_fallback_recommendation(crop, disease)

    client = Groq(api_key=api_key)
    user_prompt = _build_user_prompt(crop, disease, confidence, severity_score, location, month)

    for attempt in range(2):
        try:
            return _call_groq(client, user_prompt)
        except json.JSONDecodeError as exc:
            print(f"[AgriVision] Groq JSON parse failed on attempt {attempt + 1}: {exc}")
            if attempt == 1:
                break
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[AgriVision] Groq call failed on attempt {attempt + 1}: {exc}")
            if attempt == 1:
                break

    print("[AgriVision] Using disease-specific knowledge base fallback.")
    return _kb_fallback_recommendation(crop, disease)
