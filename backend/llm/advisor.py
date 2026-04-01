from __future__ import annotations

import json
import os
from typing import Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_SYSTEM_PROMPT = (
    "You are an expert agronomist in Karnataka, India with 20 years of experience. "
    "Always respond in valid JSON only."
)


def _default_recommendation() -> Dict[str, str]:
    return {
        "immediate_action": "Isolate affected plants and remove heavily infected leaves today.",
        "local_treatment": "Use locally available neem-based spray and consult nearest agri-input center for crop-specific fungicide.",
        "weather_warning": "Avoid spraying during rain or strong winds; prefer early morning applications.",
    }


    def _kb_fallback_recommendation(crop: str, disease: str) -> Dict[str, str]:
        """
        Generate disease-specific fallback advice from knowledge base.
    
        This replaces hardcoded generic advice with KB-driven recommendations.
        When LLM API is unavailable or fails, use knowledge base instead.
        """
        try:
            from .llm_validation import knowledge_base as kb
            from .llm_validation import utils as llm_utils
        except ImportError:
            from llm_validation import knowledge_base as kb
            from llm_validation import utils as llm_utils
    
        logger = llm_utils.setup_logger(__name__)
    
        # Get disease-specific knowledge base entry
        kb_entry = kb.get_disease_context(crop, disease)
    
        if kb_entry is None:
            # Unknown disease - return generic fallback
            logger.warning(f"Disease not in KB: {crop} - {disease}, using generic fallback")
            return {
                "immediate_action": "Isolate affected plants and remove heavily infected leaves today.",
                "local_treatment": "Consult your nearest agricultural extension office for crop-specific treatment advice.",
                "weather_warning": "Avoid spraying during rain or strong winds; prefer early morning applications.",
            }
    
        # Build KB-based disease-specific advice
        organic = kb_entry.get("organic_treatments", [])
        chemical = kb_entry.get("chemical_treatments", [])
        recovery = kb_entry.get("recovery_time_days", 21)
    
        immediate_action = organic[0] if organic else "Isolate affected plants and remove infected leaves."
        local_treatment = chemical[0] if chemical else "Consult your nearest agricultural extension office."
    
        return {
            "immediate_action": immediate_action,
            "local_treatment": local_treatment,
            "weather_warning": f"Recovery timeline: ~{recovery} days with proper treatment. Avoid spraying during rain.",
        }


def _build_user_prompt(
    crop: str,
    disease: str,
    confidence: float,
    severity_score: int,
    location: str,
    month: str,
) -> str:
    return (
        "Provide practical disease management for this case in Karnataka, India.\\n"
        f"Crop: {crop}\\n"
        f"Disease: {disease}\\n"
        f"Model confidence: {confidence:.2f}%\\n"
        f"Severity score (0-100): {severity_score}\\n"
        f"Exact location: {location}\\n"
        f"Current month: {month}\\n"
        "Give Karnataka-specific, locally available treatments. "
        "Return valid JSON with exactly these keys: "
        "immediate_action, local_treatment, weather_warning."
    )


def _call_groq(client: Groq, user_prompt: str) -> Dict[str, str]:
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    return {
        "immediate_action": str(payload.get("immediate_action", "")).strip(),
        "local_treatment": str(payload.get("local_treatment", "")).strip(),
        "weather_warning": str(payload.get("weather_warning", "")).strip(),
    }


def get_recommendation(
    crop: str,
    disease: str,
    confidence: float,
    severity_score: int,
    location: str,
    month: str,
) -> Dict[str, str]:
        """
        Get LLM-based recommendation with knowledge base fallback.
    
        Primary path: Try Groq API with retries.
        Fallback path: Use disease-specific knowledge base if API unavailable.
        """
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
                    print("[AgriVision] Using disease-specific knowledge base fallback.")
                    break
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[AgriVision] Groq call failed on attempt {attempt + 1}: {exc}")
                if attempt == 1:
                    print("[AgriVision] Using disease-specific knowledge base fallback.")
                    break

        return _kb_fallback_recommendation(crop, disease)
