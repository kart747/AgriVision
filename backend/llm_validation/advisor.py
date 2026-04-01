"""
LLM-based and fallback recommendation engine for crop disease management.

Generates structured advice for disease treatment, prevention, and recovery.
Can work in two modes: fallback (knowledge base only) or LLM-enhanced.
"""

import json
from typing import Dict, Optional, List, Any

from . import config
from . import utils
from . import schemas
from . import knowledge_base as kb
from . import prompts

logger = utils.setup_logger(__name__)

def _normalize_confidence_value(confidence: Any) -> Optional[float]:
    """Normalize confidence to 0.0..1.0, accepting 0..100 percentages."""
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    if value <= 1:
        return value
    if value <= 100:
        return value / 100.0
    return None


# ============================================================================
# FALLBACK ADVISOR (NO LLM)
# ============================================================================

def generate_fallback_advice(
    prediction_context: Dict,
    kb_entry: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate recommendations based only on local knowledge base.
    
    This is the core fallback mode that works without any LLM API.
    
    Args:
        prediction_context: Dict with crop, disease, confidence, severity, etc.
        kb_entry: Optional pre-loaded knowledge base entry. If None, loads automatically.
    
    Returns:
        Structured recommendation dict (AdvisorOutput format)
    """
    crop = prediction_context.get("crop", "Unknown")
    disease = prediction_context.get("disease", "Unknown")
    severity = prediction_context.get("severity", "Moderate")
    
    # Load knowledge base entry if not provided
    if kb_entry is None:
        kb_entry = kb.get_disease_context(crop, disease)
    
    # Build response
    advice = {
        "source": "fallback",
        "crop": crop,
        "disease": disease,
        "summary": "",
        "organic_treatment": [],
        "chemical_treatment": [],
        "recovery_time": "",
        "preventive_measures": [],
        "warnings": [],
        "notes": []
    }
    
    if kb_entry is None:
        # Unknown disease - provide generic fallback
        logger.warning(f"Disease not in knowledge base: {crop} - {disease}")
        advice["summary"] = (
            f"Disease '{disease}' on {crop} crops detected. "
            "Specific guidance not available in local knowledge base. "
            "Recommend consulting regional agricultural extension office."
        )
        advice["warnings"] = [
            "Unknown disease - recommendations are generic",
            "Contact local agricultural expert for specific guidance"
        ]
        return advice
    
    # Known disease - populate from knowledge base
    advice["summary"] = _build_disease_summary(crop, disease, severity, kb_entry)
    advice["organic_treatment"] = kb_entry.get("organic_treatments", [])[:config.MAX_RECOMMENDATIONS]
    advice["chemical_treatment"] = kb_entry.get("chemical_treatments", [])[:config.MAX_RECOMMENDATIONS]
    advice["recovery_time"] = _format_recovery_time(kb_entry.get("recovery_time_days", 21))
    advice["preventive_measures"] = kb_entry.get("preventive_measures", [])[:config.MAX_RECOMMENDATIONS]
    advice["notes"] = kb_entry.get("notes", [])
    
    # Add severity-based warnings
    if severity.lower() in ["severe", "critical"]:
        advice["warnings"].append(
            f"SEVERE: This disease at '{severity}' level may cause significant crop loss if untreated immediately."
        )
    
    logger.info(f"Generated fallback advice: {crop} - {disease}")
    return advice


# ============================================================================
# LLM-BASED ADVISOR
# ============================================================================

def generate_advice_with_llm(
    prediction_context: Dict,
    kb_entry: Optional[Dict] = None,
    api_key: Optional[str] = None,
    use_fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Generate recommendations using LLM API (Groq free tier).
    
    Falls back to knowledge base if API fails or is unavailable.
    
    Args:
        prediction_context: Dict with crop, disease, etc.
        kb_entry: Optional pre-loaded knowledge base entry
        api_key: Optional API key override
        use_fallback_on_error: If True, use fallback on API failure
    
    Returns:
        Structured recommendation dict with source="llm" or "fallback"
    """
    # Get API key
    if api_key is None:
        api_key = config.GROQ_API_KEY
    
    if not api_key:
        logger.warning("GROQ_API_KEY not set, using fallback advisor")
        return generate_fallback_advice(prediction_context, kb_entry)
    
    # Load KB entry if needed for context
    if kb_entry is None:
        crop = prediction_context.get("crop")
        disease = prediction_context.get("disease")
        kb_entry = kb.get_disease_context(crop, disease)
    
    if kb_entry is None:
        crop = prediction_context.get("crop", "Unknown")
        disease = prediction_context.get("disease", "Unknown")
        logger.info(
            "Disease not in knowledge base for LLM context: %s - %s. "
            "Proceeding with minimal context.",
            crop,
            disease,
        )
        kb_entry = {
            "disease": disease,
            "symptoms": [],
            "organic_treatments": [],
            "chemical_treatments": [],
            "recovery_time_days": 21,
            "preventive_measures": [],
            "notes": [
                "This disease label is not present in local knowledge base; "
                "generate cautious, evidence-based guidance."
            ],
        }
    
    try:
        # Call LLM API
        advice = _call_groq_api(prediction_context, kb_entry, api_key)
        
        if advice and advice.get("source") == "llm":
            logger.info(f"Generated LLM advice for: {prediction_context.get('disease')}")
            return advice
        else:
            logger.warning("LLM returned invalid response, falling back")
            if use_fallback_on_error:
                return generate_fallback_advice(prediction_context, kb_entry)
            return advice
    
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        if use_fallback_on_error:
            logger.info("Falling back to knowledge base advisor")
            return generate_fallback_advice(prediction_context, kb_entry)
        else:
            raise


def _call_groq_api(
    prediction_context: Dict,
    kb_entry: Dict,
    api_key: str
) -> Dict[str, Any]:
    """
    Internal function to call Groq API.
    
    Args:
        prediction_context: Prediction context
        kb_entry: Knowledge base entry
        api_key: Groq API key
    
    Returns:
        LLM response parsed as dict, or empty dict if failed
    """
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq library not installed, install with: pip install groq")
        return {}
    
    try:
        client = Groq(api_key=api_key)
        
        # Build prompts
        system_prompt = prompts.build_system_prompt()
        user_prompt = prompts.build_user_prompt(prediction_context, kb_entry)
        
        # Call API
        message = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=1000
        )
        
        # Extract response
        response_text = message.choices[0].message.content.strip()
        
        # Handle markdown code blocks (some LLMs wrap JSON in ```json ... ```)
        if response_text.startswith("```"):
            # Extract JSON from markdown code blocks
            lines = response_text.split('\n')
            json_lines = []
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                elif in_code:
                    json_lines.append(line)
            response_text = '\n'.join(json_lines)
        
        # Validate and parse JSON
        if not prompts.validate_prompt_response(response_text):
            logger.warning("LLM response validation failed")
            return {}
        
        advice = json.loads(response_text)
        advice["source"] = "llm"

        # Normalize optional fields so downstream validation is deterministic.
        advice.setdefault("warnings", [])
        advice.setdefault("notes", [])
        advice.setdefault("organic_treatment", [])
        advice.setdefault("chemical_treatment", [])
        advice.setdefault("preventive_measures", [])
        advice.setdefault("summary", "")
        advice.setdefault("recovery_time", "Unknown")
        
        return advice
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        return {}
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return {}


# ============================================================================
# MAIN ADVISOR ENTRY POINT
# ============================================================================

def generate_advice(
    prediction_context: Dict,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    force_mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for generating recommendations.
    
    Automatically loads knowledge base and routes to appropriate advisor.
    
    Args:
        prediction_context: Dict with crop, disease, confidence, severity, location, time_context
        use_llm: If True, attempt LLM-based recommendations
        api_key: Optional API key for LLM
        force_mode: Force "fallback" or "llm" mode (overrides use_llm setting)
    
    Returns:
        Structured recommendation dict (AdvisorOutput-compatible)
    
    Raises:
        utils.AdvisorError: If critical parameters missing or processing fails
    """
    # Validate required fields
    if "crop" not in prediction_context or "disease" not in prediction_context:
        logger.error("Missing required fields: crop and/or disease")
        raise utils.AdvisorError("prediction_context must include 'crop' and 'disease'")

    crop = prediction_context.get("crop")
    disease = prediction_context.get("disease")

    if not crop or not disease:
        logger.error("Invalid field values: crop and/or disease")
        return {
            "source": "fallback",
            "crop": str(crop or "Unknown crop"),
            "disease": str(disease or "Unknown disease"),
            "summary": "An error occurred while generating recommendations. Please consult local agricultural experts.",
            "organic_treatment": [],
            "chemical_treatment": [],
            "recovery_time": "Unknown",
            "preventive_measures": [],
            "warnings": ["Invalid prediction context: missing crop or disease value"],
            "notes": []
        }
    
    # Load knowledge base entry
    kb_entry = kb.get_disease_context(crop, disease)
    
    # Determine advisor mode
    if force_mode == "fallback":
        use_llm = False
    elif force_mode == "llm":
        use_llm = True
    elif config.FORCE_FALLBACK:
        use_llm = False
    
    # Generate advice
    try:
        if use_llm:
            advice = generate_advice_with_llm(prediction_context, kb_entry, api_key)
        else:
            advice = generate_fallback_advice(prediction_context, kb_entry)
        
        # Ensure all required fields are present
        _validate_advice_output(advice)
        
        return advice
    
    except Exception as e:
        logger.error(f"Advisor error: {e}")
        # Return minimal fallback if generation fails completely
        return {
            "source": "fallback",
            "crop": crop,
            "disease": disease,
            "summary": "An error occurred while generating recommendations. Please consult local agricultural experts.",
            "organic_treatment": [],
            "chemical_treatment": [],
            "recovery_time": "Unknown",
            "preventive_measures": [],
            "warnings": ["Error during recommendation generation"],
            "notes": []
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _build_disease_summary(crop: str, disease: str, severity: str, kb_entry: Dict) -> str:
    """
    Build a summary description of the disease.
    
    Args:
        crop: Crop name
        disease: Disease name
        severity: Severity level
        kb_entry: Knowledge base entry
    
    Returns:
        Summary string
    """
    base_summary = f"{disease} is a disease affecting {crop} crops."
    
    symptoms = kb_entry.get("symptoms", [])
    if symptoms:
        symptom_str = ", ".join(symptoms[:2])
        base_summary += f" Key symptoms include: {symptom_str}."
    
    if severity.lower() in ["severe", "critical"]:
        base_summary += " This is a severe infection requiring immediate action."
    
    return base_summary


def _format_recovery_time(days: int) -> str:
    """
    Format recovery time as human-readable string.
    
    Args:
        days: Recovery time in days
    
    Returns:
        Formatted string, e.g., "14-21 days"
    """
    if days <= 0:
        return "Variable (disease-dependent)"
    elif days <= 7:
        return f"1-{days} days"
    else:
        # Show range
        lower = max(days - 7, 1)
        return f"{lower}-{days} days"


def _validate_advice_output(advice: Dict[str, Any]) -> bool:
    """
    Validate that advice output has all required fields.
    
    Args:
        advice: Advice dictionary
    
    Returns:
        True if valid, False otherwise
    
    Raises:
        utils.AdvisorError: If validation fails
    """
    required = [
        "source", "crop", "disease", "summary",
        "organic_treatment", "chemical_treatment",
        "recovery_time", "preventive_measures"
    ]
    
    missing = [f for f in required if f not in advice]
    if missing:
        raise utils.AdvisorError(f"Advice missing required fields: {missing}")
    
    return True


def get_advisor_status() -> Dict[str, Any]:
    """
    Get current advisor configuration and status.
    
    Returns:
        Dict with advisor settings
    """
    return {
        "llm_provider": config.LLM_PROVIDER,
        "llm_enabled": bool(config.GROQ_API_KEY),
        "force_fallback": config.FORCE_FALLBACK,
        "max_recommendations": config.MAX_RECOMMENDATIONS,
        "knowledge_base_crops": kb.list_crops(),
    }
