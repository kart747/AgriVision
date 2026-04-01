"""
Prompt templates and builders for LLM-based recommendations.

Separates prompt engineering logic from advisor code for clarity and iteration.
Prompts can be easily modified or versioned without touching the main module.
"""

from typing import Dict, Optional
from . import utils

logger = utils.setup_logger(__name__)


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

def build_system_prompt() -> str:
    """
    Build the system prompt for the LLM advisor.
    
    Defines the role, constraints, and output format expectations.
    
    Returns:
        System prompt string
    """
    prompt = """You are AgriVision AgroAdvisor, an expert agronomy assistant for crop disease recovery recommendations.

Mission:
- Convert model prediction context into practical, region-aware farmer guidance.
- Use supplied crop, disease, confidence, severity, location, and time context.
- Ground recommendations in the provided knowledge-base context.

Output contract (MANDATORY):
- Return ONLY valid JSON. No markdown, no code fences, no commentary.
- Return exactly one JSON object with keys:
    source, crop, disease, summary, organic_treatment, chemical_treatment,
    recovery_time, preventive_measures, warnings, notes, estimated_cost
- Set source = "llm".
- Keep list fields as arrays of short strings.

Quality rules:
- Be specific, actionable, and farmer-friendly.
- Prefer locally feasible actions and sequence advice by urgency.
- Include safe, standard chemical guidance only when relevant.
- If dosage is uncertain, do not guess exact numeric dose; advise label/local extension verification.
- If confidence is low (<0.60), include a warning to re-capture a clearer image and verify before spraying.
- If severity is Severe/Critical, prioritize immediate containment and crop-loss warning.
- Respect regional context (location + season/time) for timing and prevention.
- If live weather indicates strong wind or high heat, explicitly warn against spraying at that time.
- Use the supplied market pricing context to estimate treatment cost.
- Do not invent diseases, chemicals, or regulations.

Formatting limits:
- summary: 1-2 sentences.
- organic_treatment: 2-5 items.
- chemical_treatment: 1-4 items (or empty array if not needed).
- preventive_measures: 3-6 items.
- warnings: 0-4 items.
- notes: 0-4 items.
- estimated_cost: one short string.
"""
    return prompt.strip()


# ============================================================================
# USER PROMPT BUILDERS
# ============================================================================

def build_user_prompt(prediction_context: Dict, kb_entry: Dict) -> str:
    """
    Build the user-facing prompt with context-aware disease information.
    
    Args:
        prediction_context: Dict with keys: crop, disease, confidence, severity, location, time_context
        kb_entry: Dict from knowledge base with disease info
    
    Returns:
        User prompt string
    """
    crop = prediction_context.get("crop", "Unknown")
    disease = prediction_context.get("disease", "Unknown")
    confidence = prediction_context.get("confidence", 0.0)
    severity = prediction_context.get("severity", "Unknown")
    location = prediction_context.get("location", "Unknown")
    time_context = prediction_context.get("time_context", "Unknown")
    live_weather = prediction_context.get("live_weather", "Weather unavailable")
    
    # Knowledge base info
    symptoms = kb_entry.get("symptoms", [])
    notes = kb_entry.get("notes", [])
    recovery_days = kb_entry.get("recovery_time_days", 21)
    
    confidence_gate = "LOW" if float(confidence) < 0.60 else "PASS"

    # Build structured prompt with deterministic sections for better judge consistency
    lines = [
        "TASK: Generate crop-disease recovery recommendations for this case.",
        "",
        "PREDICTION_CONTEXT:",
        f"- crop: {crop}",
        f"- disease: {disease}",
        f"- confidence: {confidence:.1%}",
        f"- severity: {severity}",
        f"- location: {location}",
        f"- time_context: {time_context}",
        f"- live_weather: {live_weather}",
        f"- confidence_gate_status: {confidence_gate} (threshold 0.60)",
        "",
        "KNOWLEDGE_BASE_SIGNALS:",
        "- symptoms:",
    ]
    
    for symptom in symptoms[:5]:
        lines.append(f"  - {symptom}")

    if not symptoms:
        lines.append("  - No symptom data available")
    
    if notes:
        lines.append("- notes:")
        for note in notes[:3]:
            lines.append(f"  - {note}")
    else:
        lines.append("- notes:")
        lines.append("  - No additional notes available")

    lines.extend([
        "- market_pricing_context:",
        "  - Neem-based spray (~₹250/L)",
        "  - Copper Fungicide (~₹400/kg)",
        "  - Generic Fungicide (~₹300/L)",
    ])
    
    lines.extend([
        "",
        "RESPONSE_REQUIREMENTS:",
        "- Return JSON only with required keys from system prompt.",
        f"- recovery_time should align with approx {recovery_days} days baseline.",
        "- Include immediate first-step actions in organic_treatment[0] when possible.",
        "- Keep language simple for farmers while remaining technically correct.",
        "- If confidence_gate_status is LOW, include a warning about re-capture/verification before chemical spray.",
        "- Include a short estimated_cost string based on the likely treatment category.",
    ])
    
    return "\n".join(lines)


# ============================================================================
# EXAMPLE PROMPT-RESPONSE PAIRS
# ============================================================================

def get_example_prompt_response() -> Dict:
    """
    Return an example prompt-response pair for reference.
    
    Useful for testing and documentation.
    
    Returns:
        Dict with keys: system, user_input, expected_output
    """
    return {
        "system": build_system_prompt(),
        "user_input": """Disease Detection Report:
Crop: Tomato
Disease: Tomato Yellow Leaf Curl Virus
Model Confidence: 92.0%
Severity: Moderate
Location: Mangalore, Karnataka, India
Time Context: Early Morning

Known Symptoms:
  - Yellowing of leaves
  - Leaf curling
  - Stunted growth
  - Reduced fruit production

Additional Context:
  - TYLCV is transmitted by whitefly vectors
  - Viral infections are systemic and irreversible

Provide recommendations for managing this disease on Tomato crops.
Consider the Moderate severity and urgency (expected recovery: ~21 days).
Provide both organic and chemical options suitable for small-scale farming.""",
        "expected_output": {
            "source": "llm",
            "crop": "Tomato",
            "disease": "Tomato Yellow Leaf Curl Virus",
            "summary": "TYLCV is a critical viral disease affecting tomato crops. Whitefly vectors transmit the virus. Early detection and management are essential.",
            "estimated_cost": "Approx. ₹250/L",
            "organic_treatment": [
                "Remove infected plants immediately to prevent spread",
                "Spray neem oil (5% solution) on plant surfaces every 7 days",
                "Install yellow sticky traps to monitor and reduce whitefly population",
                "Encourage natural predators like parasitoid wasps"
            ],
            "chemical_treatment": [
                "Spray imidacloprid (0.005%) at 10-day intervals targeting whitefly",
                "Use thiamethoxam (0.01%) for rapid vector control",
                "Apply acephate (0.1%) if whitefly infestation is severe",
                "Rotate chemicals to prevent resistance development"
            ],
            "recovery_time": "14-21 days with aggressive treatment; 30+ days without intervention",
            "preventive_measures": [
                "Use disease-resistant tomato varieties when available",
                "Control weeds that harbor whitefly populations",
                "Maintain field sanitation and remove plant debris",
                "Use reflective mulches to confuse insects"
            ],
            "warnings": [
                "This disease may cause permanent crop loss if untreated",
                "Whitefly populations spread rapidly in warm weather (20-28°C)"
            ],
            "notes": [
                "Focus treatment on whitefly vector control, not the virus itself",
                "Infected plants may not recover; removal may be necessary",
                "Early detection critical for disease management"
            ]
        }
    }


# ============================================================================
# PROMPT VALIDATION
# ============================================================================

def validate_prompt_response(response_str: str) -> bool:
    """
    Validate that LLM response is valid JSON and has required fields.
    
    Args:
        response_str: Response string from LLM
    
    Returns:
        True if valid, False otherwise
    """
    import json
    
    try:
        data = json.loads(response_str)
        required_fields = [
            "source", "crop", "disease", "summary",
            "organic_treatment", "chemical_treatment",
            "recovery_time", "preventive_measures",
            "warnings", "notes", "estimated_cost"
        ]
        return all(field in data for field in required_fields)
    except (json.JSONDecodeError, TypeError):
        return False


# ============================================================================
# PROMPT STATISTICS & METADATA
# ============================================================================

def get_prompt_metadata() -> Dict:
    """
    Return metadata about prompts for documentation.
    
    Returns:
        Dict with prompt information
    """
    return {
        "llm_provider": "Groq (free tier)",
        "model": "llama-3.3-70b-versatile",
        "system_prompt_version": "2.0",
        "user_prompt_template_version": "2.0",
        "required_input_fields": [
            "crop", "disease", "confidence", "severity",
            "location", "time_context"
        ],
        "required_output_fields": [
            "source", "crop", "disease", "summary",
            "organic_treatment", "chemical_treatment",
            "recovery_time", "preventive_measures",
            "warnings", "notes", "estimated_cost"
        ],
        "optional_output_fields": [],
    }
