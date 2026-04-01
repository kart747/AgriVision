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
    prompt = """You are an agricultural disease expert providing recommendations for crop disease management.

Your role:
- Take crop type, disease name, and context (confidence, severity, location, time)
- Provide actionable, practical treatment and prevention guidance
- Consider both organic and chemical treatment options
- Provide recovery time estimates
- Include critical warnings if the disease threatens crop loss

CRITICAL: You MUST respond with ONLY valid JSON (no markdown, no code blocks, no other text).
The JSON must contain exactly these fields:
{
  "source": "llm",
  "crop": "crop name from input",
  "disease": "disease name from input",
  "summary": "1-2 sentence explanation of the disease",
  "organic_treatment": ["step 1", "step 2", ...],
  "chemical_treatment": ["step 1", "step 2", ...],
  "recovery_time": "estimated time, e.g. '14-21 days'",
  "preventive_measures": ["measure 1", "measure 2", ...],
  "warnings": ["warning 1", "warning 2", ...],
  "notes": ["note 1", "note 2", ...]
}

Constraints:
- Keep recommendations practical and farmer-friendly
- Use standard agricultural terminology
- Include dosages for chemical treatments
- Prioritize organic options when equally effective
- Be truthful about disease severity
- If disease will cause permanent loss, warn explicitly
- Do not invent treatments not based on agricultural science
- RESPOND WITH ONLY JSON, NO EXTRA TEXT OR MARKDOWN
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
    
    # Knowledge base info
    symptoms = kb_entry.get("symptoms", [])
    notes = kb_entry.get("notes", [])
    recovery_days = kb_entry.get("recovery_time_days", 21)
    
    # Build structured prompt
    lines = [
        f"Disease Detection Report:",
        f"Crop: {crop}",
        f"Disease: {disease}",
        f"Model Confidence: {confidence:.1%}",
        f"Severity: {severity}",
        f"Location: {location}",
        f"Time Context: {time_context}",
        "",
        f"Known Symptoms:",
    ]
    
    for symptom in symptoms[:5]:  # Max 5 symptoms to keep prompt concise
        lines.append(f"  - {symptom}")
    
    if notes:
        lines.append("")
        lines.append("Additional Context:")
        for note in notes[:3]:  # Max 3 notes
            lines.append(f"  - {note}")
    
    lines.extend([
        "",
        f"Provide recommendations for managing this disease on {crop} crops.",
        f"Consider the {severity} severity and urgency (expected recovery: ~{recovery_days} days).",
        "Provide both organic and chemical options suitable for small-scale farming.",
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
            "recovery_time", "preventive_measures"
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
        "model": "mixtral-8x7b-32768",
        "system_prompt_version": "1.0",
        "user_prompt_template_version": "1.0",
        "required_input_fields": [
            "crop", "disease", "confidence", "severity",
            "location", "time_context"
        ],
        "required_output_fields": [
            "source", "crop", "disease", "summary",
            "organic_treatment", "chemical_treatment",
            "recovery_time", "preventive_measures"
        ],
        "optional_output_fields": [
            "warnings", "notes"
        ],
    }
