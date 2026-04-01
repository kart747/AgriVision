"""
Configuration and constants for the LLM validation module.

This module centralized all thresholds, API settings, and crop/disease mappings.
"""

import os
from typing import Dict, List

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

# Laplacian variance threshold for blur detection
# Images with Laplacian variance < BLUR_THRESHOLD are considered blurry
BLUR_THRESHOLD = 100.0

# Confidence threshold for model predictions
# Predictions with confidence < CONFIDENCE_THRESHOLD trigger validation failure
CONFIDENCE_THRESHOLD = 0.60

# ============================================================================
# LLM API CONFIGURATION
# ============================================================================

# API provider: 'groq' or 'openai' (Groq is free tier for hackathon)
LLM_PROVIDER = "groq"

# Groq API key (from environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq model name
GROQ_MODEL = "llama-3.3-70b-versatile"

# API timeout in seconds
LLM_API_TIMEOUT = 10

# ============================================================================
# LOCATION/FARM REGION CONFIGURATION
# ============================================================================

# Enable location validation
ENABLE_LOCATION_VALIDATION = True

# Expected crops per region (for validation warning)
EXPECTED_CROPS_BY_REGION: Dict[str, List[str]] = {
    "Mangalore, Karnataka, India": ["Tomato", "Grape", "Coconut"],
    "Punjab, India": ["Wheat", "Tomato", "Apple"],
    "Himachal Pradesh, India": ["Apple", "Tomato", "Grape"],
}

# ============================================================================
# CROP & DISEASE CONFIGURATION
# ============================================================================

# Supported crops for this demo
SUPPORTED_CROPS = ["Tomato", "Apple", "Grape"]

# Supported diseases (auto-loaded from knowledge base)
# This is populated at runtime from disease_knowledge.json
SUPPORTED_DISEASES = {}

# ============================================================================
# FALLBACK ADVICE CONFIGURATION
# ============================================================================

# Use fallback advice even if LLM is enabled (useful for testing)
FORCE_FALLBACK = False

# Include knowledge base context in fallback recommendations
USE_KB_CONTEXT = True

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

# Enable debug logging
DEBUG = True

# Log file path (optional)
LOG_FILE = None  # Set to a path to write logs to file

# ============================================================================
# DATA FILES
# ============================================================================

# Base directory for data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Path to disease knowledge base
DISEASE_KNOWLEDGE_FILE = os.path.join(DATA_DIR, "disease_knowledge.json")

# Path to farm regions file
FARM_REGIONS_FILE = os.path.join(DATA_DIR, "farm_regions.json")

# Path to sample cases file
SAMPLE_CASES_FILE = os.path.join(DATA_DIR, "sample_cases.json")

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Include detailed reasoning in advisor output
INCLUDE_DETAILED_REASONING = False

# Maximum length of recommendations list
MAX_RECOMMENDATIONS = 5

# Include warning notes in output
INCLUDE_WARNINGS = True


def get_config_dict() -> dict:
    """
    Return current configuration as a dictionary.
    Useful for logging and debugging.
    """
    return {
        "blur_threshold": BLUR_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "llm_provider": LLM_PROVIDER,
        "groq_model": GROQ_MODEL,
        "supported_crops": SUPPORTED_CROPS,
        "enable_location_validation": ENABLE_LOCATION_VALIDATION,
    }
