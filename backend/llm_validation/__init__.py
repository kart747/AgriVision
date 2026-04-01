"""
LLM + Validation Module for Crop Disease Intelligence System

A standalone, importable module for image validation and disease recommendation.
Combines local knowledge base with optional LLM-powered advice generation.

Key Features:
- Image quality validation (blur detection)
- Prediction confidence thresholding
- Location/region validation
- Fallback recommendation engine (no LLM required)
- Optional Groq LLM integration
- Structured JSON outputs

Usage:
    from llm_validation import generate_advice, run_validation_summary
    
    # Validate image
    validation = run_validation_summary("image.jpg", confidence=0.92)
    
    # Generate recommendations
    advice = generate_advice({
        "crop": "Tomato",
        "disease": "TYLCV",
        "confidence": 0.92
    })
"""

__version__ = "1.0.0"
__author__ = "Person 3 - LLM + Validation"
__description__ = "LLM-powered validation and recommendation engine for crop disease detection"

# Import key functions and classes for public API
from .validators import (
    validate_image_quality,
    validate_confidence,
    validate_location,
    run_validation_summary,
    ImageValidator,
    should_request_reupload,
    get_validation_message_for_ui,
)

from .advisor import (
    generate_advice,
    generate_fallback_advice,
    generate_advice_with_llm,
    get_advisor_status,
)

from .knowledge_base import (
    load_knowledge_base,
    get_disease_context,
    list_diseases,
    list_crops,
)

from .utils import (
    setup_logger,
    load_json,
    save_json,
    LLMValidationError,
    ValidationError,
    AdvisorError,
    ConfigError,
)

# Public API
__all__ = [
    # Validators
    "validate_image_quality",
    "validate_confidence",
    "validate_location",
    "run_validation_summary",
    "ImageValidator",
    "should_request_reupload",
    "get_validation_message_for_ui",
    # Advisor
    "generate_advice",
    "generate_fallback_advice",
    "generate_advice_with_llm",
    "get_advisor_status",
    # Knowledge Base
    "load_knowledge_base",
    "get_disease_context",
    "list_diseases",
    "list_crops",
    # Utilities
    "setup_logger",
    "load_json",
    "save_json",
    # Exceptions
    "LLMValidationError",
    "ValidationError",
    "AdvisorError",
    "ConfigError",
]
