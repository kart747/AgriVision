"""
Validation layer for image quality, prediction confidence, and location context.

Provides guardrail checks before recommendations are generated.
"""

import os
import re
import cv2
import numpy as np
from typing import Optional, Dict, Any

from . import config
from . import utils
from . import schemas

logger = utils.setup_logger(__name__)


def _normalize_confidence_value(confidence: float) -> Optional[float]:
    """
    Normalize confidence to 0.0..1.0.

    Accepts either probability scale (0..1) or percentage scale (0..100).
    Returns None for invalid values.
    """
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


def _match_region_key(location: str, region_config: Dict[str, Any]) -> Optional[str]:
    """Find best matching configured region from a free-form location string."""
    if not location:
        return None

    loc = utils.normalize_string(location)

    # Handle coordinates or warnings in location strings from backend pipelines.
    numeric_coord_pattern = r"[-+]?\d{1,3}(?:\.\d+)?\s*,\s*[-+]?\d{1,3}(?:\.\d+)?"
    if re.search(numeric_coord_pattern, loc):
        return None

    for region in region_config.keys():
        region_norm = utils.normalize_string(region)
        if loc == region_norm or region_norm in loc or loc in region_norm:
            return region
    return None


# ============================================================================
# IMAGE QUALITY VALIDATION (BLUR DETECTION)
# ============================================================================

def validate_image_quality(
    image_path: str,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate image quality using Laplacian variance (blur detection).
    
    Images with low Laplacian variance are blurry and may reduce CNN confidence.
    
    Args:
        image_path: Path to image file
        threshold: Blur threshold (default from config)
    
    Returns:
        Dict with keys:
        - passed: bool
        - score: Laplacian variance
        - threshold: Threshold used
        - message: Human-readable result
    """
    if threshold is None:
        threshold = config.BLUR_THRESHOLD
    
    result = {
        "passed": False,
        "score": None,
        "threshold": threshold,
        "message": "Validation did not run"
    }
    
    # Check file exists
    if not os.path.exists(image_path):
        result["message"] = f"Image file not found: {image_path}"
        logger.warning(result["message"])
        return result
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            result["message"] = f"Failed to read image: {image_path}"
            logger.warning(result["message"])
            return result
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance on an 8-bit edge map to keep scores stable.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = cv2.convertScaleAbs(laplacian).var()

        result["score"] = float(laplacian_var)
        result["passed"] = bool(laplacian_var >= threshold)
        
        if result["passed"]:
            result["message"] = f"Image quality is acceptable (Laplacian: {laplacian_var:.1f})"
            logger.debug(f"Image quality check passed: {image_path}")
        else:
            result["message"] = f"Image is blurry (Laplacian: {laplacian_var:.1f}, need >= {threshold})"
            logger.warning(f"Image quality check failed: {image_path}")
        
        return result
    
    except Exception as e:
        result["message"] = f"Error processing image: {str(e)}"
        logger.error(f"Image quality validation error: {e}", exc_info=True)
        return result


# ============================================================================
# CONFIDENCE VALIDATION
# ============================================================================

def validate_confidence(
    confidence: float,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate prediction confidence against threshold.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        threshold: Confidence threshold (default from config)
    
    Returns:
        Dict with keys:
        - passed: bool
        - score: Confidence value
        - threshold: Threshold used
        - message: Human-readable result
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD
    
    result = {
        "passed": False,
        "score": None,
        "threshold": threshold,
        "message": "Invalid confidence value"
    }
    
    normalized_confidence = _normalize_confidence_value(confidence)
    if normalized_confidence is None:
        logger.warning(f"Invalid confidence value: {confidence}")
        return result

    result["score"] = float(normalized_confidence)
    result["passed"] = normalized_confidence >= threshold
    
    if result["passed"]:
        result["message"] = f"Prediction confidence is acceptable ({normalized_confidence:.1%})"
        logger.debug(f"Confidence check passed: {normalized_confidence:.1%}")
    else:
        result["message"] = (
            f"Prediction confidence is too low ({normalized_confidence:.1%}, need >= {threshold:.1%})"
        )
        logger.warning(f"Confidence check failed: {normalized_confidence:.1%}")
    
    return result


# ============================================================================
# LOCATION VALIDATION
# ============================================================================

def validate_location(
    location: Optional[str] = None,
    crop: Optional[str] = None,
    region_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Validate location and optionally check if crop matches expected crops for region.
    
    Args:
        location: Location string (e.g., "Mangalore, Karnataka, India")
        crop: Crop name to validate against region
        region_config: Custom region config (default from config)
    
    Returns:
        Dict with keys:
        - passed: bool (always True; warnings only)
        - warning: str or None
        - message: Human-readable result
    """
    if region_config is None:
        region_config = config.EXPECTED_CROPS_BY_REGION
    
    result = {
        "passed": True,
        "warning": None,
        "message": "No location provided"
    }
    
    # If no location, pass with neutral message
    if not location or not config.ENABLE_LOCATION_VALIDATION:
        result["message"] = "Location validation disabled or not provided"
        return result
    
    result["message"] = f"Location: {location}"

    matched_region = _match_region_key(location, region_config)
    if not matched_region:
        result["message"] = f"Location provided but not mapped to configured region: {location}"
        return result
    
    # Check if crop matches expected crops for location
    if crop and matched_region in region_config:
        expected_crops = region_config[matched_region]
        if crop not in expected_crops:
            result["warning"] = (
                f"Crop '{crop}' is not typically grown in '{matched_region}'. "
                f"Expected crops: {', '.join(expected_crops)}"
            )
            logger.warning(result["warning"])
    
    return result


# ============================================================================
# COMPREHENSIVE VALIDATION
# ============================================================================

def run_validation_summary(
    image_path: Optional[str] = None,
    confidence: Optional[float] = None,
    location: Optional[str] = None,
    crop: Optional[str] = None,
    blur_threshold: Optional[float] = None,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run all validation checks and return comprehensive result.
    
    Args:
        image_path: Path to image file (for blur detection)
        confidence: Model prediction confidence (0.0-1.0)
        location: Location string
        crop: Crop name
        blur_threshold: Custom blur threshold
        confidence_threshold: Custom confidence threshold
    
    Returns:
        Dict with keys:
        - passed: bool (True if all critical checks passed)
        - checks: Dict with results of each check (blur, confidence, location)
        - warnings: List of warning messages
    """
    result = {
        "passed": True,
        "checks": {},
        "warnings": []
    }
    
    # Run blur check if image path provided
    if image_path:
        blur_result = validate_image_quality(image_path, blur_threshold)
        result["checks"]["blur"] = blur_result
        if not blur_result["passed"]:
            result["passed"] = False
            result["warnings"].append(blur_result["message"])
    
    # Run confidence check if confidence provided
    if confidence is not None:
        conf_result = validate_confidence(confidence, confidence_threshold)
        result["checks"]["confidence"] = conf_result
        if not conf_result["passed"]:
            result["passed"] = False
            result["warnings"].append(conf_result["message"])
    
    # Run location check if location provided
    if location:
        loc_result = validate_location(location, crop, config.EXPECTED_CROPS_BY_REGION)
        result["checks"]["location"] = loc_result
        if loc_result["warning"]:
            result["warnings"].append(loc_result["warning"])
    
    logger.info(
        f"Validation summary: passed={result['passed']}, "
        f"checks={len(result['checks'])}, warnings={len(result['warnings'])}"
    )
    
    return result


# ============================================================================
# VALIDATION HELPER FUNCTIONS
# ============================================================================

def should_request_reupload(validation_result: Dict[str, Any]) -> bool:
    """
    Determine if user should be asked to re-upload the image.
    
    Triggered by:
    - Blur detection failure
    - Confidence below threshold
    
    Args:
        validation_result: Result from run_validation_summary()
    
    Returns:
        True if re-upload should be requested
    """
    if not validation_result.get("passed", True):
        blur_failed = not validation_result.get("checks", {}).get("blur", {}).get("passed", True)
        conf_failed = not validation_result.get("checks", {}).get("confidence", {}).get("passed", True)
        return blur_failed or conf_failed
    
    return False


def get_validation_message_for_ui(validation_result: Dict[str, Any]) -> str:
    """
    Generate a user-friendly message from validation result.
    
    Args:
        validation_result: Result from run_validation_summary()
    
    Returns:
        String suitable for UI display
    """
    if validation_result.get("passed", True):
        num_checks = len(validation_result.get("checks", {}))
        return f"✓ Image validation passed ({num_checks} checks)"
    else:
        warnings = validation_result.get("warnings", [])
        if warnings:
            return "⚠ Image validation failed:\n• " + "\n• ".join(warnings[:3])
        return "✗ Image validation failed"


# ============================================================================
# VALIDATION CLASS (OPTIONAL OOP INTERFACE)
# ============================================================================

class ImageValidator:
    """
    Class-based interface for image validation (optional alternative to functions).
    
    Useful if validation state needs to be tracked across multiple calls.
    """
    
    def __init__(self, blur_threshold: Optional[float] = None, 
                 confidence_threshold: Optional[float] = None):
        """Initialize validator with custom thresholds."""
        self.blur_threshold = blur_threshold or config.BLUR_THRESHOLD
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
    
    def validate(self, image_path: Optional[str] = None,
                 confidence: Optional[float] = None,
                 location: Optional[str] = None,
                 crop: Optional[str] = None) -> Dict[str, Any]:
        """Run complete validation."""
        return run_validation_summary(
            image_path=image_path,
            confidence=confidence,
            location=location,
            crop=crop,
            blur_threshold=self.blur_threshold,
            confidence_threshold=self.confidence_threshold
        )
