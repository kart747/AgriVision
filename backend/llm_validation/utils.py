"""
Utility functions for the LLM validation module.

Includes JSON loading, logging, dict operations, and error handling.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any

from . import config

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """
    Set up and return a logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        level = logging.DEBUG if config.DEBUG else logging.INFO
        logger.setLevel(level)
    
    return logger


logger = setup_logger(__name__)


# ============================================================================
# JSON UTILITIES
# ============================================================================

def load_json(file_path: str, default: Any = None) -> Any:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file not found or invalid
    
    Returns:
        Loaded JSON object or default value
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"JSON file not found: {file_path}")
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {file_path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return default


def save_json(file_path: str, data: Any, indent: int = 2) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        file_path: Path to save JSON file
        data: Data to serialize
        indent: JSON indentation level
    
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Saved JSON to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


# ============================================================================
# DICTIONARY UTILITIES
# ============================================================================

def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with default.
    
    Args:
        d: Dictionary
        key: Key to retrieve
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    return d.get(key, default) if isinstance(d, dict) else default


def merge_dicts(base: dict, override: dict) -> dict:
    """
    Merge two dictionaries (override takes precedence).
    
    Args:
        base: Base dictionary
        override: Overriding dictionary
    
    Returns:
        Merged dictionary
    """
    result = base.copy() if isinstance(base, dict) else {}
    if isinstance(override, dict):
        result.update(override)
    return result


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ============================================================================
# STRING UTILITIES
# ============================================================================

def normalize_string(s: str) -> str:
    """
    Normalize string for comparison (lowercase, strip whitespace).
    
    Args:
        s: Input string
    
    Returns:
        Normalized string
    """
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to max length.
    
    Args:
        s: Input string
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def is_valid_confidence(confidence: float) -> bool:
    """
    Check if confidence value is valid (0.0 to 1.0).
    
    Args:
        confidence: Confidence value
    
    Returns:
        True if valid
    """
    try:
        return 0.0 <= float(confidence) <= 1.0
    except (TypeError, ValueError):
        return False


def is_valid_crop(crop: str) -> bool:
    """
    Check if crop is in supported crops list.
    
    Args:
        crop: Crop name
    
    Returns:
        True if valid
    """
    return normalize_string(crop) in [normalize_string(c) for c in config.SUPPORTED_CROPS]


# ============================================================================
# ERROR HANDLING
# ============================================================================

class LLMValidationError(Exception):
    """Base exception for LLM validation module."""
    pass


class ValidationError(LLMValidationError):
    """Raised when validation fails."""
    pass


class AdvisorError(LLMValidationError):
    """Raised when advisor/recommendation generation fails."""
    pass


class ConfigError(LLMValidationError):
    """Raised when configuration is invalid."""
    pass


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_info(message: str, **kwargs):
    """Log info with optional context."""
    if kwargs:
        message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(message)


def log_error(message: str, exception: Exception = None, **kwargs):
    """Log error with optional exception and context."""
    if kwargs:
        message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
    if exception:
        logger.exception(message)
    else:
        logger.error(message)


def log_debug(message: str, **kwargs):
    """Log debug with optional context."""
    if config.DEBUG:
        if kwargs:
            message += " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(message)
