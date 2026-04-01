"""
Knowledge base loading and querying for crop diseases.

Loads disease knowledge from JSON files and provides functions to query
disease information, treatments, and prevention measures.
"""

import re
from typing import Dict, Optional, List
from . import config
from . import utils

logger = utils.setup_logger(__name__)

# Global knowledge base cache
_knowledge_base_cache = None


def _canonicalize_disease_name(disease: str) -> str:
    """Normalize disease names for robust matching across modules."""
    d = utils.normalize_string(disease)
    d = d.replace("___", " ")
    d = d.replace("__", " ")
    d = d.replace("(", " ").replace(")", " ")
    d = d.replace("-", " ")
    d = re.sub(r"\s+", " ", d).strip()

    # Treat crop suffixes like "early blight tomato" and "early blight" as equivalent.
    for crop_token in ("tomato", "apple", "grape"):
        if d.endswith(" " + crop_token):
            d = d[: -len(crop_token)].strip()
    return d


def load_knowledge_base() -> Dict:
    """
    Load the disease knowledge base from JSON file.
    
    Caches the result to avoid repeated I/O.
    
    Returns:
        Dictionary with structure:
        {
            "crops": {
                "Tomato": [
                    {
                        "disease": "...",
                        "symptoms": [...],
                        "organic_treatments": [...],
                        "chemical_treatments": [...],
                        ...
                    },
                    ...
                ],
                ...
            }
        }
    """
    global _knowledge_base_cache
    
    if _knowledge_base_cache is not None:
        return _knowledge_base_cache
    
    kb = utils.load_json(config.DISEASE_KNOWLEDGE_FILE, default={"crops": {}})
    _knowledge_base_cache = kb
    
    logger.info(f"Loaded knowledge base with {len(kb.get('crops', {}))} crops")
    return kb


def get_disease_context(crop: str, disease: str) -> Optional[Dict]:
    """
    Retrieve knowledge base entry for a specific crop + disease pair.
    
    Args:
        crop: Crop name (e.g., "Tomato")
        disease: Disease name (e.g., "Early Blight")
    
    Returns:
        Dictionary with disease information, or None if not found
    """
    kb = load_knowledge_base()
    crops = kb.get("crops", {})
    
    # Normalize crop name for case-insensitive matching
    crop_normalized = utils.normalize_string(crop)
    
    for crop_key, diseases_list in crops.items():
        if utils.normalize_string(crop_key) == crop_normalized:
            canonical_target = _canonicalize_disease_name(disease)

            # Pass 1: exact normalized match.
            for disease_entry in diseases_list:
                if utils.normalize_string(disease_entry.get("disease", "")) == utils.normalize_string(disease):
                    return disease_entry

            # Pass 2: canonicalized match (handles punctuation/crop suffix variants).
            for disease_entry in diseases_list:
                entry_name = disease_entry.get("disease", "")
                if _canonicalize_disease_name(entry_name) == canonical_target:
                    return disease_entry

            # Pass 3: relaxed contains match for near-equivalent labels.
            for disease_entry in diseases_list:
                entry_canonical = _canonicalize_disease_name(disease_entry.get("disease", ""))
                if canonical_target in entry_canonical or entry_canonical in canonical_target:
                    return disease_entry
    
    logger.debug(f"Disease entry not found: {crop} - {disease}")
    return None


def list_diseases(crop: str) -> List[str]:
    """
    List all known diseases for a given crop.
    
    Args:
        crop: Crop name
    
    Returns:
        List of disease names
    """
    kb = load_knowledge_base()
    crops = kb.get("crops", {})
    
    crop_normalized = utils.normalize_string(crop)
    
    for crop_key, diseases_list in crops.items():
        if utils.normalize_string(crop_key) == crop_normalized:
            return [d.get("disease", "Unknown") for d in diseases_list]
    
    return []


def list_crops() -> List[str]:
    """
    List all crops in the knowledge base.
    
    Returns:
        List of crop names
    """
    kb = load_knowledge_base()
    return list(kb.get("crops", {}).keys())


def get_symptoms(crop: str, disease: str) -> List[str]:
    """
    Get symptoms for a disease.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        List of symptom descriptions
    """
    context = get_disease_context(crop, disease)
    if context:
        return context.get("symptoms", [])
    return []


def get_organic_treatments(crop: str, disease: str) -> List[str]:
    """
    Get organic treatment options for a disease.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        List of organic treatment steps
    """
    context = get_disease_context(crop, disease)
    if context:
        return context.get("organic_treatments", [])
    return []


def get_chemical_treatments(crop: str, disease: str) -> List[str]:
    """
    Get chemical treatment options for a disease.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        List of chemical treatment steps
    """
    context = get_disease_context(crop, disease)
    if context:
        return context.get("chemical_treatments", [])
    return []


def get_preventive_measures(crop: str, disease: str) -> List[str]:
    """
    Get preventive measures for a disease.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        List of preventive measures
    """
    context = get_disease_context(crop, disease)
    if context:
        return context.get("preventive_measures", [])
    return []


def get_recovery_time(crop: str, disease: str) -> int:
    """
    Get estimated recovery time in days.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        Recovery time in days (default 21)
    """
    context = get_disease_context(crop, disease)
    if context:
        return context.get("recovery_time_days", 21)
    return 21


def get_notes(crop: str, disease: str) -> List[str]:
    """
    Get additional notes about a disease.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        List of notes
    """
    context = get_disease_context(crop, disease)
    if context:
        return context.get("notes", [])
    return []


def is_disease_known(crop: str, disease: str) -> bool:
    """
    Check if a disease is in the knowledge base.
    
    Args:
        crop: Crop name
        disease: Disease name
    
    Returns:
        True if disease is known
    """
    return get_disease_context(crop, disease) is not None


def reload_knowledge_base():
    """
    Force reload of knowledge base from disk.
    
    Useful for testing or if data files are updated.
    """
    global _knowledge_base_cache
    _knowledge_base_cache = None
    logger.info("Knowledge base cache cleared")
