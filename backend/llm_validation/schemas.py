"""
Type schemas and dataclass definitions for input/output contracts.

These define the shape of data flowing through the module for validation and IDE support.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

@dataclass
class PredictionContext:
    """
    Input context for a crop disease prediction.
    
    Represents the prediction made by the CNN model along with optional context.
    """
    crop: str  # e.g., "Tomato"
    disease: str  # e.g., "Tomato Yellow Leaf Curl Virus"
    confidence: float  # e.g., 0.92 (0.0 to 1.0)
    severity: Optional[str] = None  # e.g., "Mild", "Moderate", "Severe"
    location: Optional[str] = None  # e.g., "Mangalore, Karnataka, India"
    time_context: Optional[str] = None  # e.g., "Early Morning", "Mid-day", "Evening"
    image_path: Optional[str] = None  # Path to the image file
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PredictionContext':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# VALIDATION OUTPUT SCHEMAS
# ============================================================================

@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    warning: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "message": self.message,
            "warning": self.warning,
        }


@dataclass
class ValidationResult:
    """
    Complete validation result for an image + prediction.
    
    Contains results of blur, confidence, and location validation checks.
    """
    passed: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "warnings": self.warnings,
        }


# ============================================================================
# ADVISOR OUTPUT SCHEMAS
# ============================================================================

@dataclass
class AdvisorOutput:
    """
    Recommendation output from the advisor module.
    
    Contains structured recommendations for treatment, prevention, and recovery.
    Always returns a structured format (not free-text-only).
    """
    source: str  # "fallback" or "llm"
    crop: str  # Crop type
    disease: str  # Disease name
    summary: str  # Brief explanation of the disease
    organic_treatment: List[str] = field(default_factory=list)  # Organic options
    chemical_treatment: List[str] = field(default_factory=list)  # Chemical options
    recovery_time: str = ""  # Estimated recovery time, e.g., "14-21 days"
    preventive_measures: List[str] = field(default_factory=list)  # Prevention steps
    warnings: List[str] = field(default_factory=list)  # Important warnings
    notes: List[str] = field(default_factory=list)  # Additional notes
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "crop": self.crop,
            "disease": self.disease,
            "summary": self.summary,
            "organic_treatment": self.organic_treatment,
            "chemical_treatment": self.chemical_treatment,
            "recovery_time": self.recovery_time,
            "preventive_measures": self.preventive_measures,
            "warnings": self.warnings,
            "notes": self.notes,
        }


# ============================================================================
# KNOWLEDGE BASE SCHEMAS
# ============================================================================

@dataclass
class DiseaseKnowledge:
    """
    Knowledge base entry for a specific disease.
    
    Contains all curated information about a disease, treatments, and prevention.
    """
    crop: str
    disease: str
    symptoms: List[str] = field(default_factory=list)
    organic_treatments: List[str] = field(default_factory=list)
    chemical_treatments: List[str] = field(default_factory=list)
    recovery_time_days: int = 21
    preventive_measures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
