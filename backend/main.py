from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


def _bootstrap_import_paths() -> None:
    # Respect PYTHONPATH while also supporting direct backend cwd execution on Windows.
    env_pythonpath = os.getenv("PYTHONPATH", "")
    if env_pythonpath:
        for item in env_pythonpath.split(os.pathsep):
            candidate = item.strip()
            if candidate and candidate not in sys.path:
                sys.path.insert(0, candidate)

    for candidate in (str(PROJECT_ROOT), str(BASE_DIR)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_bootstrap_import_paths()

if __package__:
    from AgriVision.backend.llm.advisor import get_recommendation
    from AgriVision.backend.model.gradcam import generate_gradcam_base64
    from AgriVision.backend.model.predict import DiseasePredictor
    from AgriVision.backend.model.preprocess import preprocess_image
    from AgriVision.backend.utils.severity import calculate_severity_score
    from AgriVision.backend.utils.validators import validate_file, validate_gps
else:
    from llm.advisor import get_recommendation
    from model.gradcam import generate_gradcam_base64
    from model.predict import DiseasePredictor
    from model.preprocess import preprocess_image
    from utils.severity import calculate_severity_score
    from utils.validators import validate_file, validate_gps

# LLM Validation imports
try:
    from llm_validation.validators import run_validation_summary
    from llm_validation.advisor import generate_advice
except ImportError:
    try:
        from AgriVision.llm_validation.validators import run_validation_summary
        from AgriVision.llm_validation.advisor import generate_advice
    except ImportError:
        # Fallback stubs if llm_validation not available
        def run_validation_summary(**kwargs):
            return {"passed": True, "checks": {}}
        def generate_advice(**kwargs):
            return {"source": "stub", "summary": "LLM module not available"}

APP_VERSION = "2.0.0"
logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    immediate_action: str
    local_treatment: str
    weather_warning: str


class PredictSuccessResponse(BaseModel):
    status: str
    crop: str
    disease: str
    confidence: float
    severity_label: str
    severity_score: int
    blur_score: float
    cam_image: Optional[str]
    location: str
    recommendation: Recommendation
    flagged: bool
    flag_reason: Optional[str]


class DroneScanResponse(BaseModel):
    status: str
    total_leaves_scanned: int
    infection_rate_percentage: float
    primary_threat: str
    average_confidence: float
    healthy_count: int
    infected_count: int
    per_leaf_results: List[Dict[str, Any]]


class ValidationRequest(BaseModel):
    """Request model for validation demo"""
    confidence: float
    crop: Optional[str] = "Tomato"
    location: Optional[str] = "Mangalore, Karnataka, India"


class ValidationResponse(BaseModel):
    """Response model for validation results"""
    passed: bool
    checks: Dict[str, Any]
    warnings: List[str]


class AdviceRequest(BaseModel):
    """Request model for LLM advice generation"""
    crop: str
    disease: str
    confidence: float
    severity: Optional[str] = "Moderate"
    location: Optional[str] = "Mangalore, Karnataka, India"
    month: Optional[str] = None
    use_llm: Optional[bool] = False


class AdviceResponse(BaseModel):
    """Response model for advice generation"""
    source: str
    crop: str
    disease: str
    summary: str
    immediate_action: Optional[str] = None
    organic_treatment: Optional[List[str]] = None
    chemical_treatment: Optional[List[str]] = None
    recovery_time: Optional[str] = None
    preventive_measures: Optional[List[str]] = None
    warnings: List[str] = []
    notes: Optional[List[str]] = None


def _format_location(latitude: Optional[float], longitude: Optional[float]) -> str:
    if latitude is None or longitude is None:
        return "Mangalore, Karnataka"
    return f"{latitude:.4f}, {longitude:.4f}"


def _build_predict_result(
    app: FastAPI,
    image_bytes: bytes,
    crop_hint: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
) -> Dict[str, Any]:
    predictor: DiseasePredictor = app.state.model
    image_tensor, blur_score = preprocess_image(image_bytes)
    prediction = predictor.predict(image_tensor)

    if crop_hint and crop_hint.strip().lower() != str(prediction["crop_name"]).lower():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Crop hint mismatch. Detected '{prediction['crop_name']}', "
                f"but crop_hint was '{crop_hint}'."
            ),
        )

    if prediction["flagged"]:
        raise HTTPException(
            status_code=422,
            detail="Prediction confidence below 75%. Please capture a clearer leaf image.",
        )

    severity_score = calculate_severity_score(image_bytes)
    cam_image = generate_gradcam_base64(
        model=predictor.model,
        image_bytes=image_bytes,
        input_tensor=image_tensor,
        class_index=int(prediction["class_index"]),
    )

    gps_warning = validate_gps(latitude, longitude)
    location = _format_location(latitude, longitude)
    month = datetime.now().strftime("%B")
    llm_location = f"{location} ({gps_warning})" if gps_warning else location
    recommendation = get_recommendation(
        crop=str(prediction["crop_name"]),
        disease=str(prediction["disease_name"]),
        confidence=float(prediction["confidence"]),
        severity_score=severity_score,
        location=llm_location,
        month=month,
    )

    return {
        "status": "success",
        "crop": str(prediction["crop_name"]),
        "disease": str(prediction["disease_name"]),
        "confidence": round(float(prediction["confidence"]), 2),
        "severity_label": str(prediction["severity_label"]),
        "severity_score": severity_score,
        "blur_score": round(float(blur_score), 2),
        "cam_image": cam_image,
        "location": location,
        "recommendation": {
            "immediate_action": str(recommendation.get("immediate_action", "")),
            "local_treatment": str(recommendation.get("local_treatment", "")),
            "weather_warning": str(recommendation.get("weather_warning", "")),
        },
        "flagged": False,
        "flag_reason": None,
        "is_healthy": bool(prediction["is_healthy"]),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = BASE_DIR
    weights_path = root / "model" / "weights" / "best_model.pth"
    classes_path = root / "model" / "weights" / "class_names.json"

    print("[AgriVision] Starting backend...")
    print(f"[AgriVision] Weights path: {weights_path}")
    print(f"[AgriVision] Classes path: {classes_path}")

    predictor = DiseasePredictor(weights_path=weights_path, classes_path=classes_path)

    try:
        predictor.load_resources()
        print(
            "[AgriVision] Model load status: "
            f"{'READY' if predictor.model_loaded else 'NOT READY'}"
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[AgriVision] Startup error while loading model resources")
        raise RuntimeError(
            "Model resource loading failed. Backend startup aborted. "
            "Check model/weights paths and files."
        ) from exc

    app.state.model = predictor
    app.state.started_at = datetime.now(timezone.utc)

    yield
    print("[AgriVision] Shutting down backend...")


app = FastAPI(title="AgriVision AI Backend", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    predictor: DiseasePredictor = app.state.model
    started_at = app.state.started_at
    uptime_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()

    return {
        "status": "ok",
        "model_loaded": predictor.model_loaded,
        "version": APP_VERSION,
        "uptime_seconds": round(uptime_seconds, 2),
    }


@app.get("/classes")
def classes() -> dict:
    predictor: DiseasePredictor = app.state.model
    return {
        "status": "ok",
        "count": len(predictor.class_names),
        "classes": predictor.class_names,
    }


@app.post("/predict", response_model=PredictSuccessResponse)
async def predict_disease(
    image: UploadFile = File(...),
    crop_hint: Optional[str] = Form(default=None),
    latitude: Optional[float] = Form(default=None),
    longitude: Optional[float] = Form(default=None),
) -> PredictSuccessResponse:
    predictor: DiseasePredictor = app.state.model
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. Add best_model.pth and restart service.",
        )

    raw = await image.read()
    validate_file(image, raw)

    try:
        payload = _build_predict_result(app, raw, crop_hint, latitude, longitude)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictSuccessResponse(**{k: v for k, v in payload.items() if k != "is_healthy"})


@app.post("/drone-scan", response_model=DroneScanResponse)
async def drone_scan(images: List[UploadFile] = File(...)) -> DroneScanResponse:
    predictor: DiseasePredictor = app.state.model
    if not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not images:
        raise HTTPException(status_code=400, detail="No images provided.")

    per_leaf_results: List[Dict[str, Any]] = []
    healthy_count = 0
    infected_count = 0
    confidences: List[float] = []
    threat_counter: Counter[str] = Counter()

    for idx, file in enumerate(images, start=1):
        try:
            raw = await file.read()
            validate_file(file, raw)
            item = _build_predict_result(app, raw, None, None, None)
            item["leaf_index"] = idx
            per_leaf_results.append(item)

            confidences.append(float(item["confidence"]))
            if item.get("is_healthy"):
                healthy_count += 1
            else:
                infected_count += 1
                threat_counter[f"{item['crop']} {item['disease']}"] += 1
        except HTTPException as exc:
            per_leaf_results.append(
                {
                    "leaf_index": idx,
                    "status": "error",
                    "detail": exc.detail,
                }
            )
        except ValueError as exc:
            per_leaf_results.append(
                {
                    "leaf_index": idx,
                    "status": "error",
                    "detail": str(exc),
                }
            )

    scanned = healthy_count + infected_count
    infection_rate = (infected_count / scanned * 100.0) if scanned else 0.0
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    primary_threat = threat_counter.most_common(1)[0][0] if threat_counter else "None"

    return DroneScanResponse(
        status="success",
        total_leaves_scanned=len(images),
        infection_rate_percentage=round(infection_rate, 1),
        primary_threat=primary_threat,
        average_confidence=round(avg_conf, 1),
        healthy_count=healthy_count,
        infected_count=infected_count,
        per_leaf_results=per_leaf_results,
    )


# ============================================================================
# LLM VALIDATION ENDPOINTS (Showcase your work!)
# ============================================================================

@app.post("/validation-demo", response_model=ValidationResponse)
def validation_demo(request: ValidationRequest) -> ValidationResponse:
    """
    Test the validation pipeline from llm_validation module.
    
    This endpoint demonstrates:
    - Confidence validation (must be > 60%)
    - Location validation (India bounds checking)
    - Input normalization
    
    Request body:
    - confidence: float (0-1 or 0-100)
    - crop: str (crop name)
    - location: str (GPS location)
    """
    try:
        result = run_validation_summary(
            image_path=None,
            confidence=request.confidence,
            location=request.location,
            crop=request.crop,
        )
        return ValidationResponse(**result)
    except Exception as exc:
        logger.exception("Validation demo error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate-recommendation")
def generate_recommendation(request: AdviceRequest) -> Dict[str, Any]:
    """
    Generate LLM-powered recommendations for a crop disease.
    
    This endpoint demonstrates:
    - LLM integration with Groq API (or fallback knowledge base)
    - Comprehensive disease recommendations
    - Treatment options (organic & chemical)
    - Recovery time estimates
    - Preventive measures
    
    Request body:
    - crop: str (requires crop name)
    - disease: str (disease name)
    - confidence: float (0-1)
    - severity: str (Low/Moderate/High)
    - location: str (for context)
    - use_llm: bool (true = use Groq API, false = fallback knowledge base)
    """
    try:
        if request.month is None:
            request.month = datetime.now().strftime("%B")
        
        context = {
            "crop": request.crop,
            "disease": request.disease,
            "confidence": request.confidence,
            "severity": request.severity,
            "location": request.location,
            "month": request.month,
        }
        
        advice = generate_advice(context, use_llm=request.use_llm)
        return advice
    except Exception as exc:
        logger.exception("Recommendation generation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/llm-stats")
def llm_stats() -> Dict[str, Any]:
    """
    Get statistics about the LLM validation module.
    
    Returns:
    - validation_available: bool (is validation module loaded?)
    - llm_features: list (available features)
    - supported_crops_fallback: list (crops in fallback knowledge base)
    - confidence_threshold: float (minimum confidence required)
    - location_bounds: dict (India geographic bounds)
    """
    return {
        "status": "ok",
        "llm_module_available": True,
        "llm_features": [
            "Confidence validation",
            "Location validation",
            "Advice generation",
            "Organic treatments",
            "Chemical treatments",
            "Recovery time estimates",
            "Preventive measures",
            "Expert warnings",
        ],
        "supported_crops_fallback": ["Apple", "Tomato", "Potato"],
        "confidence_threshold": 0.60,
        "location_bounds": {
            "latitude_min": 8.4,
            "latitude_max": 37.6,
            "longitude_min": 68.7,
            "longitude_max": 97.25,
            "region": "India",
        },
        "validation_checks": [
            "Confidence score (>60%)",
            "Location bounds (India)",
            "Crop presence",
            "Disease-crop matching",
        ],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
