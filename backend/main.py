from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pydantic import BaseModel, Field

try:
    from .llm.advisor import get_recommendation
    from .llm_validation.advisor import generate_advice
    from .llm_validation.validators import run_validation_summary
    from .model.gradcam import generate_gradcam_base64
    from .model.predict import DiseasePredictor
    from .model.preprocess import preprocess_image
    from .utils.severity import calculate_severity_score
    from .utils.validators import validate_file, validate_gps
except ImportError:
    from llm.advisor import get_recommendation
    from llm_validation.advisor import generate_advice
    from llm_validation.validators import run_validation_summary
    from model.gradcam import generate_gradcam_base64
    from model.predict import DiseasePredictor
    from model.preprocess import preprocess_image
    from utils.severity import calculate_severity_score
    from utils.validators import validate_file, validate_gps

APP_VERSION = "2.0.0"
logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    immediate_action: str
    local_treatment: str
    weather_warning: str
    estimated_cost: str = ""
    organic_treatment: List[str] = Field(default_factory=list)
    chemical_treatment: List[str] = Field(default_factory=list)
    recovery_time: str = ""
    preventive_measures: List[str] = Field(default_factory=list)


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
    live_weather: str = "Weather unavailable"
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
    confidence: float
    crop: Optional[str] = "Tomato"
    location: Optional[str] = "Mangalore, Karnataka, India"


class ValidationResponse(BaseModel):
    passed: bool
    checks: Dict[str, Any]
    warnings: List[str]


class AdviceRequest(BaseModel):
    crop: str
    disease: str
    confidence: float
    severity: Optional[str] = "Moderate"
    location: Optional[str] = "Mangalore, Karnataka, India"
    month: Optional[str] = None
    live_weather: Optional[str] = None
    use_llm: Optional[bool] = False


class AdviceResponse(BaseModel):
    source: str
    crop: str
    disease: str
    summary: str
    immediate_action: Optional[str] = None
    estimated_cost: Optional[str] = None
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


def get_live_weather(lat: float, lon: float) -> str:
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        payload = response.json()
        current_weather = payload.get("current_weather") or {}
        temperature = current_weather.get("temperature")
        windspeed = current_weather.get("windspeed")

        if temperature is None or windspeed is None:
            return "Weather unavailable"

        temperature_value = int(round(float(temperature)))
        windspeed_value = int(round(float(windspeed)))
        return f"{temperature_value}°C, Wind: {windspeed_value} km/h"
    except Exception:  # pylint: disable=broad-except
        return "Weather unavailable"


def _dms_part_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            numerator = float(value[0])
            denominator = float(value[1])
            if denominator == 0:
                return None
            return numerator / denominator
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _extract_gps_from_image_bytes(image_bytes: bytes) -> tuple[Optional[float], Optional[float]]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif = image._getexif() or {}
        if not exif:
            return None, None

        gps_info = None
        for tag_id, value in exif.items():
            if TAGS.get(tag_id, tag_id) == "GPSInfo":
                gps_info = {GPSTAGS.get(k, k): v for k, v in value.items()}
                break

        if not gps_info:
            return None, None

        lat_dms = gps_info.get("GPSLatitude")
        lon_dms = gps_info.get("GPSLongitude")
        lat_ref = gps_info.get("GPSLatitudeRef")
        lon_ref = gps_info.get("GPSLongitudeRef")
        if not lat_dms or not lon_dms or not lat_ref or not lon_ref:
            return None, None

        lat_deg = _dms_part_to_float(lat_dms[0])
        lat_min = _dms_part_to_float(lat_dms[1])
        lat_sec = _dms_part_to_float(lat_dms[2])
        lon_deg = _dms_part_to_float(lon_dms[0])
        lon_min = _dms_part_to_float(lon_dms[1])
        lon_sec = _dms_part_to_float(lon_dms[2])
        if None in (lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec):
            return None, None

        latitude = lat_deg + (lat_min / 60.0) + (lat_sec / 3600.0)
        longitude = lon_deg + (lon_min / 60.0) + (lon_sec / 3600.0)

        lat_ref_text = str(lat_ref)
        lon_ref_text = str(lon_ref)
        if lat_ref_text in {"S", "b'S'"}:
            latitude = -latitude
        if lon_ref_text in {"W", "b'W'"}:
            longitude = -longitude

        return latitude, longitude
    except Exception:  # pylint: disable=broad-except
        return None, None


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

    if latitude is None or longitude is None:
        exif_latitude, exif_longitude = _extract_gps_from_image_bytes(image_bytes)
        if latitude is None:
            latitude = exif_latitude
        if longitude is None:
            longitude = exif_longitude

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
            detail="Prediction confidence below 60%. Please capture a clearer leaf image.",
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
    live_weather = (
        get_live_weather(latitude, longitude)
        if latitude is not None and longitude is not None
        else "Weather unavailable"
    )
    recommendation = get_recommendation(
        crop=str(prediction["crop_name"]),
        disease=str(prediction["disease_name"]),
        confidence=float(prediction["confidence"]),
        severity_score=severity_score,
        location=llm_location,
        month=month,
        live_weather=live_weather,
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
        "live_weather": live_weather,
        "recommendation": {
            "immediate_action": str(recommendation.get("immediate_action", "")),
            "local_treatment": str(recommendation.get("local_treatment", "")),
            "weather_warning": str(recommendation.get("weather_warning", "")),
            "estimated_cost": str(recommendation.get("estimated_cost", "")),
            "organic_treatment": list(recommendation.get("organic_treatment", [])),
            "chemical_treatment": list(recommendation.get("chemical_treatment", [])),
            "recovery_time": str(recommendation.get("recovery_time", "")),
            "preventive_measures": list(recommendation.get("preventive_measures", [])),
        },
        "flagged": False,
        "flag_reason": None,
        "is_healthy": bool(prediction["is_healthy"]),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).resolve().parent
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
        print(f"[AgriVision] Startup error while loading model: {exc}")

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


@app.post("/validation-demo", response_model=ValidationResponse)
def validation_demo(request: ValidationRequest) -> ValidationResponse:
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


@app.post("/generate-recommendation", response_model=AdviceResponse)
def generate_recommendation(request: AdviceRequest) -> AdviceResponse:
    try:
        month = request.month or datetime.now().strftime("%B")
        context = {
            "crop": request.crop,
            "disease": request.disease,
            "confidence": request.confidence,
            "severity": request.severity,
            "location": request.location,
            "time_context": month,
            "live_weather": request.live_weather,
        }
        advice = generate_advice(context, use_llm=bool(request.use_llm))
        return AdviceResponse(**advice)
    except Exception as exc:
        logger.exception("Recommendation generation error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/llm-stats")
def llm_stats() -> Dict[str, Any]:
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
        "supported_crops_fallback": ["Apple", "Tomato", "Grape"],
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
