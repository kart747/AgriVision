from __future__ import annotations

from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.predict import predictor
from model.preprocess import preprocess_image

APP_VERSION = "1.0.0"
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


class Recommendation(BaseModel):
    summary: str
    immediate_actions: List[str]
    organic_treatment: List[str]
    chemical_treatment: List[str]
    preventive_measures: List[str]
    estimated_recovery_days: str
    spray_window: str
    risk_level: str
    weekly_monitoring: str


class PredictSuccessResponse(BaseModel):
    status: str
    crop: str
    disease: str
    confidence: float
    severity: str
    blur_score: float
    location_warning: Optional[str]
    recommendation: Recommendation


def _build_recommendation(
    crop: str,
    disease: str,
    severity: str,
    confidence: float,
    is_healthy: bool,
) -> Recommendation:
    if is_healthy:
        return Recommendation(
            summary=f"{crop} appears healthy with strong model confidence.",
            immediate_actions=[
                "Continue regular irrigation and nutrition schedule",
                "Keep weekly visual checks for new spots or curling",
            ],
            organic_treatment=[
                "Apply seaweed extract foliar spray once in 10-14 days",
                "Use compost tea to maintain leaf vigor",
            ],
            chemical_treatment=[
                "No chemical treatment needed at this stage",
            ],
            preventive_measures=[
                "Sanitize pruning tools before each use",
                "Avoid overhead watering during late evening",
            ],
            estimated_recovery_days="0-3 days",
            spray_window="Early morning between 6:00 AM and 8:00 AM",
            risk_level="Low",
            weekly_monitoring="Inspect 10 random leaves per plant block once every week.",
        )

    risk_level = "High" if severity == "High" else "Moderate"
    return Recommendation(
        summary=(
            f"Detected {disease} in {crop} with {confidence:.1f}% confidence. "
            "Start treatment immediately to prevent spread."
        ),
        immediate_actions=[
            "Isolate visibly infected plants where possible",
            "Remove severely affected leaves and dispose away from field",
            "Avoid working in wet foliage to reduce pathogen transfer",
        ],
        organic_treatment=[
            "Spray neem oil solution (3-5 ml/L) every 5-7 days",
            "Use Bacillus subtilis-based biofungicide as label directed",
        ],
        chemical_treatment=[
            "Use a crop-approved systemic fungicide/insecticide based on local extension advice",
            "Rotate active ingredients every 10-14 days to reduce resistance risk",
        ],
        preventive_measures=[
            "Maintain wider plant spacing and airflow",
            "Disinfect tools, trays, and irrigation touchpoints weekly",
            "Use certified disease-free seedlings in next cycle",
        ],
        estimated_recovery_days="14-21 days",
        spray_window="Spray during calm, dry weather in early morning or late afternoon",
        risk_level=risk_level,
        weekly_monitoring=(
            "Track lesion count, new leaf symptoms, and spread ratio twice weekly; "
            "escalate if symptoms increase after 7 days."
        ),
    )


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPG, PNG, or WEBP.",
        )


def _location_warning(latitude: Optional[float], longitude: Optional[float]) -> Optional[str]:
    if latitude is None and longitude is None:
        return None
    if latitude is None or longitude is None:
        return "Incomplete location coordinates provided. Send both latitude and longitude."
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return "Location coordinates are out of valid range."
    return None


app = FastAPI(title="AgriVision AI Backend", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": predictor.model_loaded,
        "version": APP_VERSION,
    }


@app.get("/classes")
def classes() -> dict:
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
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model weights are not loaded. Add best_model.pth and restart service.",
        )

    raw = await image.read()
    _validate_upload(image, raw)

    try:
        image_tensor, blur_score = preprocess_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = predictor.predict(image_tensor)

    if "error" in prediction:
        raise HTTPException(status_code=422, detail=prediction["error"])

    if prediction["confidence_gate_triggered"]:
        raise HTTPException(
            status_code=422,
            detail=(
                "Prediction confidence is too low (<60%). "
                "Please upload a clearer and closer leaf image."
            ),
        )

    crop = str(prediction["crop_name"])
    if crop_hint and crop_hint.strip().lower() != crop.lower():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Crop hint mismatch. Detected '{crop}', but crop_hint was '{crop_hint}'. "
                "Please verify the selected crop."
            ),
        )

    disease = str(prediction["disease_name"])
    severity = str(prediction["severity"])
    confidence = float(prediction["confidence"])
    is_healthy = bool(prediction["is_healthy"])

    recommendation = _build_recommendation(
        crop=crop,
        disease=disease,
        severity=severity,
        confidence=confidence,
        is_healthy=is_healthy,
    )

    return PredictSuccessResponse(
        status="success",
        crop=crop,
        disease=disease,
        confidence=round(confidence, 2),
        severity=severity,
        blur_score=round(float(blur_score), 2),
        location_warning=_location_warning(latitude, longitude),
        recommendation=recommendation,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
