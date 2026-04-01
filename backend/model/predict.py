from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torchvision import models

SUPPORTED_CROPS = {"Tomato", "Apple", "Grape"}


class DiseasePredictor:
    def __init__(self) -> None:
        self.base_path = Path(__file__).resolve().parent
        self.weights_path = self.base_path / "weights" / "best_model.pth"
        self.classes_path = self.base_path / "weights" / "class_names.json"
        self.model: Optional[torch.nn.Module] = None
        self.class_names: List[str] = []
        self.model_loaded = False

        self._load_class_names()
        self._load_model()

    def _load_class_names(self) -> None:
        if not self.classes_path.exists():
            print(f"[AgriVision] Missing class names file: {self.classes_path}")
            self.class_names = []
            return

        with self.classes_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            self.class_names = [str(item) for item in payload]
        elif isinstance(payload, dict):
            # Supports either index->name or name->index shaped json.
            if all(str(k).isdigit() for k in payload.keys()):
                self.class_names = [
                    str(v) for _, v in sorted(payload.items(), key=lambda kv: int(kv[0]))
                ]
            else:
                self.class_names = [str(k) for k in payload.keys()]
        else:
            raise ValueError("class_names.json must be a list or dict")

    def _build_model(self) -> torch.nn.Module:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 38)
        return model

    def _load_model(self) -> None:
        if not self.weights_path.exists():
            print(
                "[AgriVision] Model weights not found at "
                f"{self.weights_path}. Please add best_model.pth before inference."
            )
            self.model_loaded = False
            return

        self.model = self._build_model()
        state_dict = torch.load(self.weights_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_loaded = True

    @staticmethod
    def _parse_class_name(class_name: str) -> Dict[str, object]:
        normalized = class_name.replace("___", "__")
        parts = normalized.split("__", maxsplit=1)

        crop = parts[0].replace("_", " ").strip().title() if parts else "Unknown"
        disease_raw = parts[1] if len(parts) > 1 else "Unknown"
        disease_name = disease_raw.replace("_", " ").strip().title()
        is_healthy = "healthy" in disease_raw.lower()

        return {
            "crop_name": crop,
            "disease_name": disease_name,
            "is_healthy": is_healthy,
        }

    @staticmethod
    def _severity_from_confidence(confidence: float) -> str:
        if confidence >= 85.0:
            return "High"
        if confidence >= 60.0:
            return "Moderate"
        return "Low"

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, object]:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Check weights file in backend/model/weights.")

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        confidence_pct = float(confidence.item() * 100.0)
        idx = int(pred_idx.item())

        if not self.class_names:
            raise RuntimeError("Class names are not loaded. Check class_names.json.")

        if idx >= len(self.class_names):
            raise RuntimeError(
                f"Predicted class index {idx} exceeds class names size {len(self.class_names)}"
            )

        disease_class = self.class_names[idx]
        parsed = self._parse_class_name(disease_class)

        crop_name = str(parsed["crop_name"])
        if crop_name not in SUPPORTED_CROPS:
            return {
                "error": (
                    "Unsupported crop detected. This model currently supports only "
                    "Tomato, Apple, and Grape."
                ),
                "disease_class": disease_class,
                "confidence": round(confidence_pct, 2),
                "crop_name": crop_name,
            }

        severity = self._severity_from_confidence(confidence_pct)
        return {
            "disease_class": disease_class,
            "confidence": round(confidence_pct, 2),
            "severity": severity,
            "crop_name": crop_name,
            "disease_name": parsed["disease_name"],
            "is_healthy": bool(parsed["is_healthy"]),
            "confidence_gate_triggered": confidence_pct < 60.0,
        }


predictor = DiseasePredictor()
