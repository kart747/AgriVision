from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torchvision import models

SUPPORTED_CROPS = {"Tomato", "Apple", "Grape"}


class DiseasePredictor:
    def __init__(self, weights_path: Path, classes_path: Path, num_classes: int = 18) -> None:
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.num_classes = num_classes
        self.model: torch.nn.Module | None = None
        self.class_names: List[str] = []
        self.model_loaded = False

    def load_resources(self) -> None:
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        self.model_loaded = self.model is not None

    def _load_class_names(self) -> List[str]:
        if not self.classes_path.exists():
            print(f"[AgriVision] Missing class names file: {self.classes_path}")
            return []

        with self.classes_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            names = [str(item) for item in payload]
        elif isinstance(payload, dict) and all(str(k).isdigit() for k in payload.keys()):
            names = [str(v) for _, v in sorted(payload.items(), key=lambda kv: int(kv[0]))]
        else:
            raise ValueError("class_names.json must be a list or index-keyed dict")

        if len(names) != self.num_classes:
            print(
                "[AgriVision] Warning: class_names count is "
                f"{len(names)}, expected {self.num_classes}."
            )
        return names

    def _build_model(self) -> torch.nn.Module:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, self.num_classes)
        return model

    def _load_model(self) -> torch.nn.Module | None:
        if not self.weights_path.exists():
            print(
                "[AgriVision] Model weights not found at "
                f"{self.weights_path}. Please add best_model.pth before inference."
            )
            return None

        model = self._build_model()
        state_dict = torch.load(self.weights_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model

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
            raise ValueError(
                "Unsupported crop detected. This model currently supports only "
                "Tomato, Apple, and Grape."
            )

        severity_label = self._severity_from_confidence(confidence_pct)
        flagged = confidence_pct < 75.0
        return {
            "class_index": idx,
            "confidence": round(confidence_pct, 2),
            "severity_label": severity_label,
            "crop_name": crop_name,
            "disease_name": parsed["disease_name"],
            "is_healthy": bool(parsed["is_healthy"]),
            "flagged": flagged,
            "flag_reason": "Low confidence (<75%)." if flagged else None,
        }
