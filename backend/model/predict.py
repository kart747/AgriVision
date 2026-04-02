from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torchvision import models

SUPPORTED_CROPS = {"Tomato", "Apple", "Grape"}


class DiseasePredictor:
    def __init__(
        self,
        weights_path: Path,
        classes_path: Path,
        num_classes: Optional[int] = None,
    ) -> None:
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.num_classes = num_classes
        self.detected_num_classes = 0
        self.architecture = "efficientnet_b0"
        self.model: torch.nn.Module | None = None
        self.class_names: List[str] = []
        self.supported_indices: List[int] = []
        self.model_loaded = False

    def load_resources(self) -> None:
        self.class_names = self._load_class_names()
        self.detected_num_classes = len(self.class_names)
        self._refresh_supported_indices()
        self.model = self._load_model()
        self.model_loaded = self.model is not None
        if self.model_loaded:
            print(f"Model loaded: {self.detected_num_classes} classes (restored working checkpoint)")

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

        if self.num_classes is not None and len(names) != self.num_classes:
            print(
                "[AgriVision] Warning: class_names count is "
                f"{len(names)}, expected {self.num_classes}."
            )
        return names

    def _refresh_supported_indices(self) -> None:
        self.supported_indices = []
        for idx, name in enumerate(self.class_names):
            parsed = self._parse_class_name(name)
            if str(parsed["crop_name"]) in SUPPORTED_CROPS:
                self.supported_indices.append(idx)

    @staticmethod
    def _strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not state_dict:
            return state_dict
        if all(k.startswith("module.") for k in state_dict.keys()):
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def _infer_checkpoint_num_classes(state_dict: Dict[str, torch.Tensor], architecture: str, fallback: int) -> int:
        arch = architecture.lower()
        candidate_keys: List[str] = []
        if arch in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0", "mobilenet_v2", "mobilenetv2"}:
            candidate_keys = ["classifier.1.weight"]
        elif arch in {"resnet50", "resnet_50"}:
            candidate_keys = ["fc.weight"]

        for key in candidate_keys:
            tensor = state_dict.get(key)
            if tensor is not None and getattr(tensor, "ndim", 0) == 2:
                return int(tensor.shape[0])

        for key, tensor in state_dict.items():
            if key.endswith("classifier.1.weight") and getattr(tensor, "ndim", 0) == 2:
                return int(tensor.shape[0])
            if key.endswith("fc.weight") and getattr(tensor, "ndim", 0) == 2:
                return int(tensor.shape[0])

        return fallback

    @staticmethod
    def _resize_classifier_head(model: torch.nn.Module, architecture: str, num_classes: int) -> torch.nn.Module:
        arch = architecture.lower()

        if arch in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
            old_head = model.classifier[1]
            new_head = torch.nn.Linear(old_head.in_features, num_classes)
            copy_rows = min(old_head.out_features, num_classes)
            with torch.no_grad():
                if copy_rows > 0:
                    new_head.weight[:copy_rows].copy_(old_head.weight[:copy_rows])
                    new_head.bias[:copy_rows].copy_(old_head.bias[:copy_rows])
            model.classifier[1] = new_head
            return model

        if arch in {"mobilenet_v2", "mobilenetv2"}:
            old_head = model.classifier[1]
            new_head = torch.nn.Linear(old_head.in_features, num_classes)
            copy_rows = min(old_head.out_features, num_classes)
            with torch.no_grad():
                if copy_rows > 0:
                    new_head.weight[:copy_rows].copy_(old_head.weight[:copy_rows])
                    new_head.bias[:copy_rows].copy_(old_head.bias[:copy_rows])
            model.classifier[1] = new_head
            return model

        if arch in {"resnet50", "resnet_50"}:
            old_head = model.fc
            new_head = torch.nn.Linear(old_head.in_features, num_classes)
            copy_rows = min(old_head.out_features, num_classes)
            with torch.no_grad():
                if copy_rows > 0:
                    new_head.weight[:copy_rows].copy_(old_head.weight[:copy_rows])
                    new_head.bias[:copy_rows].copy_(old_head.bias[:copy_rows])
            model.fc = new_head
            return model

        raise ValueError(f"Unsupported architecture metadata in checkpoint: {architecture}")

    def _build_model(self, architecture: str, num_classes: int) -> torch.nn.Module:
        arch = architecture.lower()
        if arch in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, num_classes)
            self.architecture = "efficientnet_b0"
            return model

        if arch in {"mobilenet_v2", "mobilenetv2"}:
            model = models.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, num_classes)
            self.architecture = "mobilenet_v2"
            return model

        if arch in {"resnet50", "resnet_50"}:
            model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
            self.architecture = "resnet50"
            return model

        raise ValueError(f"Unsupported architecture metadata in checkpoint: {architecture}")

    def _load_model(self) -> torch.nn.Module | None:
        if not self.weights_path.exists():
            print(
                "[AgriVision] Model weights not found at "
                f"{self.weights_path}. Please add best_model.pth before inference."
            )
            return None

        payload = torch.load(self.weights_path, map_location=torch.device("cpu"))

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            architecture = str(payload.get("architecture", "efficientnet_b0"))
        else:
            state_dict = payload
            architecture = "efficientnet_b0"

        state_dict = self._strip_prefix_if_needed(state_dict)
        ckpt_num_classes = self._infer_checkpoint_num_classes(
            state_dict,
            architecture,
            self.detected_num_classes or 18,
        )

        model = self._build_model(architecture=architecture, num_classes=ckpt_num_classes)
        model.load_state_dict(state_dict, strict=False)

        if self.detected_num_classes and self.detected_num_classes != ckpt_num_classes:
            print(
                f"[AgriVision] Adjusting classifier head from {ckpt_num_classes} checkpoint classes "
                f"to {self.detected_num_classes} class_names entries."
            )
            model = self._resize_classifier_head(model, architecture, self.detected_num_classes)

        model.eval()
        return model

    @staticmethod
    def _parse_class_name(class_name: str) -> Dict[str, object]:
        # Preferred format from PlantVillage: Crop___Disease
        if "___" in class_name or "__" in class_name:
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

        # Unified merged-dataset format: crop_disease_name (e.g. tomato_late_blight)
        tokens = [t for t in class_name.strip().split("_") if t]
        if tokens:
            crop_token = tokens[0].lower()
            if crop_token in {"tomato", "apple", "grape"}:
                crop = crop_token.title()
                disease_raw = "_".join(tokens[1:]) if len(tokens) > 1 else "unknown"
                disease_name = disease_raw.replace("_", " ").strip().title()
                is_healthy = "healthy" in disease_raw.lower()
                return {
                    "crop_name": crop,
                    "disease_name": disease_name,
                    "is_healthy": is_healthy,
                }

        crop = "Unknown"
        disease_raw = class_name
        disease_name = class_name.replace("_", " ").strip().title()
        is_healthy = "healthy" in class_name.lower()

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
            if self.detected_num_classes == 38:
                if not self.supported_indices:
                    raise ValueError("No supported Tomato/Apple/Grape classes found in class_names.")
                filtered_probs = probs[:, self.supported_indices]
                confidence, filtered_idx = torch.max(filtered_probs, dim=1)
                pred_idx = torch.tensor(
                    [self.supported_indices[int(filtered_idx.item())]],
                    dtype=torch.long,
                )
            else:
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
        flagged = confidence_pct < 60.0
        return {
            "class_index": idx,
            "confidence": round(confidence_pct, 2),
            "severity_label": severity_label,
            "crop_name": crop_name,
            "disease_name": parsed["disease_name"],
            "is_healthy": bool(parsed["is_healthy"]),
            "flagged": flagged,
            "flag_reason": "Low confidence (<60%)." if flagged else None,
        }
