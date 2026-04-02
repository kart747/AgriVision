from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForImageClassification

HF_REPO_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
HF_WEIGHTS_PATH = WEIGHTS_DIR / "hf_model.pth"
HF_CLASSES_PATH = WEIGHTS_DIR / "hf_class_names.json"


def _ordered_class_names(id2label: dict[int | str, str]) -> list[str]:
    return [
        str(v)
        for _, v in sorted(
            ((int(k), v) for k, v in id2label.items()),
            key=lambda kv: kv[0],
        )
    ]


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from Hugging Face: {HF_REPO_ID}")
    model = AutoModelForImageClassification.from_pretrained(HF_REPO_ID)

    class_names = _ordered_class_names(model.config.id2label)
    payload = {
        "architecture": "mobilenet_v2",
        "state_dict": model.state_dict(),
    }

    torch.save(payload, HF_WEIGHTS_PATH)
    with HF_CLASSES_PATH.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("Saved files:")
    print(f"- Weights: {HF_WEIGHTS_PATH}")
    print(f"- Classes: {HF_CLASSES_PATH}")
    print(f"Class count: {len(class_names)}")
    print("Class names:")
    for idx, name in enumerate(class_names):
        print(f"{idx:02d}: {name}")


if __name__ == "__main__":
    main()
