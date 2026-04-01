from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Sequence, Tuple
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.model.predict import DiseasePredictor
from backend.model.preprocess import preprocess_image

WEIGHTS_PATH = ROOT / "backend" / "model" / "weights" / "best_model.pth"
CLASSES_PATH = ROOT / "backend" / "model" / "weights" / "class_names.json"
TMP_DIR = ROOT / "training" / "tmp_test_images"

TEST_CASES: List[Tuple[str, str, Sequence[str]]] = [
    (
        "Tomato with Early Blight",
        "Tomato",
        [
            "https://source.unsplash.com/800x600/?tomato,leaf,disease",
            "https://source.unsplash.com/800x600/?tomato,plant,leaf",
            "https://picsum.photos/seed/tomato-leaf/800/600",
        ],
    ),
    (
        "Apple with Scab",
        "Apple",
        [
            "https://source.unsplash.com/800x600/?apple,leaf,disease",
            "https://source.unsplash.com/800x600/?apple,orchard,leaf",
            "https://picsum.photos/seed/apple-leaf/800/600",
        ],
    ),
    (
        "Grape with Black Rot",
        "Grape",
        [
            "https://source.unsplash.com/800x600/?grape,leaf,disease",
            "https://source.unsplash.com/800x600/?vineyard,grape,leaf",
            "https://picsum.photos/seed/grape-leaf/800/600",
        ],
    ),
]


def _download_images() -> List[Tuple[str, str, Path]]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Tuple[str, str, Path]] = []
    for idx, (label, expected_crop, urls) in enumerate(TEST_CASES, start=1):
        out = TMP_DIR / f"sample_{idx}.jpg"
        downloaded = False
        last_error: Exception | None = None
        for url in urls:
            try:
                urlretrieve(url, out)
                downloaded = True
                break
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                continue

        if not downloaded:
            raise RuntimeError(f"Failed to download test image for '{label}': {last_error}")
        saved.append((label, expected_crop, out))
    return saved


def main() -> None:
    predictor = DiseasePredictor(weights_path=WEIGHTS_PATH, classes_path=CLASSES_PATH)
    predictor.load_resources()

    if not predictor.model_loaded:
        print("FAIL: Model failed to load. Run training/download_pretrained.py first.")
        return

    samples = _download_images()

    for label, expected_crop, image_path in samples:
        try:
            img_bytes = image_path.read_bytes()
            input_tensor, _ = preprocess_image(img_bytes)
            pred = predictor.predict(input_tensor)

            predicted_crop = str(pred["crop_name"])
            confidence = float(pred["confidence"])
            disease = str(pred["disease_name"])

            print(f"\nCase: {label}")
            print(f"Predicted: {predicted_crop} - {disease}")
            print(f"Confidence: {confidence:.2f}%")

            if predicted_crop.lower() == expected_crop.lower():
                print("PASS: Correct crop detected")
            else:
                print("FAIL: Wrong crop detected")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\nCase: {label}")
            print(f"FAIL: {exc}")


if __name__ == "__main__":
    main()
