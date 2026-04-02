"""
Test model on images in TestData/ directory and calculate F1 score.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "TestData"
WEIGHTS_DIR = ROOT / "backend" / "model" / "weights"


IMAGE_LABELS = {
    "0a31e630-0d98-416b-b0e4-88a88aad1dc5___RS_HL 9653.JPG": "tomato_healthy",
    "0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG": "apple_scab",
    "0abc57ec-7f3b-482a-8579-21f3b2fb780b___RS_Erly.B 7609.JPG": "tomato_early_blight",
    "0c1667a2-61d7-4dee-b4d9-0d141a1ceb20___Mt.N.V_HL 9127.JPG": "grape_healthy",
    "0cd24b0c-0a9d-483f-8734-5c08988e029f___FREC_C.Rust 3762.JPG": "apple_rust",
}


def load_model():
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from model.predict import DiseasePredictor
    
    predictor = DiseasePredictor(
        weights_path=WEIGHTS_DIR / "best_model.pth",
        classes_path=WEIGHTS_DIR / "class_names.json"
    )
    predictor.load_resources()
    return predictor


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def main():
    print("Loading model...")
    predictor = load_model()
    transform = get_transform()
    
    image_files = [f for f in list(TEST_DIR.glob("*.JPG")) + list(TEST_DIR.glob("*.jpg")) if f.name in IMAGE_LABELS]
    
    print(f"\nFound {len(image_files)} images to test\n")
    
    results = []
    all_preds = []
    all_labels = []
    
    for img_path in sorted(image_files):
        actual = IMAGE_LABELS[img_path.name]
        
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        result = predictor.predict(tensor)
        
        crop = result['crop_name'].lower()
        disease = result['disease_name'].lower().replace(' ', '_')
        
        if result['is_healthy']:
            pred_class = f"{crop}_healthy"
        else:
            pred_class = f"{crop}_{disease}"
        
        correct = pred_class == actual
        all_preds.append(pred_class)
        all_labels.append(actual)
        
        print(f"{img_path.name}")
        print(f"  Actual:     {actual}")
        print(f"  Predicted:  {pred_class}")
        print(f"  Confidence: {result['confidence']:.2f}%")
        print(f"  {'✓ CORRECT' if correct else '✗ WRONG'}\n")
        
        results.append({
            "file": img_path.name,
            "actual": actual,
            "predicted": pred_class,
            "confidence": result['confidence'],
            "correct": correct
        })
    
    accuracy = sum(1 for p, a in zip(all_preds, all_labels) if p == a) / len(all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total images: {len(all_preds)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    output_file = ROOT / "training" / "testdata_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "total_images": len(all_preds),
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
