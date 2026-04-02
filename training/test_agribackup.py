"""
Test model on images in /home/pranam/Downloads/AgriBackup/ directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]  # AgriVision/
TEST_DIR = ROOT.parent  # AgriBackup/


IMAGE_LABELS = {
    "medium (1).jpg": "unknown",
    "medium.jpeg": "unknown",
    "medium.jpg": "unknown",
    "alternaria-leaf-spot-diseased-apple-treet-25564777.webp": "apple_alternaria",
    "default-92c5c2dadb77f4bcc59ca7337e48c284.webp": "unknown",
    "green-apple-leaves-free-photo.webp": "apple_healthy",
    "NDSU Agriculture Extension - Cedar Apple Rust Underside Leaf.webp": "apple_rust",
    "NDSU Agriculture Extension - Cedar Apple Rust.webp": "apple_rust",
}


def load_model():
    import sys
    backend_path = str(ROOT / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from model.predict import DiseasePredictor
    
    weights_dir = ROOT / "backend" / "model" / "weights"
    predictor = DiseasePredictor(
        weights_path=weights_dir / "best_model.pth",
        classes_path=weights_dir / "class_names.json"
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
    
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.webp", "*.WEBP"]
    image_files = []
    for ext in extensions:
        image_files.extend(TEST_DIR.glob(ext))
    
    print(f"\nFound {len(image_files)} images to test\n")
    
    results = []
    all_preds = []
    all_labels = []
    
    for img_path in sorted(image_files):
        actual = IMAGE_LABELS.get(img_path.name, "unknown")
        
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            result = predictor.predict(tensor)
            
            crop = result['crop_name'].lower()
            disease = result['disease_name'].lower().replace(' ', '_')
            
            if result['is_healthy']:
                pred_class = f"{crop}_healthy"
            else:
                pred_class = f"{crop}_{disease}"
            
            correct = pred_class == actual if actual != "unknown" else None
            all_preds.append(pred_class)
            if actual != "unknown":
                all_labels.append(actual)
            
            print(f"{img_path.name}")
            print(f"  Predicted:  {pred_class}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Crop:       {result['crop_name']}")
            print(f"  Disease:   {result['disease_name']}")
            if actual != "unknown":
                print(f"  Actual:     {actual}")
                print(f"  {'✓ CORRECT' if correct else '✗ WRONG'}")
            print()
            
            results.append({
                "file": img_path.name,
                "actual": actual,
                "predicted": pred_class,
                "confidence": result['confidence'],
                "crop": result['crop_name'],
                "disease": result['disease_name'],
            })
            
        except Exception as e:
            print(f"{img_path.name}: Error - {e}\n")
            results.append({
                "file": img_path.name,
                "error": str(e)
            })
    
    known_images = [r for r in results if "error" not in r and r["actual"] != "unknown"]
    if known_images:
        all_labels = [r["actual"] for r in known_images]
        all_preds = [r["predicted"] for r in known_images]
        
        accuracy = sum(1 for p, a in zip(all_preds, all_labels) if p == a) / len(all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        print("=" * 60)
        print("RESULTS (known labels only)")
        print("=" * 60)
        print(f"Total images: {len(known_images)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
    else:
        print("No images with known labels to evaluate.")
    
    output_file = ROOT / "training" / "agribackup_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
