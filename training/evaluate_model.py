"""
Model Evaluation Script - Tests the trained model on:
1. Test split from merged_dataset (calculates F1 scores)
2. Sample images from TestData/
3. Internet images (Apple, Grape, Tomato)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "backend" / "model" / "weights"
BEST_MODEL = WEIGHTS_DIR / "best_model.pth"
CLASS_NAMES_JSON = WEIGHTS_DIR / "class_names.json"
TEST_DATA_DIR = ROOT / "TestData"
MERGED_TEST_DIR = TEST_DATA_DIR / "merged_dataset" / "test"
SAMPLE_IMAGES_DIR = TEST_DATA_DIR

INTERNET_IMAGES = {
    "apple_healthy": [
        "https://images.unsplash.com/photo-1568702846914-96b305d2uj54?w=224",
    ],
    "apple_scab": [
        "https://images.unsplash.com/photo-1598515082267-4307aed16e09?w=224",
    ],
    "grape_healthy": [
        "https://images.unsplash.com/photo-1506905925346-21bda4d3df7f?w=224",
    ],
    "tomato_healthy": [
        "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=224",
    ],
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
    print(f"Model loaded: {predictor.detected_num_classes} classes")
    return predictor


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict_image(predictor, image_path: Path) -> Dict:
    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return predictor.predict(tensor)


def test_on_merged_dataset(predictor) -> Dict:
    print("\n" + "="*60)
    print("TEST 1: Evaluating on merged_dataset test split")
    print("="*60)
    
    if not MERGED_TEST_DIR.exists():
        print(f"Test directory not found: {MERGED_TEST_DIR}")
        return {}
    
    class_names = predictor.class_names
    all_preds = []
    all_labels = []
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}
    
    transform = get_transform()
    
    for class_name in class_names:
        class_dir = MERGED_TEST_DIR / class_name
        if not class_dir.exists():
            print(f"Class directory not found: {class_dir}")
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        if not image_files:
            image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG"))
        
        for img_path in image_files:
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
                
                actual_class = class_name
                
                all_preds.append(pred_class)
                all_labels.append(actual_class)
                
                class_total[class_name] += 1
                if pred_class == actual_class:
                    class_correct[class_name] += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    if not all_preds:
        print("No predictions made!")
        return {}
    
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = sum(1 for p, a in zip(all_preds, all_labels) if p == a) / len(all_preds)
    
    print(f"\nOverall Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Total samples: {len(all_preds)}")
    
    print(f"\nPer-class accuracy:")
    for class_name in class_names:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name]
            print(f"  {class_name}: {acc:.4f} ({class_correct[class_name]}/{class_total[class_name]})")
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_samples": len(all_preds),
        "per_class": {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0 
                      for c in class_names}
    }


def test_on_sample_images(predictor) -> Dict:
    print("\n" + "="*60)
    print("TEST 2: Testing on sample images from TestData/")
    print("="*60)
    
    sample_files = list(SAMPLE_IMAGES_DIR.glob("*.jpg")) + list(SAMPLE_IMAGES_DIR.glob("*.JPG"))
    if not sample_files:
        print("No sample images found!")
        return {}
    
    results = []
    for img_path in sample_files[:5]:
        try:
            result = predict_image(predictor, img_path)
            print(f"\n{img_path.name}:")
            print(f"  Crop: {result['crop_name']}")
            print(f"  Disease: {result['disease_name']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Severity: {result['severity_label']}")
            results.append({
                "file": img_path.name,
                "crop": result['crop_name'],
                "disease": result['disease_name'],
                "confidence": result['confidence'],
                "is_healthy": result['is_healthy']
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return {"predictions": results, "total": len(results)}


def download_image(url: str) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            from io import BytesIO
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return None


def test_on_internet_images(predictor) -> Dict:
    print("\n" + "="*60)
    print("TEST 3: Testing on internet images")
    print("="*60)
    
    results = []
    transform = get_transform()
    
    for class_name, urls in INTERNET_IMAGES.items():
        print(f"\n{class_name}:")
        for url in urls[:2]:
            img = download_image(url)
            if img:
                tensor = transform(img).unsqueeze(0)
                try:
                    result = predictor.predict(tensor)
                    print(f"  URL: {url[:50]}...")
                    print(f"    Predicted: {result['crop_name']} - {result['disease_name']}")
                    print(f"    Confidence: {result['confidence']:.2f}%")
                    results.append({
                        "url": url,
                        "actual_class": class_name,
                        "predicted_crop": result['crop_name'],
                        "predicted_disease": result['disease_name'],
                        "confidence": result['confidence']
                    })
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print(f"  Failed to download")
    
    return {"predictions": results, "total": len(results)}


def main():
    print("Loading model...")
    predictor = load_model()
    
    results = {}
    
    results["merged_dataset"] = test_on_merged_dataset(predictor)
    
    results["sample_images"] = test_on_sample_images(predictor)
    
    results["internet_images"] = test_on_internet_images(predictor)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if "merged_dataset" in results and results["merged_dataset"]:
        print(f"Test Set - Accuracy: {results['merged_dataset']['accuracy']:.4f}, "
              f"Macro F1: {results['merged_dataset']['macro_f1']:.4f}")
    
    if "sample_images" in results and results["sample_images"]:
        print(f"Sample Images - Tested: {results['sample_images']['total']} images")
    
    if "internet_images" in results and results["internet_images"]:
        print(f"Internet Images - Tested: {results['internet_images']['total']} images")
    
    output_file = ROOT / "training" / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
