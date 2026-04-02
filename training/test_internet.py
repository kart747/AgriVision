"""
Download test images from Wikimedia Commons and test the model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from PIL import Image
from sklearn.metrics import classification_report, f1_score


TEST_IMAGES = {
    "apple_scab": [
        "https://upload.wikimedia.org/wikipedia/commons/1/15/Apple_scab_SEM.jpg",
    ],
    "tomato_late_blight": [
        "https://upload.wikimedia.org/wikipedia/commons/e/eb/Late_blight_on_tomato_leaf_%287871756748%29.jpg",
    ],
    "tomato_septoria_leaf_spot": [
        "https://upload.wikimedia.org/wikipedia/commons/6/63/Septoria_leaf_spot_on_tomato.jpg",
    ],
    "tomato_early_blight": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Tomato_early_blight_on_leaf.jpg/800px-Tomato_early_blight_on_leaf.jpg",
    ],
    "grape_black_rot": [
        "https://upload.wikimedia.org/wikipedia/commons/2/29/Guignardia_bidwellii_%28black_rot%29_on_grape_1.jpg",
    ],
    "grape_healthy": [
        "https://upload.wikimedia.org/wikipedia/commons/3/3b/Grape_leaf.jpg",
    ],
}


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "TestData" / "internet_test"
WEIGHTS_DIR = ROOT / "backend" / "model" / "weights"


def download_image(url: str, save_path: Path) -> bool:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        time.sleep(1.5)
        response = requests.get(url, timeout=20, headers=headers)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False


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
    import torch
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    predictor = load_model()
    transform = get_transform()
    
    downloaded = []
    
    print("\nDownloading test images from Wikimedia Commons...")
    for class_name, urls in TEST_IMAGES.items():
        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls):
            filename = f"{class_name}_{i+1}.jpg"
            save_path = class_dir / filename
            
            print(f"  Downloading {url[:60]}...")
            if download_image(url, save_path):
                print(f"    Saved to {save_path}")
                downloaded.append((class_name, save_path))
            else:
                print(f"    Failed!")
    
    print(f"\nDownloaded {len(downloaded)} images")
    
    print("\n" + "="*60)
    print("Testing on internet images")
    print("="*60)
    
    all_preds = []
    all_labels = []
    results = []
    
    for actual_class, img_path in downloaded:
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
            
            all_preds.append(pred_class)
            all_labels.append(actual_class)
            
            correct = pred_class == actual_class
            print(f"\n{img_path.name}:")
            print(f"  Actual: {actual_class}")
            print(f"  Predicted: {pred_class}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  {'✓ CORRECT' if correct else '✗ WRONG'}")
            
            results.append({
                "file": img_path.name,
                "actual": actual_class,
                "predicted": pred_class,
                "confidence": result['confidence'],
                "correct": correct
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if all_preds:
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        accuracy = sum(1 for p, a in zip(all_preds, all_labels) if p == a) / len(all_preds)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Total images: {len(all_preds)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        
        output_file = ROOT / "training" / "internet_test_results.json"
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
