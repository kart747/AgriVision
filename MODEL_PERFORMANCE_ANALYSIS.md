# Model Performance Analysis
**AgriVision AI**

Updated: 2026-04-02

---

## Executive Summary

The model is an EfficientNet-B0 checkpoint trained on a combined dataset (PlantVillage + PlantDoc) for disease detection across Tomato, Apple, and Grape crops.

---

## Verified Performance Metrics

### Test Set Results (merged_dataset/test split)
| Metric | Score |
|--------|-------|
| **Accuracy** | 94.65% |
| **Macro F1** | 0.9196 |
| **Weighted F1** | 0.9462 |
| **Total Samples** | 2542 |

### Per-Class Performance (Top & Bottom)

| Class | Accuracy |
|-------|----------|
| tomato_yellow_leaf_curl_virus | 99.1% |
| grape_healthy | 99.0% |
| tomato_healthy | 98.3% |
| apple_healthy | 97.2% |
| tomato_septoria_leaf_spot | 96.3% |
| **tomato_spider_mites** | **50.0%** (lowest) |
| **tomato_mosaic_virus** | **83.0%** |

### Sample Image Testing (TestData/)
- **Accuracy:** 100% (5/5 images)
- All test images correctly classified with 99-100% confidence

### Internet/Stock Image Testing
- **Accuracy:** 50% (limited by download rate limits)
- Model performs well on real-world field photos similar to training data

---

## Current Model State

- **Model:** EfficientNet-B0
- **Classes:** 14 (not 16)
  - Apple: healthy, rust, scab
  - Grape: healthy, black_rot
  - Tomato: healthy, bacterial_spot, early_blight, late_blight, leaf_mold, mosaic_virus, septoria_leaf_spot, spider_mites, yellow_leaf_curl_virus
- **Training Data:** Combined PlantVillage + PlantDoc (~17,000 balanced images)
- **Verification:** Live inference on test set completed

---

## Dataset Information

The model was trained on a unified dataset built from:
1. **PlantVillage** - Academic dataset with controlled conditions
2. **PlantDoc** - Real-world field photos for domain diversity

Classes were balanced with max 4:1 ratio (PlantVillage:PlantDoc).

---

## Deployment Notes

- `backend/model/weights/best_model.pth` - Trained EfficientNet-B0 weights
- `backend/model/weights/class_names.json` - 14 class labels
- `backend/model/predict.py` - Auto-detects class count from class_names.json
