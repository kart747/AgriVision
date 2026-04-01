# Model Performance Analysis
**AgriVision AI**

Verified on 2026-04-02 from live inference after restoring the known-good checkpoint from `origin/add-training-script`.

---

## Executive Summary

The restored model is an EfficientNet-B0 checkpoint for 16 classes covering Tomato, Apple, and Grape.

Performance is currently verified through live inference on test images with high-confidence predictions. Formal evaluation is pending a larger curated test set.

---

## Current Verified State

- Model: EfficientNet-B0
- Classes: 16
- Crops covered: Tomato, Apple, Grape
- Verification method: live backend inference on `TestData/3(1).jpeg`
- Formal metrics report: pending evaluation run

---

## Deployment Notes

- `backend/model/weights/best_model.pth` was restored from `origin/add-training-script`.
- `backend/model/weights/class_names.json` was restored from `origin/add-training-script`.
- `backend/model/predict.py` auto-detects the class count from `class_names.json` and prints the restored-checkpoint startup message.
- The old bad training logs were removed from `backend/model/weights/`.
