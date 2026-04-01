from __future__ import annotations

import cv2
import numpy as np


def calculate_severity_score(image_bytes: bytes) -> int:
    """Estimate damage percentage based on non-green pixels within leaf area."""
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid image file for severity scoring.")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # White background is excluded from the leaf area mask.
    _, leaf_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    healthy_lower = np.array([35, 40, 40], dtype=np.uint8)
    healthy_upper = np.array([85, 255, 255], dtype=np.uint8)
    healthy_mask = cv2.inRange(hsv, healthy_lower, healthy_upper)

    leaf_pixels = int(cv2.countNonZero(leaf_mask))
    if leaf_pixels == 0:
        return 0

    non_healthy = cv2.bitwise_not(healthy_mask)
    damaged_mask = cv2.bitwise_and(non_healthy, leaf_mask)
    damaged_pixels = int(cv2.countNonZero(damaged_mask))

    damage_pct = round((damaged_pixels / leaf_pixels) * 100.0)
    damage_pct = max(0, min(100, int(damage_pct)))
    return damage_pct
