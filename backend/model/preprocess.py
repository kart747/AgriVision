from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from fastapi import HTTPException

_IMAGE_SIZE = (224, 224)
_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD = [0.229, 0.224, 0.225]
_BLUR_THRESHOLD = 50.0


def preprocess_image(image_bytes: bytes) -> Tuple[torch.Tensor, float]:
    """Decode image bytes, run blur check, and return normalized 224x224 tensor."""
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if blur_score < _BLUR_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail="Image is unclear — please capture a closer leaf photo.",
        )

    resized = cv2.resize(bgr, _IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1)
    mean = torch.tensor(_NORMALIZE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_NORMALIZE_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0)
    return tensor, blur_score
