from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam_base64(
    model: torch.nn.Module | None,
    image_bytes: bytes,
    input_tensor: torch.Tensor,
    class_index: int,
) -> Optional[str]:
    """Generate EigenCAM overlay and return it as data URI base64 JPEG."""
    if model is None:
        return None

    try:
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None

        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        rgb_float = rgb.astype(np.float32) / 255.0

        target_layer = model.features[7]

        with EigenCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]

        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=0.5)
        visualization = np.clip(visualization, 0, 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        ok, encoded = cv2.imencode(".jpg", vis_bgr)
        if not ok:
            return None

        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[AgriVision] Grad-CAM generation failed: {exc}")
        return None
