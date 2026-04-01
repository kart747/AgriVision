from __future__ import annotations

from io import BytesIO
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_IMAGE_SIZE = (224, 224)
_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD = [0.229, 0.224, 0.225]
_BLUR_THRESHOLD = 100.0

_transform = transforms.Compose(
    [
        transforms.Resize(_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
    ]
)


def _load_pil_image(image_input: Union[bytes, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(image_input)).convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError("Invalid image payload.") from exc

    raise TypeError("image_input must be PIL.Image.Image or bytes")


def _compute_blur_score(rgb_image: Image.Image) -> float:
    np_img = np.array(rgb_image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preprocess_image(image_input: Union[bytes, Image.Image]) -> Tuple[torch.Tensor, float]:
    """Preprocess an image for EfficientNetB0 and return tensor + blur score."""
    pil_img = _load_pil_image(image_input)
    blur_score = _compute_blur_score(pil_img)

    if blur_score < _BLUR_THRESHOLD:
        raise ValueError("Image is unclear — please capture a closer leaf photo.")

    tensor = _transform(pil_img).unsqueeze(0)
    return tensor, blur_score
